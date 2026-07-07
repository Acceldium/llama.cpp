// tools/tts/silero_vad.cpp
#include "silero_vad.h"

#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

namespace {

constexpr int VAD_MAX_NODES = 2048;
constexpr int VAD_HOP       = 128; // stft hop @ 16kHz
constexpr int VAD_NFFT      = 256; // stft window @ 16kHz
constexpr int VAD_PAD       = (VAD_NFFT - VAD_HOP) / 4; // 32 - verified empirically against the ONNX graph's actual Pad node (640 padded length for a 576-sample window, not 704)
constexpr int VAD_HIDDEN    = 128;

// Minimal GGUF tensor loader, mirroring qwen3tts-lib.cpp's gguf_tensor_loader
// (kept separate/self-contained here on purpose - this file has no other
// dependency on qwen3tts-lib.cpp and shouldn't gain one just for this).
struct tensor_loader {
    ggml_context * ctx = nullptr;
    gguf_context * guf = nullptr;
    std::map<std::string, ggml_tensor *> tensors;

    ~tensor_loader() {
        if (guf) gguf_free(guf);
        if (ctx) ggml_free(ctx);
    }

    bool load(const char * path) {
        gguf_init_params params;
        params.no_alloc = false;
        params.ctx = &ctx;
        guf = gguf_init_from_file(path, params);
        if (!guf) {
            fprintf(stderr, "ERROR: cannot open GGUF: %s\n", path);
            return false;
        }
        int64_t n = gguf_get_n_tensors(guf);
        for (int64_t i = 0; i < n; i++) {
            const char * name = gguf_get_tensor_name(guf, i);
            ggml_tensor * t = ggml_get_tensor(ctx, name);
            if (t) tensors[name] = t;
        }
        return true;
    }

    ggml_tensor * get(const char * name) const {
        auto it = tensors.find(name);
        return it != tensors.end() ? it->second : nullptr;
    }
};

ggml_tensor * conv1d_bias(ggml_context * ctx0, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x,
                           int stride, int pad, int dilation) {
    if (w->type != GGML_TYPE_F16) w = ggml_cast(ctx0, w, GGML_TYPE_F16);
    ggml_tensor * y = ggml_conv_1d(ctx0, w, x, stride, pad, dilation);
    if (b) {
        int64_t oc = y->ne[1];
        y = ggml_add(ctx0, y, ggml_reshape_3d(ctx0, b, 1, oc, 1));
    }
    return y;
}

} // namespace

struct silero_vad::impl {
    tensor_loader loader;

    ggml_tensor * stft_basis = nullptr;
    ggml_tensor * enc_w[4] = {};
    ggml_tensor * enc_b[4] = {};
    ggml_tensor * rnn_w_ih = nullptr;
    ggml_tensor * rnn_w_hh = nullptr;
    ggml_tensor * rnn_b_ih = nullptr;
    ggml_tensor * rnn_b_hh = nullptr;
    ggml_tensor * decoder_w = nullptr; // reshaped to 2D [128,1] at load time
    ggml_tensor * decoder_b = nullptr;

    std::vector<float> h_state = std::vector<float>(VAD_HIDDEN, 0.0f);
    std::vector<float> c_state = std::vector<float>(VAD_HIDDEN, 0.0f);
    std::vector<float> context = std::vector<float>(SILERO_VAD_CONTEXT_SAMPLES, 0.0f);
};

silero_vad::silero_vad() : pimpl(std::make_unique<impl>()) {}
silero_vad::~silero_vad() = default;

bool silero_vad::load(const std::string & path) {
    if (!pimpl->loader.load(path.c_str())) {
        return false;
    }
    auto & m = *pimpl;
    auto get = [&](const char * name) -> ggml_tensor * {
        ggml_tensor * t = m.loader.get(name);
        if (!t) fprintf(stderr, "WARN: silero_vad missing tensor %s\n", name);
        return t;
    };

    m.stft_basis = get("silero_vad.stft.basis");
    for (int i = 0; i < 4; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "silero_vad.encoder.%d.weight", i); m.enc_w[i] = get(buf);
        snprintf(buf, sizeof(buf), "silero_vad.encoder.%d.bias", i);   m.enc_b[i] = get(buf);
    }
    m.rnn_w_ih = get("silero_vad.rnn.weight_ih");
    m.rnn_w_hh = get("silero_vad.rnn.weight_hh");
    m.rnn_b_ih = get("silero_vad.rnn.bias_ih");
    m.rnn_b_hh = get("silero_vad.rnn.bias_hh");
    m.decoder_b = get("silero_vad.decoder.bias");
    m.decoder_w = get("silero_vad.decoder.weight"); // ne=[1,128,1] (kernel,in,out) - reshaped to 2D per-call in process_chunk

    bool ok = m.stft_basis && m.rnn_w_ih && m.rnn_w_hh && m.rnn_b_ih && m.rnn_b_hh &&
              m.decoder_w && m.decoder_b;
    for (int i = 0; i < 4; i++) ok = ok && m.enc_w[i] && m.enc_b[i];
    if (!ok) {
        fprintf(stderr, "ERROR: silero_vad GGUF is missing required tensors\n");
    }
    return ok;
}

void silero_vad::reset_states() {
    std::fill(pimpl->h_state.begin(), pimpl->h_state.end(), 0.0f);
    std::fill(pimpl->c_state.begin(), pimpl->c_state.end(), 0.0f);
    std::fill(pimpl->context.begin(), pimpl->context.end(), 0.0f);
}

float silero_vad::process_chunk(const float * chunk, int n) {
    if (n != SILERO_VAD_CHUNK_SAMPLES) {
        fprintf(stderr, "ERROR: silero_vad::process_chunk expects exactly %d samples, got %d\n",
                SILERO_VAD_CHUNK_SAMPLES, n);
        return 0.0f;
    }
    auto & m = *pimpl;

    // context (64) + new chunk (512) = 576, then reflect-pad by VAD_PAD (64) each side for the STFT conv.
    std::vector<float> windowed(SILERO_VAD_CONTEXT_SAMPLES + SILERO_VAD_CHUNK_SAMPLES);
    std::copy(m.context.begin(), m.context.end(), windowed.begin());
    std::copy(chunk, chunk + n, windowed.begin() + SILERO_VAD_CONTEXT_SAMPLES);

    const int in_len = (int)windowed.size(); // 576
    const int padded_len = in_len + 2 * VAD_PAD; // 704
    std::vector<float> padded(padded_len);
    for (int i = 0; i < padded_len; i++) {
        int src = i - VAD_PAD;
        if (src < 0) src = -src;                       // reflect left
        else if (src >= in_len) src = 2 * in_len - src - 2; // reflect right
        src = std::max(0, std::min(in_len - 1, src));
        padded[i] = windowed[src];
    }

    size_t ctx_size = ggml_tensor_overhead() * VAD_MAX_NODES + 256 * 1024 * 1024;
    ggml_init_params ctx_params = { ctx_size, nullptr, true };
    ggml_context * ctx0 = ggml_init(ctx_params);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, VAD_MAX_NODES, false);

    ggml_tensor * wav = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, padded_len);
    ggml_set_name(wav, "wav_input"); ggml_set_input(wav);
    ggml_tensor * cur = ggml_reshape_3d(ctx0, wav, padded_len, 1, 1);

    cur = conv1d_bias(ctx0, m.stft_basis, nullptr, cur, VAD_HOP, 0, 1); // [n_frames, 258, 1]
    int64_t n_frames = cur->ne[0];

    ggml_tensor * real = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, n_frames, 129, 1, cur->nb[1], cur->nb[2], 0));
    ggml_tensor * imag = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, n_frames, 129, 1, cur->nb[1], cur->nb[2],
                                                       129 * cur->nb[1]));
    ggml_tensor * mag = ggml_sqrt(ctx0, ggml_add(ctx0, ggml_sqr(ctx0, real), ggml_sqr(ctx0, imag)));

    cur = mag; // [n_frames, 129, 1]
    const int strides[4] = {1, 2, 2, 1};
    for (int i = 0; i < 4; i++) {
        cur = ggml_pad_ext(ctx0, cur, 1, 1, 0, 0, 0, 0, 0, 0); // kernel=3, pad=1 both sides
        cur = conv1d_bias(ctx0, m.enc_w[i], m.enc_b[i], cur, strides[i], 0, 1);
        cur = ggml_relu(ctx0, cur);
    }
    // cur: [1, 128, 1] - exactly one timestep for a 512-sample chunk + 64-sample context

    ggml_tensor * x_t = ggml_reshape_1d(ctx0, cur, VAD_HIDDEN);

    ggml_tensor * h_prev = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, VAD_HIDDEN);
    ggml_set_name(h_prev, "h_prev"); ggml_set_input(h_prev);
    ggml_tensor * c_prev = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, VAD_HIDDEN);
    ggml_set_name(c_prev, "c_prev"); ggml_set_input(c_prev);

    ggml_tensor * gates = ggml_add(ctx0,
        ggml_add(ctx0, ggml_mul_mat(ctx0, m.rnn_w_ih, x_t), m.rnn_b_ih),
        ggml_add(ctx0, ggml_mul_mat(ctx0, m.rnn_w_hh, h_prev), m.rnn_b_hh));
    // gates: [512] = concat(i, f, g, o), matching torch's LSTMCell gate order

    const size_t es = ggml_element_size(gates);
    ggml_tensor * i_gate = ggml_sigmoid(ctx0, ggml_cont(ctx0, ggml_view_1d(ctx0, gates, VAD_HIDDEN, 0 * VAD_HIDDEN * es)));
    ggml_tensor * f_gate = ggml_sigmoid(ctx0, ggml_cont(ctx0, ggml_view_1d(ctx0, gates, VAD_HIDDEN, 1 * VAD_HIDDEN * es)));
    ggml_tensor * g_gate = ggml_tanh   (ctx0, ggml_cont(ctx0, ggml_view_1d(ctx0, gates, VAD_HIDDEN, 2 * VAD_HIDDEN * es)));
    ggml_tensor * o_gate = ggml_sigmoid(ctx0, ggml_cont(ctx0, ggml_view_1d(ctx0, gates, VAD_HIDDEN, 3 * VAD_HIDDEN * es)));

    ggml_tensor * c_new = ggml_add(ctx0, ggml_mul(ctx0, f_gate, c_prev), ggml_mul(ctx0, i_gate, g_gate));
    ggml_tensor * h_new = ggml_mul(ctx0, o_gate, ggml_tanh(ctx0, c_new));
    ggml_set_name(h_new, "h_new");
    ggml_set_name(c_new, "c_new");

    ggml_tensor * decoder_w_2d = ggml_reshape_2d(ctx0, m.decoder_w, m.decoder_w->ne[1], m.decoder_w->ne[2]);
    ggml_tensor * dec = ggml_relu(ctx0, h_new);
    ggml_tensor * prob = ggml_add(ctx0, ggml_mul_mat(ctx0, decoder_w_2d, dec), m.decoder_b);
    prob = ggml_sigmoid(ctx0, prob);
    ggml_set_name(prob, "prob");

    ggml_build_forward_expand(gf, prob);
    ggml_build_forward_expand(gf, h_new);
    ggml_build_forward_expand(gf, c_new);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    ggml_backend_tensor_set(wav, padded.data(), 0, padded.size() * sizeof(float));
    ggml_backend_tensor_set(h_prev, m.h_state.data(), 0, m.h_state.size() * sizeof(float));
    ggml_backend_tensor_set(c_prev, m.c_state.data(), 0, m.c_state.size() * sizeof(float));

    ggml_backend_graph_compute(backend, gf);

    float prob_out = 0.0f;
    ggml_backend_tensor_get(prob, &prob_out, 0, sizeof(float));
    ggml_backend_tensor_get(h_new, m.h_state.data(), 0, m.h_state.size() * sizeof(float));
    ggml_backend_tensor_get(c_new, m.c_state.data(), 0, m.c_state.size() * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_backend_free(backend);
    ggml_free(ctx0);

    std::copy(chunk + n - SILERO_VAD_CONTEXT_SAMPLES, chunk + n, m.context.begin());

    return prob_out;
}
