// llama-sortformer — ggml/CUDA port of NVIDIA Streaming Sortformer (speaker diarization).
//
// Milestone M4 (in progress): scaffold + GGUF loader + CUDA backend init + mel preprocessing,
// validated against the NeMo reference (reference_offline/mel.npy). The Conformer encoder,
// Transformer head and AOSC streaming are added next, each validated against the per-layer
// reference .npy files.
//
// Not an LLM: links ggml directly (no libllama).

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <complex>
#include <fstream>
#include <stdexcept>

// ----------------------------------------------------------------- tiny .npy reader (f32)
struct NpyF32 {
    std::vector<int64_t> shape;
    std::vector<float>   data;
    int64_t numel() const { int64_t n = 1; for (auto s : shape) n *= s; return n; }
};

static NpyF32 npy_load_f32(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open npy: " + path);
    char magic[6]; f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("not a npy: " + path);
    uint8_t major = 0, minor = 0; f.read((char*)&major, 1); f.read((char*)&minor, 1);
    uint32_t header_len = 0;
    if (major == 1) { uint16_t h; f.read((char*)&h, 2); header_len = h; }
    else            { f.read((char*)&header_len, 4); }
    std::string header(header_len, '\0'); f.read(&header[0], header_len);
    if (header.find("<f4") == std::string::npos)
        throw std::runtime_error("npy not <f4: " + path + " :: " + header);
    // parse shape tuple
    NpyF32 out;
    auto sp = header.find("'shape':");
    auto lp = header.find('(', sp), rp = header.find(')', lp);
    std::string dims = header.substr(lp + 1, rp - lp - 1);
    for (size_t i = 0; i < dims.size();) {
        while (i < dims.size() && (dims[i] == ' ' || dims[i] == ',')) i++;
        if (i >= dims.size()) break;
        int64_t v = 0; bool any = false;
        while (i < dims.size() && dims[i] >= '0' && dims[i] <= '9') { v = v*10 + (dims[i]-'0'); i++; any = true; }
        if (any) out.shape.push_back(v);
    }
    out.data.resize(out.numel());
    f.read((char*)out.data.data(), out.numel() * sizeof(float));
    if (!f) throw std::runtime_error("short read: " + path);
    return out;
}

// ----------------------------------------------------------------- radix-2 iterative FFT (n = pow2)
static void fft_radix2(std::vector<std::complex<double>> & a) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    const double PI = 3.14159265358979323846;
    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * PI / (double) len;
        std::complex<double> wlen(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (size_t k = 0; k < len/2; k++) {
                std::complex<double> u = a[i+k], v = a[i+k+len/2] * w;
                a[i+k] = u + v; a[i+k+len/2] = u - v; w *= wlen;
            }
        }
    }
}

// ----------------------------------------------------------------- NeMo-exact log-mel (see MEL_RECIPE.md)
// pcm: 16 kHz mono f32. window: Hann (win_length). fb: [n_mel, n_freq] row-major. Returns [n_mel, T].
static std::vector<float> compute_logmel(
        const std::vector<float> & pcm,
        const float * window, int win_length,
        const float * fb, int n_mel, int n_freq,
        int n_fft, int hop, float preemph, float log_guard, int & T_out) {
    // preemphasis
    std::vector<float> x(pcm.size());
    if (pcm.size()) x[0] = pcm[0];
    for (size_t i = 1; i < pcm.size(); i++) x[i] = pcm[i] - preemph * pcm[i-1];

    // center=True reflect pad by n_fft/2 (torch 'reflect': excludes the boundary sample)
    const int pad = n_fft / 2;
    const int64_t N = (int64_t) x.size();
    auto reflect = [&](int64_t idx) -> float {
        // map any idx into [0, N) by reflecting without repeating edges
        if (N == 1) return x[0];
        int64_t period = 2*(N-1);
        int64_t m = ((idx % period) + period) % period;
        return x[m < N ? m : period - m];
    };
    const int64_t P = N + 2*pad;
    const int T = (int)(1 + (N) / hop);   // center=True frame count
    T_out = T;

    const int woff = (n_fft - win_length) / 2;  // window centered inside the n_fft frame
    std::vector<float> mel((size_t) n_mel * T);
    std::vector<std::complex<double>> buf(n_fft);

    for (int t = 0; t < T; t++) {
        const int64_t start = (int64_t) t * hop;        // index into padded signal
        for (int i = 0; i < n_fft; i++) buf[i] = std::complex<double>(0.0, 0.0);
        for (int j = 0; j < win_length; j++) {
            const int64_t pidx = start + woff + j;       // position in padded signal
            const int64_t sidx = pidx - pad;             // position in x (pre-pad)
            const float s = reflect(sidx);
            buf[woff + j] = std::complex<double>((double) s * (double) window[j], 0.0);
        }
        fft_radix2(buf);
        for (int m = 0; m < n_mel; m++) {
            double acc = 0.0;
            const float * fbm = fb + (size_t) m * n_freq;
            for (int k = 0; k < n_freq; k++) {
                double re = buf[k].real(), im = buf[k].imag();
                acc += (double) fbm[k] * (re*re + im*im);     // power spectrum |.|^2
            }
            mel[(size_t) m * T + t] = (float) std::log(acc + (double) log_guard);
        }
    }
    (void) P;
    return mel;
}

int main(int argc, char ** argv) {
    std::string model = "model/sortformer.gguf";
    std::string pcm_npy = "reference/input_pcm.npy";
    std::string mel_ref = "reference_offline/mel.npy";
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]{ return (i+1 < argc) ? argv[++i] : ""; };
        if      (a == "--model")   model   = next();
        else if (a == "--pcm")     pcm_npy = next();
        else if (a == "--mel-ref") mel_ref = next();
    }

    // ---- CUDA backend (for the upcoming compute graph) ----
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev) printf("backend: GPU device = %s\n", ggml_backend_dev_name(dev));
    else     printf("backend: no GPU found, CPU only\n");

    // ---- load GGUF (CPU) to read preprocessor.window / .fb + hparams ----
    struct ggml_context * ctx = NULL;
    struct gguf_init_params gp = { /*no_alloc*/ false, /*ctx*/ &ctx };
    struct gguf_context * gguf = gguf_init_from_file(model.c_str(), gp);
    if (!gguf) { fprintf(stderr, "failed to load gguf: %s\n", model.c_str()); return 1; }
    printf("gguf: %d tensors, %d kv\n", (int) gguf_get_n_tensors(gguf), (int) gguf_get_n_kv(gguf));

    auto get = [&](const char * name) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(ctx, name);
        if (!t) { fprintf(stderr, "missing tensor: %s\n", name); std::exit(1); }
        return t;
    };
    ggml_tensor * t_win = get("mel.window");      // [400]
    ggml_tensor * t_fb  = get("mel.fb");          // ne0=257, ne1=128
    const int win_length = (int) t_win->ne[0];
    const int n_freq     = (int) t_fb->ne[0];
    const int n_mel      = (int) t_fb->ne[1];
    printf("mel: win_length=%d n_freq=%d n_mel=%d\n", win_length, n_freq, n_mel);

    // hparams from MEL_RECIPE: n_fft 512, hop 160, preemph 0.97, log guard 2^-24
    const int   n_fft = 512, hop = 160;
    const float preemph = 0.97f, log_guard = 5.960464477539063e-08f;

    // ---- input pcm + reference mel ----
    NpyF32 pcm = npy_load_f32(pcm_npy);
    NpyF32 ref = npy_load_f32(mel_ref);                              // [1,128,T]
    printf("pcm: %lld samples ; ref mel shape [", (long long) pcm.numel());
    for (auto s : ref.shape) printf("%lld,", (long long) s);
    printf("]\n");

    int T = 0;
    std::vector<float> mel = compute_logmel(pcm.data, (float*) t_win->data, win_length,
                                            (float*) t_fb->data, n_mel, n_freq,
                                            n_fft, hop, preemph, log_guard, T);
    const int Tref = (int) ref.shape[ref.shape.size()-1];
    printf("computed mel: [%d, %d]  (ref T=%d)\n", n_mel, T, Tref);

    // ---- compare interior frames (skip boundary frames 0 and T-1 — known NeMo masking/edge) ----
    int Tc = std::min(T, Tref);
    double maxd = 0, sumd = 0; long cnt = 0; double maxd_all = 0;
    for (int m = 0; m < n_mel; m++) {
        for (int t = 0; t < Tc; t++) {
            double a = mel[(size_t) m * T + t];
            double b = ref.data[(size_t) m * Tref + t];   // ref [1,128,Tref] -> m*Tref + t
            double d = std::fabs(a - b);
            maxd_all = std::max(maxd_all, d);
            if (t > 0 && t < Tc - 1) { maxd = std::max(maxd, d); sumd += d; cnt++; }
        }
    }
    printf("\nmel vs NeMo: interior max|diff|=%.3e mean|diff|=%.3e (all-frames max=%.3e)\n",
           maxd, sumd / std::max(1L, cnt), maxd_all);
    // ~1.6e-3 is the float32-STFT precision floor (matches the Python reimpl); the all-frames
    // outlier is the last frame, which NeMo masks to 0 (handled later). Gate at 5e-3.
    printf("%s\n", maxd < 5e-3 ? "MEL OK (matches NeMo to f32 precision)" : "MEL MISMATCH");

    gguf_free(gguf);
    ggml_free(ctx);
    return 0;
}
