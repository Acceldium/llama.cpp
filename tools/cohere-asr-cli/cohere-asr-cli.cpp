// Standalone CLI for the cohere-asr port: decodes an audio file (WAV/MP3/FLAC
// via miniaudio), computes the log-mel recipe used by the HF reference
// (preemphasis 0.97, STFT n_fft=512/hop=160/win=400 center-reflect-pad, mel
// filterbank + window read directly from the GGUF, per-feature normalization
// with biased std), runs the Conformer encoder via llama_encode(), then
// greedily decodes via llama_decode() using the cohere-asr control-token
// prompt for the requested language.
//
// Clips longer than --max-clip-seconds (default 35s, matching the original
// model's own max_audio_clip_s) are split into energy/silence-boundary
// chunks -- same split_audio_chunks_energy() algorithm as
// modeling_cohere_asr.py's CohereAsrForConditionalGeneration.transcribe() --
// transcribed independently, and rejoined. The encoder is dense/non-causal
// attention with no windowing, so a single-shot pass over very long audio
// would be both slow (O(n^2) in frame count) and memory-heavy; chunking
// keeps each pass bounded to --max-clip-seconds worth of audio.
//
// Deliberately NOT integrated into llama-mtmd-cli/libmtmd: this model needs a
// real llama_encode() + cross-attention llama_decode() split, which libmtmd's
// embedding-splice-into-one-decoder design doesn't support (same reason T5
// has no mtmd-cli integration either). An optional --dump-dir flag captures
// intermediate tensors as .npy for validation against cohere/reference_dump.py
// (only for the first chunk, if the clip is long enough to need more than one).

// fix problem with std::min and std::max (miniaudio pulls in windows.h)
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "llama.h"
#include "gguf.h"
#include "ggml.h"

#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_ENCODING
#define MA_NO_DEVICE_IO
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_API static
#include "miniaudio/miniaudio.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

static constexpr float PI_F = 3.14159265358979323846f;

// supported per config.json's "supported_languages"
static const std::set<std::string> SUPPORTED_LANGS = {
    "en", "fr", "de", "es", "it", "pt", "nl", "pl", "el", "ar", "ja", "zh", "vi", "ko",
};
static const std::set<std::string> NO_SPACE_LANGS = {"ja", "zh"};

// matches CohereAsrConfig defaults in configuration_cohere_asr.py
static constexpr float OVERLAP_CHUNK_SECONDS = 5.0f;
static constexpr int64_t MIN_ENERGY_WINDOW_SAMPLES = 1600;

static bool decode_audio_file(const std::string & path, int target_sample_rate, std::vector<float> & pcm_mono) {
    ma_decoder_config cfg = ma_decoder_config_init(ma_format_f32, 1, (ma_uint32) target_sample_rate);
    ma_decoder decoder;
    if (ma_decoder_init_file(path.c_str(), &cfg, &decoder) != MA_SUCCESS) {
        fprintf(stderr, "failed to open/decode audio file: %s\n", path.c_str());
        return false;
    }

    ma_uint64 frame_count = 0;
    if (ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count) != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        return false;
    }

    pcm_mono.resize(frame_count);
    ma_uint64 frames_read = 0;
    ma_result result = ma_decoder_read_pcm_frames(&decoder, pcm_mono.data(), frame_count, &frames_read);
    ma_decoder_uninit(&decoder);
    if (result != MA_SUCCESS) {
        fprintf(stderr, "failed to read PCM frames from: %s\n", path.c_str());
        return false;
    }
    pcm_mono.resize(frames_read);
    return true;
}

// simple recursive radix-2 FFT (n_fft is always a power of 2 here)
static void fft(std::vector<std::complex<float>> & a) {
    const size_t n = a.size();
    if (n <= 1) return;
    std::vector<std::complex<float>> even(n / 2), odd(n / 2);
    for (size_t i = 0; i < n / 2; ++i) {
        even[i] = a[2 * i];
        odd[i]  = a[2 * i + 1];
    }
    fft(even);
    fft(odd);
    for (size_t k = 0; k < n / 2; ++k) {
        float ang = -2.0f * PI_F * (float) k / (float) n;
        std::complex<float> t = std::polar(1.0f, ang) * odd[k];
        a[k]         = even[k] + t;
        a[k + n / 2] = even[k] - t;
    }
}

// mel_fb: [n_fft_bins, n_mel] row-major (ggml ne0=n_fft_bins, ne1=n_mel)
// mel_window: [win_length]
static std::vector<float> compute_logmel(
        const std::vector<float> & wav,
        const std::vector<float> & mel_fb, int64_t n_fft_bins, int64_t n_mel,
        const std::vector<float> & window,
        int n_fft, int hop, int win_length) {
    const float preemph = 0.97f;
    std::vector<float> x(wav.size());
    x[0] = wav[0];
    for (size_t i = 1; i < wav.size(); ++i) {
        x[i] = wav[i] - preemph * wav[i - 1];
    }

    // reflect-pad by n_fft/2 on both sides (matches torch.stft center=True)
    const int pad = n_fft / 2;
    std::vector<float> padded((size_t) pad * 2 + x.size());
    for (int i = 0; i < pad; ++i) {
        padded[pad - 1 - i] = x[std::min((size_t) i + 1, x.size() - 1)];
    }
    for (size_t i = 0; i < x.size(); ++i) {
        padded[pad + i] = x[i];
    }
    for (int i = 0; i < pad; ++i) {
        padded[pad + x.size() + i] = x[x.size() - 2 - std::min((size_t) i, x.size() - 2)];
    }

    const int64_t n_frames = (int64_t) ((padded.size() - n_fft) / hop) + 1;
    std::vector<float> logmel((size_t) n_mel * n_frames, 0.0f);

    // torch.stft centers a shorter window within the n_fft-sample frame: the
    // window is applied to frame-relative samples [win_offset, win_offset +
    // win_length), not to the first win_length samples of the frame.
    const int win_offset = (n_fft - win_length) / 2;
    std::vector<float> frame(n_fft, 0.0f);
    for (int64_t t = 0; t < n_frames; ++t) {
        std::fill(frame.begin(), frame.end(), 0.0f);
        for (int i = 0; i < win_length; ++i) {
            frame[win_offset + i] = padded[t * hop + win_offset + i] * window[i];
        }
        std::vector<std::complex<float>> spec(n_fft);
        for (int i = 0; i < n_fft; ++i) spec[i] = frame[i];
        fft(spec);

        for (int64_t m = 0; m < n_mel; ++m) {
            double acc = 0.0;
            for (int64_t k = 0; k < n_fft_bins; ++k) {
                float power = spec[k].real() * spec[k].real() + spec[k].imag() * spec[k].imag();
                acc += (double) mel_fb[(size_t) m * n_fft_bins + k] * power;
            }
            logmel[(size_t) m * n_frames + t] = (float) std::log(acc + std::pow(2.0, -24));
        }
    }

    // per-feature normalization: zero-mean/unit-var per mel bin across time,
    // biased (population) std, eps 1e-5
    for (int64_t m = 0; m < n_mel; ++m) {
        double mean = 0.0;
        for (int64_t t = 0; t < n_frames; ++t) mean += logmel[(size_t) m * n_frames + t];
        mean /= (double) n_frames;
        double var = 0.0;
        for (int64_t t = 0; t < n_frames; ++t) {
            double d = logmel[(size_t) m * n_frames + t] - mean;
            var += d * d;
        }
        var /= (double) n_frames;
        const double std_ = std::sqrt(var);
        for (int64_t t = 0; t < n_frames; ++t) {
            logmel[(size_t) m * n_frames + t] = (float) ((logmel[(size_t) m * n_frames + t] - mean) / (std_ + 1e-5));
        }
    }

    return logmel; // [n_mel, n_frames] row-major, mel-major
}

struct AudioSegment {
    int64_t start;
    int64_t end;
};

// Port of _find_split_point_energy() in modeling_cohere_asr.py: finds the
// quietest MIN_ENERGY_WINDOW_SAMPLES-sized window (lowest RMS energy) within
// [start_idx, end_idx) and returns its start index -- a good place to cut
// without slicing through speech.
static int64_t find_split_point_energy(const std::vector<float> & waveform, int64_t start_idx, int64_t end_idx) {
    const int64_t seg_len = end_idx - start_idx;
    if (seg_len <= MIN_ENERGY_WINDOW_SAMPLES) {
        return (start_idx + end_idx) / 2;
    }

    double min_energy = std::numeric_limits<double>::infinity();
    int64_t quietest_idx = start_idx;
    const int64_t upper = seg_len - MIN_ENERGY_WINDOW_SAMPLES;
    for (int64_t i = 0; i < upper; i += MIN_ENERGY_WINDOW_SAMPLES) {
        double sum_sq = 0.0;
        for (int64_t j = 0; j < MIN_ENERGY_WINDOW_SAMPLES; ++j) {
            const float v = waveform[start_idx + i + j];
            sum_sq += (double) v * v;
        }
        const double energy = std::sqrt(sum_sq / (double) MIN_ENERGY_WINDOW_SAMPLES);
        if (energy < min_energy) {
            min_energy = energy;
            quietest_idx = start_idx + i;
        }
    }
    return quietest_idx;
}

// Port of split_audio_chunks_energy() in modeling_cohere_asr.py: splits audio
// longer than max_clip_seconds into chunks, choosing each cut point at the
// quietest moment within the last overlap_chunk_seconds before the cutoff
// (not literal overlapping audio -- a search window for a good silence point).
static std::vector<AudioSegment> split_audio_chunks_energy(
        const std::vector<float> & waveform, int sample_rate, float max_clip_seconds) {
    const int64_t chunk_size = std::max<int64_t>(1, (int64_t) std::llround((double) max_clip_seconds * sample_rate));
    const int64_t boundary_context = std::max<int64_t>(1, (int64_t) std::llround((double) OVERLAP_CHUNK_SECONDS * sample_rate));
    const int64_t total_samples = (int64_t) waveform.size();

    std::vector<AudioSegment> segments;
    if (total_samples <= chunk_size) {
        segments.push_back({0, total_samples});
        return segments;
    }

    int64_t idx = 0;
    while (idx < total_samples) {
        if (idx + chunk_size >= total_samples) {
            segments.push_back({idx, total_samples});
            break;
        }

        const int64_t search_start = std::max(idx, idx + chunk_size - boundary_context);
        const int64_t search_end = std::min(idx + chunk_size, total_samples);

        int64_t split_point;
        if (search_end <= search_start) {
            split_point = idx + chunk_size;
        } else {
            split_point = find_split_point_energy(waveform, search_start, search_end);
        }
        split_point = std::max(idx + 1, std::min(split_point, total_samples));

        segments.push_back({idx, split_point});
        idx = split_point;
    }
    return segments;
}

// reads a small F32 tensor's raw data directly out of the GGUF file, without
// loading the whole (multi-GB) file into memory a second time
static bool read_gguf_f32_tensor(const std::string & gguf_path, const std::string & name, std::vector<float> & out) {
    struct gguf_init_params params = { true, nullptr };
    struct gguf_context * ctx = gguf_init_from_file(gguf_path.c_str(), params);
    if (!ctx) return false;

    int64_t idx = gguf_find_tensor(ctx, name.c_str());
    if (idx < 0) {
        gguf_free(ctx);
        return false;
    }
    if (gguf_get_tensor_type(ctx, idx) != GGML_TYPE_F32) {
        fprintf(stderr, "tensor %s is not F32 (convert with --outtype f32)\n", name.c_str());
        gguf_free(ctx);
        return false;
    }

    const size_t offset = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, idx);
    const size_t size   = gguf_get_tensor_size(ctx, idx);
    gguf_free(ctx);

    out.resize(size / sizeof(float));
    std::ifstream f(gguf_path, std::ios::binary);
    f.seekg((std::streamoff) offset, std::ios::beg);
    f.read((char *) out.data(), (std::streamsize) size);
    return (bool) f;
}

// minimal NPY v1.0 writer for F32 data. `ggml_shape` is given in ggml's ne[]
// order (ne0 fastest); it is written reversed as the numpy shape, matching
// gguf-py's convention -- the raw bytes need no reordering, only the shape
// metadata differs.
static void write_npy_f32(const std::string & path, const float * data, const std::vector<int64_t> & ggml_shape) {
    std::vector<int64_t> shape(ggml_shape.rbegin(), ggml_shape.rend());
    std::string shape_str = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        shape_str += std::to_string(shape[i]);
        shape_str += (shape.size() == 1 || i + 1 < shape.size()) ? "," : "";
        if (i + 1 < shape.size()) shape_str += " ";
    }
    shape_str += ")";

    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_str + ", }";
    size_t unpadded_len = 10 + header.size() + 1; // magic(6)+ver(2)+hlen(2) + header + '\n'
    size_t padded_len = ((unpadded_len + 63) / 64) * 64;
    size_t pad = padded_len - unpadded_len;
    header.append(pad, ' ');
    header += "\n";

    uint16_t hlen = (uint16_t) header.size();

    std::ofstream f(path, std::ios::binary);
    f.write("\x93NUMPY", 6);
    const char ver[2] = {1, 0};
    f.write(ver, 2);
    f.write((const char *) &hlen, 2);
    f.write(header.data(), (std::streamsize) header.size());

    int64_t n = 1;
    for (auto d : ggml_shape) n *= d;
    f.write((const char *) data, (std::streamsize) (n * (int64_t) sizeof(float)));
}

struct DumpState {
    std::string dir;
    std::set<std::string> want; // ggml tensor names to capture in the current phase
    std::map<std::string, std::string> out_name; // ggml name -> output .npy stem
};

static bool cohere_asr_dump_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (DumpState *) user_data;
    if (st->want.empty() || st->want.find(t->name) == st->want.end()) {
        return true;
    }
    if (ask) {
        return true;
    }

    std::vector<float> buf(ggml_nelements(t));
    if (ggml_backend_buffer_is_host(t->buffer)) {
        memcpy(buf.data(), t->data, ggml_nbytes(t));
    } else {
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
    }

    std::vector<int64_t> shape(t->ne, t->ne + GGML_MAX_DIMS);
    while (shape.size() > 1 && shape.back() == 1) shape.pop_back();

    const std::string & stem = st->out_name.count(t->name) ? st->out_name[t->name] : std::string(t->name);
    write_npy_f32(st->dir + "/" + stem + ".npy", buf.data(), shape);
    fprintf(stderr, "dumped %s -> %s.npy\n", t->name, stem.c_str());

    return true;
}

struct Args {
    std::string model;
    std::string audio;
    std::string lang = "en";
    bool punctuation = true;
    int max_tokens = 256;
    int n_gpu_layers = 0;
    float max_clip_seconds = 35.0f;
    std::string dump_dir;
};

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m <model.gguf> -f <audio file> [options]\n"
        "  -m, --model <path>       path to the cohere-asr GGUF model (required)\n"
        "  -f, --file <path>        audio file to transcribe: wav/mp3/flac (required)\n"
        "      --lang <code>        language code, default \"en\" (%s)\n"
        "      --no-pnc             disable punctuation/capitalization\n"
        "  -n, --max-tokens <n>     max generated tokens per chunk, default 256\n"
        "  -ngl, --n-gpu-layers <n> number of layers to offload to GPU, default 0 (CPU)\n"
        "      --max-clip-seconds <s>  split audio longer than this into chunks, default 35\n"
        "      --dump-dir <path>    dump intermediate tensors as .npy for validation\n",
        argv0, [] {
            std::string s;
            for (const auto & l : SUPPORTED_LANGS) { if (!s.empty()) s += ","; s += l; }
            return s;
        }().c_str());
}

static bool parse_args(int argc, char ** argv, Args & args) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", flag);
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") {
            const char * v = next(a.c_str()); if (!v) return false;
            args.model = v;
        } else if (a == "-f" || a == "--file") {
            const char * v = next(a.c_str()); if (!v) return false;
            args.audio = v;
        } else if (a == "--lang") {
            const char * v = next(a.c_str()); if (!v) return false;
            args.lang = v;
        } else if (a == "--no-pnc") {
            args.punctuation = false;
        } else if (a == "-n" || a == "--max-tokens") {
            const char * v = next(a.c_str()); if (!v) return false;
            args.max_tokens = atoi(v);
        } else if (a == "-ngl" || a == "--n-gpu-layers") {
            const char * v = next(a.c_str()); if (!v) return false;
            args.n_gpu_layers = atoi(v);
        } else if (a == "--max-clip-seconds") {
            const char * v = next(a.c_str()); if (!v) return false;
            args.max_clip_seconds = (float) atof(v);
        } else if (a == "--dump-dir") {
            const char * v = next(a.c_str()); if (!v) return false;
            args.dump_dir = v;
        } else if (a == "-h" || a == "--help") {
            return false;
        } else {
            fprintf(stderr, "unknown argument: %s\n", a.c_str());
            return false;
        }
    }
    if (args.model.empty() || args.audio.empty()) {
        return false;
    }
    if (args.max_clip_seconds <= 0.0f) {
        fprintf(stderr, "--max-clip-seconds must be positive\n");
        return false;
    }
    if (SUPPORTED_LANGS.find(args.lang) == SUPPORTED_LANGS.end()) {
        fprintf(stderr, "warning: '%s' is not in the model's documented supported_languages list\n", args.lang.c_str());
    }
    return true;
}

// Greedily transcribes one already-encoded segment. Assumes llama_encode()
// has just been called for this segment's mel frames (populating the cross-
// attention context) and that the decoder's own KV cache is empty (fresh
// llama_memory_clear() before this call, or first segment).
static std::string decode_segment(
        llama_context * ctx, const llama_vocab * vocab,
        const std::vector<llama_token> & prompt_ids, int max_new_tokens,
        llama_batch & dec_batch,
        bool capture_dump, DumpState * dump_state) {
    std::vector<llama_token> ids = prompt_ids;
    const int prompt_len = (int) ids.size();
    const llama_token eos_id = llama_vocab_eos(vocab);

    std::string transcript;

    for (int step = 0; step < prompt_len + max_new_tokens; ++step) {
        llama_token cur_id = step < prompt_len ? ids[step] : ids.back();

        dec_batch.n_tokens = 1;
        dec_batch.token[0] = cur_id;
        dec_batch.pos[0] = (llama_pos) step;
        dec_batch.n_seq_id[0] = 1;
        dec_batch.seq_id[0][0] = 0;
        dec_batch.logits[0] = 1;

        if (capture_dump && dump_state && step == prompt_len - 1) {
            // note: layer 7 is the last decoder layer, so its "l_out-7" name
            // gets overwritten (same ggml_tensor object) by the very next
            // cb(cur, "result_embd", -1) call in the graph before any
            // computation happens -- capture "result_embd" instead, which is
            // the same pre-final-norm tensor the reference hook captured.
            dump_state->want = {"l_out-0", "l_out-4", "result_embd", "result_output"};
            dump_state->out_name = {
                {"l_out-0", "dec_layer_0_step0_cpp"}, {"l_out-4", "dec_layer_4_step0_cpp"},
                {"result_embd", "dec_layer_7_step0_cpp"}, {"result_output", "first_step_logits_cpp"},
            };
        }

        if (llama_decode(ctx, dec_batch) != 0) {
            fprintf(stderr, "llama_decode failed at step %d\n", step);
            return transcript;
        }

        if (capture_dump && dump_state && step == prompt_len - 1) {
            dump_state->want.clear();
        }

        if (step + 1 < prompt_len) {
            continue; // still feeding the fixed prompt, don't sample yet
        }

        const float * logits = llama_get_logits_ith(ctx, 0);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        int32_t best = 0;
        float best_val = logits[0];
        for (int32_t i = 1; i < n_vocab; ++i) {
            if (logits[i] > best_val) { best_val = logits[i]; best = i; }
        }

        if (best == eos_id) {
            break;
        }
        ids.push_back(best);

        char buf[256];
        int32_t n = llama_token_to_piece(vocab, best, buf, sizeof(buf), 0, true);
        if (n > 0) transcript.append(buf, n);
    }

    return transcript;
}

int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 1;
    }

    std::vector<float> wav;
    if (!decode_audio_file(args.audio, 16000, wav)) {
        return 1;
    }

    const int n_fft = 512;
    const int hop = 160;
    const int win_length = 400;
    const int64_t n_mel = 128;
    const int64_t n_fft_bins = n_fft / 2 + 1;

    std::vector<float> mel_fb, mel_window;
    if (!read_gguf_f32_tensor(args.model, "enc.mel_fb.weight", mel_fb) ||
        !read_gguf_f32_tensor(args.model, "enc.mel_window.weight", mel_window)) {
        fprintf(stderr, "failed to read mel_fb / mel_window from gguf\n");
        return 1;
    }

    // --- split into chunks (single chunk if the clip is short enough) and
    // compute log-mel per chunk up front, so we know the largest one before
    // sizing the context/batch buffers ---
    std::vector<AudioSegment> segments = split_audio_chunks_energy(wav, 16000, args.max_clip_seconds);
    std::vector<std::vector<float>> segment_logmels(segments.size());
    int64_t max_n_frames = 0;
    for (size_t i = 0; i < segments.size(); ++i) {
        std::vector<float> seg_wav(wav.begin() + segments[i].start, wav.begin() + segments[i].end);
        segment_logmels[i] = compute_logmel(seg_wav, mel_fb, n_fft_bins, n_mel, mel_window, n_fft, hop, win_length);
        const int64_t n_frames = (int64_t) segment_logmels[i].size() / n_mel;
        max_n_frames = std::max(max_n_frames, n_frames);
    }
    if (segments.size() > 1) {
        fprintf(stderr, "audio is %.1fs, split into %zu chunks (max --max-clip-seconds %.1fs)\n",
                (double) wav.size() / 16000.0, segments.size(), (double) args.max_clip_seconds);
    }

    DumpState dump_state;
    if (!args.dump_dir.empty()) {
        dump_state.dir = args.dump_dir;
        // only the first chunk's mel is dumped for validation -- dump_dir is
        // meant for short, single-chunk reference comparisons
        write_npy_f32(args.dump_dir + "/mel_cpp.npy", segment_logmels[0].data(),
                {(int64_t) segment_logmels[0].size() / n_mel, n_mel});
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args.n_gpu_layers;
    llama_model * model = llama_model_load_from_file(args.model.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    // n_ctx must cover the single-shot encoder batch (max_n_frames raw mel
    // frames across all chunks, which can exceed the decoder's own 1024-token
    // context) as well as the decoder's up-to-1024-token generation --
    // llama.cpp clamps n_ubatch to n_ctx internally, so n_ctx < n_frames would
    // silently break the "n_ubatch >= n_tokens" assumption llama_encode()
    // requires. Chunking (above) keeps max_n_frames bounded regardless of
    // total clip length.
    cparams.n_ctx = (uint32_t) std::max<int64_t>(max_n_frames, 1024);
    cparams.n_batch = (uint32_t) std::max<int64_t>(max_n_frames, 512);
    cparams.n_ubatch = cparams.n_batch;
    if (!args.dump_dir.empty()) {
        cparams.cb_eval = cohere_asr_dump_cb;
        cparams.cb_eval_user_data = &dump_state;
    }
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    // --- build the cohere-asr control-token prompt for the requested language ---
    const std::string pnc_token = args.punctuation ? "<|pnc|>" : "<|nopnc|>";
    const std::string prompt =
        "<|startofcontext|><|startoftranscript|><|emo:undefined|>"
        "<|" + args.lang + "|><|" + args.lang + "|>" + pnc_token +
        "<|noitn|><|notimestamp|><|nodiarize|>";

    std::vector<llama_token> prompt_ids(prompt.size() + 16);
    int32_t n_prompt = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
            prompt_ids.data(), (int32_t) prompt_ids.size(), /*add_special=*/false, /*parse_special=*/true);
    if (n_prompt < 0) {
        fprintf(stderr, "prompt tokenization buffer too small\n");
        return 1;
    }
    prompt_ids.resize(n_prompt);

    // enc_batch is allocated once for the largest chunk and reused (n_tokens
    // set per-chunk) -- same pattern dec_batch already uses per decode step.
    llama_batch enc_batch = llama_batch_init((int32_t) max_n_frames, (int32_t) n_mel, 1);
    llama_batch dec_batch = llama_batch_init(1, 0, 1);

    std::vector<std::string> segment_texts;
    llama_memory_t mem = llama_get_memory(ctx);

    for (size_t s = 0; s < segments.size(); ++s) {
        const std::vector<float> & logmel = segment_logmels[s];
        const int64_t n_frames = (int64_t) logmel.size() / n_mel;

        enc_batch.n_tokens = (int32_t) n_frames;
        for (int64_t t = 0; t < n_frames; ++t) {
            for (int64_t m = 0; m < n_mel; ++m) {
                enc_batch.embd[t * n_mel + m] = logmel[(size_t) m * n_frames + t];
            }
            enc_batch.pos[t] = (llama_pos) t;
            enc_batch.n_seq_id[t] = 1;
            enc_batch.seq_id[t][0] = 0;
            enc_batch.logits[t] = 0;
        }

        const bool is_first = (s == 0);
        if (is_first && !args.dump_dir.empty()) {
            dump_state.want = {"l_out-0", "l_out-1", "l_out-24", "l_out-47", "result_embd", "result_embd_proj"};
            dump_state.out_name = {
                {"l_out-0", "enc_layer_0"}, {"l_out-1", "enc_layer_1"},
                {"l_out-24", "enc_layer_24"}, {"l_out-47", "enc_layer_47"},
                {"result_embd", "encoder_out_cpp"}, {"result_embd_proj", "encoder_proj_cpp"},
            };
        }

        if (llama_encode(ctx, enc_batch) != 0) {
            fprintf(stderr, "llama_encode failed on chunk %zu\n", s);
            return 1;
        }
        dump_state.want.clear();
        fprintf(stderr, "encode done (chunk %zu/%zu, %lld mel frames)\n", s + 1, segments.size(), (long long) n_frames);

        std::string text = decode_segment(ctx, vocab, prompt_ids, args.max_tokens, dec_batch,
                is_first, args.dump_dir.empty() ? nullptr : &dump_state);
        segment_texts.push_back(text);

        // reset the decoder's KV cache before the next independent chunk
        // (the encoder's cross-attention context is fully overwritten by the
        // next llama_encode() call automatically, no explicit reset needed there)
        if (s + 1 < segments.size()) {
            llama_memory_clear(mem, true);
        }
    }

    const std::string separator = NO_SPACE_LANGS.count(args.lang) ? "" : " ";
    std::string transcript;
    for (const auto & text : segment_texts) {
        std::string t = text;
        size_t b = t.find_first_not_of(" \t\r\n");
        size_t e = t.find_last_not_of(" \t\r\n");
        if (b == std::string::npos) continue; // empty/whitespace-only chunk
        t = t.substr(b, e - b + 1);
        if (!transcript.empty()) transcript += separator;
        transcript += t;
    }

    printf("%s\n", transcript.c_str());

    llama_batch_free(dec_batch);
    llama_batch_free(enc_batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
