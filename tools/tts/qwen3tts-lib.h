// tools/tts/qwen3tts-lib.h
// Qwen3-TTS shared library interface.
// Included by both llama-qwen3tts CLI and llama-server to support /v1/audio/speech.
#pragma once

#include "llama.h"

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <cstdint>

// ─── Sampler parameters ───────────────────────────────────────────────────────
struct tts_sampler_params {
    float temp        = 0.9f;
    int   top_k       = 50;
    float top_p       = 1.0f;
    float rep_penalty = 1.05f;
    int   rep_last_n  = 64;
    bool  greedy      = false;
    std::vector<int32_t> suppress_tokens; // token IDs set to -inf during sampling
};

// ─── Per-request parameters ───────────────────────────────────────────────────
struct qwen3_tts_request {
    std::string text;
    std::string language        = "english";
    std::string ref_audio_path; // optional: path to 24 kHz reference WAV for voice cloning
    std::string ref_text;       // optional: transcript of ref audio (enables ICL mode)
    std::string ref_codes_path; // optional: pre-encoded codec codes file for ICL mode
    std::string speaker_name;   // optional: named speaker preset (CustomVoice checkpoints only)
    std::string instruct;       // optional: natural-language style/emotion instruction (CustomVoice, 1.7B+)
    bool        streaming_text  = false;

    tts_sampler_params talker_params;
    tts_sampler_params cp_params;

    int seed       = -1;
    int max_tokens = 2048;
};

// ─── Persistent TTS context ───────────────────────────────────────────────────
// Load once at server/tool startup with load(), then call generate() per request.
// Multiple concurrent generate() calls are serialised via an internal mutex.
struct qwen3_tts_ctx {
    // Declared out-of-line (see qwen3tts-lib.cpp): a defaulted ctor/dtor defined
    // here inline would need spk_encoder_cache to be a complete type already.
    qwen3_tts_ctx();

    // Public: read by server speech handler to check availability
    llama_model * talker_model = nullptr;
    llama_model * cp_model     = nullptr;
    std::string   talker_path;  // GGUF path (needed for spk_enc.* / tts.* tensor access)
    std::string   cp_path;      // GGUF path (needed for tts.cp.* tensor access)
    std::string   vocoder_path; // GGUF path (needed for tok_dec.* / tok_enc.*)
    int           n_gpu_layers = 0;

    // Load models from disk.  Replaces any previously loaded models.
    bool load(const std::string & talker, const std::string & cp,
              const std::string & vocoder, int n_gpu = 0);

    // Free loaded models.
    void unload();

    bool is_loaded() const { return talker_model != nullptr && cp_model != nullptr; }

    // Named speaker presets available in the loaded Talker GGUF (CustomVoice
    // checkpoints only; empty for Base checkpoints, which have no spk_id table).
    std::vector<std::string> get_supported_speakers() const;

    // Extract a 1024-dim speaker embedding (x-vector) from 24kHz mono audio,
    // using only the ECAPA-TDNN speaker encoder weights (spk_enc.* tensors)
    // in the Talker GGUF at `talker_path`. Does not require cp/vocoder or a
    // loaded llama_model. Returns false if the GGUF has no speaker encoder.
    //
    // The encoder weights are read from disk once (on the first call) and
    // cached for the lifetime of this ctx; every later call only re-runs the
    // (cheap, ~6M-param) forward pass on the new audio. This is what makes
    // repeated calls on a persistent process (the server) fast - a fresh CLI
    // process has no ctx to cache into, so it always pays the one-time load.
    bool extract_speaker_embedding(const std::vector<float> & audio_samples,
                                    std::vector<float> & embedding_out) const;

    // Generate speech: text → 24 kHz mono float PCM.
    // Thread-safe (internally serialised with a mutex).
    // Returns false and leaves audio_out unmodified on error.
    //
    // If `stream_callback` is set, the vocoder is re-run on the growing prefix of
    // decoded codec frames every `frames_per_chunk` frames, and each newly available
    // slice of PCM is delivered via the callback *while the autoregressive decode
    // loop is still running* (i.e. before the full text has finished generating).
    // Return false from the callback to abort generation early. audio_out is still
    // populated with the complete audio on success, same as when streaming is off.
    bool generate(const qwen3_tts_request & req, std::vector<float> & audio_out,
                  const std::function<bool(std::vector<float>)> & stream_callback = nullptr,
                  int frames_per_chunk = 12);

    // Streaming generate: invokes `callback(chunk)` with float PCM chunks as audio
    // becomes available *during* decoding (not after the fact). The first chunk is
    // produced after `frames_per_chunk` decode frames complete; subsequent chunks
    // arrive roughly every frames_per_chunk frames while later frames are still
    // being generated. Thin wrapper around generate() with stream_callback set.
    //
    // Calling this while another generate/generate_streaming call is in progress
    // will block until that call finishes (mutex-serialised).
    //
    // Return false from the callback to abort early.
    // Returns false if an unrecoverable error occurred.
    bool generate_streaming(const qwen3_tts_request & req,
                            const std::function<bool(std::vector<float>)> & callback,
                            int frames_per_chunk = 12);

    ~qwen3_tts_ctx();

private:
    mutable std::mutex mtx;

    // Cached speaker-encoder weights (opaque here; defined in qwen3tts-lib.cpp).
    // Populated once, lazily, by extract_speaker_embedding(). Defining the
    // destructor out-of-line (see qwen3tts-lib.cpp) is required so unique_ptr
    // can be destroyed with only a forward-declared type visible here.
    struct spk_encoder_cache;
    mutable std::once_flag spk_cache_once;
    mutable std::unique_ptr<spk_encoder_cache> spk_cache;
};

// ─── WAV helpers ──────────────────────────────────────────────────────────────

// Encode 24 kHz mono float PCM to in-memory WAV bytes (full, known length).
std::vector<uint8_t> qwen3_tts_encode_wav(const std::vector<float> & samples,
                                           int sample_rate = 24000);

// Build a 44-byte streaming WAV header (data chunk size = 0x7FFFFFFF).
// Append raw int16 PCM after this header.  Compatible with most players.
std::vector<uint8_t> qwen3_tts_streaming_wav_header(int sample_rate = 24000);

// Convert float PCM samples to signed-16 little-endian bytes.
std::vector<uint8_t> qwen3_tts_pcm_to_s16_bytes(const std::vector<float> & samples);

// Write 24 kHz mono float PCM to a WAV file.
void qwen3_tts_write_wav(const char * path, const float * samples, int n,
                          int sr = 24000);

// Read a mono WAV file into float samples at the given target_sr (no resampling,
// only checks the rate and warns).
bool qwen3_tts_read_wav(const char * path, std::vector<float> & out,
                         int target_sr = 24000);

// Same as qwen3_tts_read_wav, but parses an in-memory WAV buffer (e.g. an
// HTTP file upload) instead of reading from disk.
bool qwen3_tts_read_wav_from_memory(const uint8_t * data, size_t len,
                                     std::vector<float> & out, int target_sr = 24000);
