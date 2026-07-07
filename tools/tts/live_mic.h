// tools/tts/live_mic.h
// Live microphone capture + simple RMS-energy VAD utterance segmentation,
// used by `llama-qwen3tts --embed-only --live-mic`.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

// All thresholds/durations are tunable from the CLI (--vad-threshold,
// --silence-gap-ms, --min-utterance-ms, --max-utterance-ms).
struct live_mic_vad_params {
    float silence_rms      = 0.01f; // below this RMS (samples in [-1,1]), a chunk counts as silence
    int   silence_gap_ms   = 400;   // trailing silence needed to close an utterance
    int   min_utterance_ms = 500;   // shorter utterances are discarded as noise
    int   max_utterance_ms = 8000;  // force-cut a continuous utterance at this length

    // If non-empty, use the ggml-native Silero VAD (silero_vad.h, loaded from
    // this GGUF - see tools/tts/convert_silero_vad.py) instead of the
    // RMS-energy threshold above. Capture internally switches to 16kHz to
    // match what that model expects; utterances handed to on_utterance are
    // always resampled back to 24kHz first, same as the RMS path.
    std::string vad_model_path;
    float       vad_threshold = 0.5f; // Silero speech-probability threshold

    // If > 0, ignore VAD entirely: just record continuously for this many
    // seconds, then emit the whole buffer as a single utterance and stop.
    // Used by --record-seconds for "record exactly N seconds, one
    // embedding, done" - simpler and more predictable than VAD-based
    // segmentation when you already know how long you're about to talk.
    float record_seconds = 0.0f;
};

// Captures mono 24kHz float audio from the default input device on a
// miniaudio-managed audio thread, and segments it into utterances on a
// separate worker thread using simple RMS-energy VAD. `on_utterance` is
// invoked from that worker thread (never from the audio callback) once each
// utterance closes, so a slow callback (e.g. running embedding extraction)
// never blocks or glitches the live capture.
struct live_mic_capture {
    live_mic_capture();
    ~live_mic_capture();

    // Returns false if the device could not be opened (no input device,
    // permission denied, etc).
    bool start(const live_mic_vad_params & params,
               const std::function<void(const std::vector<float> &)> & on_utterance);

    // Stops capture and joins the worker thread. Safe to call more than
    // once; also called automatically by the destructor.
    void stop();

    // Opaque; defined in live_mic.cpp. Public only so the free-function audio
    // callback and worker thread in that file can take a pointer to it -
    // still not part of the usable API (there's no way to construct one
    // outside live_mic.cpp).
    struct impl;

private:
    std::unique_ptr<impl> pimpl;
};
