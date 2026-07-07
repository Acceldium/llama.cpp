// tools/tts/silero_vad.h
// ggml-native Silero VAD (16kHz path only). No torch, no onnxruntime - loads
// weights from a small GGUF file (see tools/tts/convert_silero_vad.py) and
// runs the model directly via ggml, the same way this codebase already runs
// the ECAPA-TDNN speaker encoder without needing a full llama_model.
//
// This is the C++ CLI's optional upgrade from the default RMS-energy VAD in
// live_mic.cpp - see examples/meeting-voice-viz/HOW_IT_WORKS.md for why RMS
// was the default and this exists as an opt-in alternative behind
// --vad-model, not a replacement.
#pragma once

#include <memory>
#include <string>

constexpr int SILERO_VAD_SR              = 16000;
constexpr int SILERO_VAD_CHUNK_SAMPLES   = 512; // 32ms @ 16kHz - the only size this model accepts
constexpr int SILERO_VAD_CONTEXT_SAMPLES = 64;  // trailing context carried between calls

struct silero_vad {
    silero_vad();
    ~silero_vad();

    // Load weights from a GGUF file produced by convert_silero_vad.py.
    bool load(const std::string & path);

    // Feed exactly SILERO_VAD_CHUNK_SAMPLES new 16kHz mono float samples in
    // [-1,1]. Internally prepends the trailing context from the previous
    // call (zero for the first call), runs the model, and returns the
    // speech probability in [0,1]. Recurrent (h,c) state and the context
    // tail are both updated for the next call.
    float process_chunk(const float * chunk, int n);

    // Resets recurrent state and context - call when starting a fresh
    // utterance/stream so old audio doesn't bleed into the first decision.
    void reset_states();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
