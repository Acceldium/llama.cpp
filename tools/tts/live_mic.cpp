// tools/tts/live_mic.cpp
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio/miniaudio.h"

#include "live_mic.h"
#include "silero_vad.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

static constexpr int RMS_SR             = 24000;
static constexpr int RMS_CHUNK_MS       = 20;
static constexpr int RMS_CHUNK_SAMPLES  = RMS_SR * RMS_CHUNK_MS / 1000; // 480

struct live_mic_capture::impl {
    ma_device device{};
    bool device_ready = false;

    std::mutex mtx;
    std::condition_variable cv;
    std::deque<float> queue; // filled by the audio callback, drained by the worker thread
    bool stop_requested = false;

    std::thread worker;
    std::function<void(const std::vector<float> &)> on_utterance;
    live_mic_vad_params params;
    std::unique_ptr<silero_vad> vad; // non-null only when params.vad_model_path was loaded successfully
};

static void live_mic_data_callback(ma_device * device, void * /*output*/, const void * input, ma_uint32 frame_count) {
    auto * self = static_cast<live_mic_capture::impl *>(device->pUserData);
    const float * samples = static_cast<const float *>(input);
    std::lock_guard<std::mutex> lock(self->mtx);
    self->queue.insert(self->queue.end(), samples, samples + frame_count);
    self->cv.notify_one();
}

// Linear-interpolation resample - simple and dependency-free, adequate for a
// once-per-utterance rate conversion (16kHz Silero capture -> 24kHz that
// extract_speaker_embedding expects). Not used at all on the RMS/24kHz path.
static std::vector<float> resample_linear(const std::vector<float> & in, int sr_in, int sr_out) {
    if (sr_in == sr_out || in.empty()) return in;
    size_t n_out = (size_t)((double)in.size() * sr_out / sr_in);
    std::vector<float> out(n_out);
    for (size_t i = 0; i < n_out; i++) {
        double src_pos = (double)i * sr_in / sr_out;
        size_t i0 = (size_t)src_pos;
        size_t i1 = (std::min)(i0 + 1, in.size() - 1); // extra parens: avoid Windows.h's min() macro (pulled in via miniaudio.h)
        double frac = src_pos - (double)i0;
        out[i] = (float)(in[i0] * (1.0 - frac) + in[i1] * frac);
    }
    return out;
}

// Fixed-duration mode: no VAD at all, just record record_seconds worth of
// audio and emit it as one utterance. Kept as its own straight-line function
// rather than folded into the VAD state machine below - simpler to read and
// nothing about it needs silence-gap/min/max-utterance logic.
static void live_mic_record_seconds_thread(live_mic_capture::impl * self, int sr, int chunk_samples) {
    const int target_samples = (int)(self->params.record_seconds * sr);
    std::vector<float> buffer;
    buffer.reserve(target_samples);

    while ((int)buffer.size() < target_samples) {
        std::vector<float> chunk(chunk_samples);
        {
            std::unique_lock<std::mutex> lock(self->mtx);
            self->cv.wait(lock, [&] {
                return self->stop_requested || self->queue.size() >= (size_t)chunk_samples;
            });
            if (self->stop_requested && self->queue.size() < (size_t)chunk_samples) {
                break; // stopped early (Ctrl+C) before reaching the target duration
            }
            for (int i = 0; i < chunk_samples; i++) {
                chunk[i] = self->queue.front();
                self->queue.pop_front();
            }
        }
        buffer.insert(buffer.end(), chunk.begin(), chunk.end());
    }

    if (!buffer.empty()) {
        if ((int)buffer.size() > target_samples) buffer.resize(target_samples); // trim chunk-granularity overshoot
        std::vector<float> out = (sr == RMS_SR) ? buffer : resample_linear(buffer, sr, RMS_SR);
        self->on_utterance(out);
    }
}

static void live_mic_worker_thread(live_mic_capture::impl * self, int sr, int chunk_samples) {
    if (self->params.record_seconds > 0.0f) {
        live_mic_record_seconds_thread(self, sr, chunk_samples);
        return;
    }

    const int chunk_ms = 1000 * chunk_samples / sr;
    std::vector<float> chunk(chunk_samples);
    std::vector<float> utterance;
    bool in_speech      = false;
    int  silence_ms     = 0;
    int  utterance_ms   = 0;

    auto emit = [&]() {
        std::vector<float> out = (sr == RMS_SR) ? utterance : resample_linear(utterance, sr, RMS_SR);
        self->on_utterance(out);
        if (self->vad) self->vad->reset_states();
    };

    for (;;) {
        {
            std::unique_lock<std::mutex> lock(self->mtx);
            self->cv.wait(lock, [&] {
                return self->stop_requested || self->queue.size() >= (size_t)chunk_samples;
            });
            if (self->stop_requested && self->queue.size() < (size_t)chunk_samples) {
                break;
            }
            for (int i = 0; i < chunk_samples; i++) {
                chunk[i] = self->queue.front();
                self->queue.pop_front();
            }
        }

        bool is_speech;
        if (self->vad) {
            float prob = self->vad->process_chunk(chunk.data(), (int)chunk.size());
            is_speech = prob >= self->params.vad_threshold;
        } else {
            float rms = 0.0f;
            for (float v : chunk) rms += v * v;
            rms = sqrtf(rms / chunk.size());
            is_speech = rms >= self->params.silence_rms;
        }

        if (is_speech) {
            utterance.insert(utterance.end(), chunk.begin(), chunk.end());
            utterance_ms += chunk_ms;
            in_speech  = true;
            silence_ms = 0;
        } else if (in_speech) {
            // Keep trailing silence in the buffer - it's a natural pause, not the end of speech,
            // until silence_gap_ms is reached below.
            utterance.insert(utterance.end(), chunk.begin(), chunk.end());
            utterance_ms += chunk_ms;
            silence_ms += chunk_ms;
        }

        const bool gap_reached = in_speech && silence_ms >= self->params.silence_gap_ms;
        const bool max_reached = in_speech && utterance_ms >= self->params.max_utterance_ms;
        if (gap_reached || max_reached) {
            if (utterance_ms >= self->params.min_utterance_ms) {
                emit();
            }
            utterance.clear();
            in_speech    = false;
            silence_ms   = 0;
            utterance_ms = 0;
        }
    }

    if (in_speech && utterance_ms >= self->params.min_utterance_ms) {
        emit(); // flush a trailing in-progress utterance on stop
    }
}

live_mic_capture::live_mic_capture() : pimpl(std::make_unique<impl>()) {}
live_mic_capture::~live_mic_capture() { stop(); }

bool live_mic_capture::start(const live_mic_vad_params & params,
                              const std::function<void(const std::vector<float> &)> & on_utterance) {
    pimpl->params       = params;
    pimpl->on_utterance = on_utterance;

    if (!params.vad_model_path.empty()) {
        auto vad = std::make_unique<silero_vad>();
        if (vad->load(params.vad_model_path)) {
            pimpl->vad = std::move(vad);
        } else {
            fprintf(stderr, "WARN: failed to load VAD model %s, falling back to RMS-energy VAD\n",
                    params.vad_model_path.c_str());
        }
    }

    const int sr             = pimpl->vad ? SILERO_VAD_SR          : RMS_SR;
    const int chunk_samples  = pimpl->vad ? SILERO_VAD_CHUNK_SAMPLES : RMS_CHUNK_SAMPLES;

    ma_device_config config = ma_device_config_init(ma_device_type_capture);
    config.capture.format   = ma_format_f32;
    config.capture.channels = 1;
    config.sampleRate       = sr;
    config.dataCallback     = live_mic_data_callback;
    config.pUserData        = pimpl.get();

    if (ma_device_init(nullptr, &config, &pimpl->device) != MA_SUCCESS) {
        return false;
    }
    pimpl->device_ready = true;

    if (ma_device_start(&pimpl->device) != MA_SUCCESS) {
        ma_device_uninit(&pimpl->device);
        pimpl->device_ready = false;
        return false;
    }

    pimpl->worker = std::thread(live_mic_worker_thread, pimpl.get(), sr, chunk_samples);
    return true;
}

void live_mic_capture::stop() {
    if (pimpl->device_ready) {
        ma_device_uninit(&pimpl->device); // stops the device too; never call from the callback itself
        pimpl->device_ready = false;
    }
    {
        std::lock_guard<std::mutex> lock(pimpl->mtx);
        pimpl->stop_requested = true;
        pimpl->cv.notify_one();
    }
    if (pimpl->worker.joinable()) {
        pimpl->worker.join();
    }
}
