// tools/tts/qwen3tts.cpp
// Qwen3-TTS CLI: thin wrapper around the qwen3tts-lib shared library.
//
// Usage:
//   llama-qwen3tts \
//     --model-talker talker.gguf \
//     --model-cp code-predictor.gguf \
//     --model-vocoder tokenizer.gguf \
//     --ref-audio reference.wav \
//     --text "Hello world" \
//     --output output.wav

#include "qwen3tts-lib.h"
#include "live_mic.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

static void print_usage(const char * prog) {
    fprintf(stderr,
        "Qwen3-TTS: text-to-speech via llama.cpp\n\n"
        "Usage:\n"
        "  %s --model-talker <talker.gguf> \\\n"
        "     --model-cp <code-predictor.gguf> \\\n"
        "     --model-vocoder <tokenizer.gguf> \\\n"
        "     --text \"Hello world\" \\\n"
        "     --output <output.wav>\n\n"
        "Required:\n"
        "  --model-talker        Talker GGUF (contains speaker encoder + LLM)\n"
        "  --model-cp            Code Predictor GGUF\n"
        "  --text                Input text to synthesize\n\n"
        "Voice cloning:\n"
        "  --ref-audio <wav>     Reference audio for speaker cloning\n"
        "  --ref-text <text>     Reference transcript for ICL cloning (requires --ref-audio)\n"
        "  --ref-codes <file>    Precomputed codec codes file (optional for ICL)\n\n"
        "Speaker embedding extraction:\n"
        "  --embed-only          Extract the 1024-dim speaker embedding (x-vector) from\n"
        "                        --ref-audio and exit. Only --model-talker and --ref-audio\n"
        "                        are required (no --model-cp, no --text).\n"
        "                        Written as raw float32 to --output (default: embedding.bin)\n"
        "  --interactive         With --embed-only: instead of a single extraction, keep the\n"
        "                        process (and the loaded speaker encoder weights) alive and\n"
        "                        read \"<input_wav>[TAB]<output_bin>\" lines from stdin, one\n"
        "                        embedding per line, until stdin closes. Prints \"OK <bytes>\"\n"
        "                        or \"ERROR <message>\" per line. Unloads and exits on EOF.\n"
        "                        This avoids paying the model-load cost on every embedding -\n"
        "                        run once, feed it many files, e.g. from a driver script.\n\n"
        "  --live-mic            With --embed-only: capture continuously from the default\n"
        "                        microphone instead of stdin/--ref-audio. A simple RMS-energy\n"
        "                        VAD segments the stream into utterances; each utterance's\n"
        "                        embedding is written to --output-dir/utterance_NNNN.bin and\n"
        "                        a line \"UTTERANCE <idx> <duration_s> <path>\" is printed to\n"
        "                        stdout as soon as it's ready. Runs until Ctrl+C (or until\n"
        "                        --max-utterances is reached), then unloads and exits.\n"
        "  --output-dir <dir>    Where --live-mic writes utterance_NNNN.bin files (default: .)\n"
        "  --record-seconds <f>  --live-mic: ignore VAD entirely, just record this many seconds\n"
        "                        from the mic, then write one embedding and exit. Simpler than\n"
        "                        VAD segmentation when you already know how long you'll talk -\n"
        "                        e.g. for enrolling one speaker at a time.\n"
        "  --max-utterances <n>  --live-mic: exit automatically after capturing n utterances\n"
        "                        instead of waiting for Ctrl+C (default: 0 = unlimited; ignored\n"
        "                        when --record-seconds is set, which is always exactly 1)\n"
        "  --vad-threshold <f>   --live-mic: RMS level below which a chunk counts as silence\n"
        "                        (default: 0.01)\n"
        "  --silence-gap-ms <n>  --live-mic: trailing silence that closes an utterance (default: 400)\n"
        "  --min-utterance-ms <n> --live-mic: utterances shorter than this are discarded (default: 500)\n"
        "  --max-utterance-ms <n> --live-mic: force-cut a continuous utterance at this length\n"
        "                        (default: 8000)\n"
        "  --vad-model <gguf>    --live-mic: use the ggml-native Silero VAD (see\n"
        "                        tools/tts/convert_silero_vad.py) instead of the default\n"
        "                        RMS-energy threshold. More robust to non-speech noise;\n"
        "                        falls back to RMS if the file fails to load.\n"
        "  --vad-prob-threshold <f> --live-mic --vad-model: speech probability [0,1]\n"
        "                        above which a chunk counts as speech (default: 0.5)\n\n"
        "Named speakers (CustomVoice checkpoints only):\n"
        "  --speaker <name>      Named speaker preset, e.g. \"vivian\". Takes precedence\n"
        "                        over --ref-audio if both are given.\n"
        "  --list-speakers       Print the speakers available in --model-talker and exit\n\n"
        "Emotion / style (CustomVoice 1.7B+ only; not supported on 0.6B checkpoints):\n"
        "  --instruct <text>     Natural-language style instruction, e.g.\n"
        "                        \"speak in a happy, excited voice\"\n\n"
        "Language:\n"
        "  --language <lang>     Target language (default: english)\n"
        "                        Supported: english, chinese, german, spanish, french,\n"
        "                        italian, japanese, korean, portuguese, russian, auto\n\n"
        "Sampling:\n"
        "  --temp <float>        Talker temperature (default: 0.9, 0 = greedy)\n"
        "  --top-k <int>         Talker top-k (default: 50, 0 = disabled)\n"
        "  --top-p <float>       Talker top-p / nucleus (default: 1.0)\n"
        "  --rep-penalty <float> Repetition penalty (default: 1.05, 1.0 = off)\n"
        "  --cp-temp <float>     Code Predictor temperature (default: 0.9)\n"
        "  --cp-top-k <int>      Code Predictor top-k (default: 50)\n"
        "  --greedy              Force greedy decoding (overrides temp/top-k)\n"
        "  --seed <int>          Random seed (default: random)\n\n"
        "Other options:\n"
        "  --model-vocoder       Tokenizer GGUF (vocoder decoder)\n"
        "  --output              Output WAV path (default: output.wav)\n"
        "  --max-tokens N        Max decode frames (default: 2048)\n"
        "  --streaming-text      Enable streaming text mode\n"
        "  --n-gpu-layers N      Number of GPU layers (default: 0)\n"
        "  --stream              Stream raw s16le PCM (24kHz mono) to stdout as it is\n"
        "                        generated, instead of writing a WAV file. Pipe to a\n"
        "                        player, e.g.:\n"
        "                          ... --stream | ffplay -f s16le -ar 24000 -ac 1 -i -\n"
        "  --stream-chunk-frames Decode frames per streamed chunk (default: 12, ~1s)\n\n",
        prog);
}

int main(int argc, char ** argv) {
    if (argc < 2) { print_usage(argv[0]); return 1; }

    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    std::string talker_path, cp_path, vocoder_path;
    std::string output_path = "output.wav";

    qwen3_tts_request req;
    req.language = "english";
    req.seed     = -1;
    req.max_tokens = 2048;

    int n_gpu = 0;
    bool stream = false;
    int stream_chunk_frames = 12;
    bool list_speakers = false;
    bool embed_only = false;
    bool interactive = false;
    bool live_mic = false;
    std::string output_dir = ".";
    live_mic_vad_params vad_params;
    int max_utterances = 0; // 0 = unlimited (Ctrl+C to stop)

    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "--model-talker")  == 0 && i + 1 < argc) talker_path = argv[++i];
        else if (strcmp(argv[i], "--model-cp")      == 0 && i + 1 < argc) cp_path     = argv[++i];
        else if (strcmp(argv[i], "--model-vocoder") == 0 && i + 1 < argc) vocoder_path = argv[++i];
        else if (strcmp(argv[i], "--ref-audio")     == 0 && i + 1 < argc) req.ref_audio_path = argv[++i];
        else if (strcmp(argv[i], "--ref-text")      == 0 && i + 1 < argc) req.ref_text       = argv[++i];
        else if (strcmp(argv[i], "--ref-codes")     == 0 && i + 1 < argc) req.ref_codes_path = argv[++i];
        else if (strcmp(argv[i], "--speaker")       == 0 && i + 1 < argc) req.speaker_name   = argv[++i];
        else if (strcmp(argv[i], "--list-speakers") == 0) list_speakers = true;
        else if (strcmp(argv[i], "--embed-only")    == 0) embed_only = true;
        else if (strcmp(argv[i], "--interactive")   == 0) interactive = true;
        else if (strcmp(argv[i], "--live-mic")      == 0) live_mic = true;
        else if (strcmp(argv[i], "--output-dir")       == 0 && i + 1 < argc) output_dir                  = argv[++i];
        else if (strcmp(argv[i], "--vad-threshold")    == 0 && i + 1 < argc) vad_params.silence_rms      = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--silence-gap-ms")   == 0 && i + 1 < argc) vad_params.silence_gap_ms   = atoi(argv[++i]);
        else if (strcmp(argv[i], "--min-utterance-ms") == 0 && i + 1 < argc) vad_params.min_utterance_ms = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-utterance-ms") == 0 && i + 1 < argc) vad_params.max_utterance_ms = atoi(argv[++i]);
        else if (strcmp(argv[i], "--vad-model")         == 0 && i + 1 < argc) vad_params.vad_model_path   = argv[++i];
        else if (strcmp(argv[i], "--vad-prob-threshold") == 0 && i + 1 < argc) vad_params.vad_threshold   = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--max-utterances")    == 0 && i + 1 < argc) max_utterances              = atoi(argv[++i]);
        else if (strcmp(argv[i], "--record-seconds")    == 0 && i + 1 < argc) vad_params.record_seconds   = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--instruct")      == 0 && i + 1 < argc) req.instruct       = argv[++i];
        else if (strcmp(argv[i], "--text")          == 0 && i + 1 < argc) req.text           = argv[++i];
        else if (strcmp(argv[i], "--output")        == 0 && i + 1 < argc) output_path        = argv[++i];
        else if (strcmp(argv[i], "--language")      == 0 && i + 1 < argc) req.language       = argv[++i];
        else if (strcmp(argv[i], "--max-tokens")    == 0 && i + 1 < argc) req.max_tokens     = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temp")          == 0 && i + 1 < argc) req.talker_params.temp        = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top-k")         == 0 && i + 1 < argc) req.talker_params.top_k       = atoi(argv[++i]);
        else if (strcmp(argv[i], "--top-p")         == 0 && i + 1 < argc) req.talker_params.top_p       = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--rep-penalty")   == 0 && i + 1 < argc) req.talker_params.rep_penalty = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--cp-temp")       == 0 && i + 1 < argc) req.cp_params.temp            = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--cp-top-k")      == 0 && i + 1 < argc) req.cp_params.top_k           = atoi(argv[++i]);
        else if (strcmp(argv[i], "--greedy")        == 0) { req.talker_params.greedy = true; req.cp_params.greedy = true; }
        else if (strcmp(argv[i], "--seed")          == 0 && i + 1 < argc) req.seed          = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n-gpu-layers")  == 0 && i + 1 < argc) n_gpu             = atoi(argv[++i]);
        else if (strcmp(argv[i], "--streaming-text") == 0) req.streaming_text = true;
        else if (strcmp(argv[i], "--stream")        == 0) stream = true;
        else if (strcmp(argv[i], "--stream-chunk-frames") == 0 && i + 1 < argc) stream_chunk_frames = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]); return 0;
        }
    }

    if (talker_path.empty()) {
        fprintf(stderr, "ERROR: --model-talker is required\n");
        return 1;
    }
    if (list_speakers) {
        qwen3_tts_ctx ctx;
        ctx.talker_path = talker_path;
        auto speakers = ctx.get_supported_speakers();
        if (speakers.empty()) {
            printf("No named speakers found (not a CustomVoice checkpoint).\n");
        } else {
            printf("Available speakers:\n");
            for (auto & s : speakers) printf("  %s\n", s.c_str());
        }
        return 0;
    }
    if (embed_only) {
        qwen3_tts_ctx ctx;
        ctx.talker_path = talker_path;

        if (interactive) {
            fprintf(stderr, "Interactive embedding mode: reading \"<input_wav>[TAB]<output_bin>\" lines "
                             "from stdin (Ctrl+D / EOF to stop).\n");
            std::string line;
            while (std::getline(std::cin, line)) {
                if (line.empty()) continue;
                size_t tab = line.find('\t');
                if (tab == std::string::npos) {
                    printf("ERROR invalid line (expected <input_wav>[TAB]<output_bin>)\n");
                    fflush(stdout);
                    continue;
                }
                const std::string in_path  = line.substr(0, tab);
                const std::string out_path = line.substr(tab + 1);

                std::vector<float> audio_samples, embedding;
                if (!qwen3_tts_read_wav(in_path.c_str(), audio_samples)) {
                    printf("ERROR cannot read %s\n", in_path.c_str());
                } else if (!ctx.extract_speaker_embedding(audio_samples, embedding)) {
                    printf("ERROR embedding extraction failed for %s\n", in_path.c_str());
                } else {
                    FILE * f = fopen(out_path.c_str(), "wb");
                    if (!f) {
                        printf("ERROR cannot write %s\n", out_path.c_str());
                    } else {
                        fwrite(embedding.data(), sizeof(float), embedding.size(), f);
                        fclose(f);
                        printf("OK %zu\n", embedding.size() * sizeof(float));
                    }
                }
                fflush(stdout);
            }
            fprintf(stderr, "Stdin closed.\n");
            ctx.unload();
            return 0;
        }

        if (live_mic) {
            static std::atomic<bool> g_stop{false};
            std::signal(SIGINT, [](int) { g_stop = true; });

            if (vad_params.record_seconds > 0.0f) {
                // Fixed-duration mode is inherently single-shot: one recording, one embedding, exit.
                max_utterances = 1;
                fprintf(stderr, "Live mic mode: recording %.1fs from the default input device, no VAD.\n",
                        vad_params.record_seconds);
            } else {
                if (max_utterances > 0) {
                    fprintf(stderr, "Live mic mode: capturing from the default input device. "
                                    "Exits automatically after %d utterance%s (Ctrl+C also works).\n",
                            max_utterances, max_utterances == 1 ? "" : "s");
                } else {
                    fprintf(stderr, "Live mic mode: capturing from the default input device. Ctrl+C to stop.\n");
                }
                if (!vad_params.vad_model_path.empty()) {
                    fprintf(stderr, "  VAD: Silero (%s), prob_threshold=%.2f, silence_gap_ms=%d "
                                    "min_utterance_ms=%d max_utterance_ms=%d\n",
                            vad_params.vad_model_path.c_str(), vad_params.vad_threshold,
                            vad_params.silence_gap_ms, vad_params.min_utterance_ms, vad_params.max_utterance_ms);
                } else {
                    fprintf(stderr, "  VAD: RMS-energy, silence_rms=%.4f silence_gap_ms=%d "
                                    "min_utterance_ms=%d max_utterance_ms=%d\n",
                            vad_params.silence_rms, vad_params.silence_gap_ms,
                            vad_params.min_utterance_ms, vad_params.max_utterance_ms);
                }
            }

            int utter_idx = 0;
            live_mic_capture cap;
            const bool started = cap.start(vad_params, [&](const std::vector<float> & samples) {
                std::vector<float> embedding;
                if (!ctx.extract_speaker_embedding(samples, embedding)) {
                    printf("ERROR embedding extraction failed for utterance %d\n", utter_idx);
                    fflush(stdout);
                    return;
                }
                char path[1024];
                snprintf(path, sizeof(path), "%s/utterance_%04d.bin", output_dir.c_str(), utter_idx);
                FILE * f = fopen(path, "wb");
                if (!f) {
                    printf("ERROR cannot write %s\n", path);
                    fflush(stdout);
                    return;
                }
                fwrite(embedding.data(), sizeof(float), embedding.size(), f);
                fclose(f);
                printf("UTTERANCE %d %.3f %s\n", utter_idx, (float)samples.size() / 24000.0f, path);
                fflush(stdout);
                utter_idx++;
            });

            if (!started) {
                fprintf(stderr, "ERROR: failed to start microphone capture (no input device / permission denied?)\n");
                return 1;
            }

            while (!g_stop && (max_utterances <= 0 || utter_idx < max_utterances)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            fprintf(stderr, "\nStopping...\n");
            cap.stop();
            ctx.unload();
            return 0;
        }

        if (req.ref_audio_path.empty()) {
            fprintf(stderr, "ERROR: --embed-only requires --ref-audio (or --interactive / --live-mic)\n");
            return 1;
        }

        std::vector<float> audio_samples;
        if (!qwen3_tts_read_wav(req.ref_audio_path.c_str(), audio_samples)) {
            fprintf(stderr, "ERROR: cannot read reference audio\n");
            return 1;
        }

        std::vector<float> embedding;
        if (!ctx.extract_speaker_embedding(audio_samples, embedding)) {
            fprintf(stderr, "ERROR: speaker embedding extraction failed (no speaker encoder weights in %s?)\n",
                    talker_path.c_str());
            return 1;
        }

        float rms = 0.0f;
        for (float v : embedding) rms += v * v;
        rms = sqrtf(rms / embedding.size());
        fprintf(stderr, "Speaker embedding: dim=%d, rms=%.6f\n", (int)embedding.size(), rms);

        const std::string embed_output = (output_path == "output.wav") ? "embedding.bin" : output_path;
        FILE * f = fopen(embed_output.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "ERROR: cannot write %s\n", embed_output.c_str());
            return 1;
        }
        fwrite(embedding.data(), sizeof(float), embedding.size(), f);
        fclose(f);
        printf("Wrote embedding to %s (%d floats, %zu bytes)\n",
               embed_output.c_str(), (int)embedding.size(), embedding.size() * sizeof(float));
        ctx.unload();
        return 0;
    }

    if (cp_path.empty()) {
        fprintf(stderr, "ERROR: --model-cp is required\n");
        return 1;
    }
    if (req.text.empty()) {
        fprintf(stderr, "ERROR: --text is required\n");
        return 1;
    }
    if (stream && vocoder_path.empty()) {
        fprintf(stderr, "ERROR: --stream requires --model-vocoder\n");
        return 1;
    }

    qwen3_tts_ctx ctx;
    if (!ctx.load(talker_path, cp_path, vocoder_path, n_gpu)) {
        fprintf(stderr, "ERROR: failed to load TTS models\n");
        return 1;
    }

    if (vocoder_path.empty()) {
        fprintf(stderr, "NOTE: --model-vocoder not specified; codec codes will be generated but no WAV written.\n");
    }

    if (stream) {
        // Reserve stdout exclusively for raw PCM; all logging goes to stderr.
#ifdef _WIN32
        _setmode(_fileno(stdout), _O_BINARY);
#endif
        auto write_pcm = [](std::vector<float> chunk) -> bool {
            if (chunk.empty()) return true;
            std::vector<uint8_t> bytes = qwen3_tts_pcm_to_s16_bytes(chunk);
            size_t written = fwrite(bytes.data(), 1, bytes.size(), stdout);
            fflush(stdout);
            return written == bytes.size();
        };

        if (!ctx.generate_streaming(req, write_pcm, stream_chunk_frames)) {
            fprintf(stderr, "ERROR: TTS streaming generation failed\n");
            return 1;
        }
        return 0;
    }

    std::vector<float> audio;
    if (!ctx.generate(req, audio)) {
        fprintf(stderr, "ERROR: TTS generation failed\n");
        return 1;
    }

    if (!audio.empty()) {
        qwen3_tts_write_wav(output_path.c_str(), audio.data(), (int)audio.size());
    } else if (vocoder_path.empty()) {
        fprintf(stderr, "No audio output (vocoder not specified).\n");
    }

    return 0;
}
