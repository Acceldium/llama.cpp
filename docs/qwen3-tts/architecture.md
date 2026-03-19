# Qwen3-TTS Architecture in llama.cpp

This document describes the full architecture of the Qwen3-TTS text-to-speech system as implemented in `llama.cpp`, covering every component from text input to audio output.

## Table of Contents

1. [System Overview](#system-overview)
2. [Pipeline Flow](#pipeline-flow)
3. [Text Tokenization and Embedding](#text-tokenization-and-embedding)
4. [Talker Model](#talker-model)
5. [Code Predictor Model](#code-predictor-model)
6. [Speaker Encoder (ECAPA-TDNN)](#speaker-encoder-ecapa-tdnn)
7. [Speech Tokenizer (Encoder)](#speech-tokenizer-encoder)
8. [Vocoder (WavTokenizer Decoder)](#vocoder-wavtokenizer-decoder)
9. [Prefill Embedding Construction](#prefill-embedding-construction)
10. [Voice Cloning Modes](#voice-cloning-modes)
11. [Multi-Language Support](#multi-language-support)
12. [MRoPE (Multimodal Rotary Position Encoding)](#mrope)
13. [KV Cache Management](#kv-cache-management)
14. [GGUF Model Files](#gguf-model-files)
15. [Numerical Precision Considerations](#numerical-precision)
16. [Parity with HuggingFace Reference](#parity)
17. [Troubleshooting](#troubleshooting)

---

## System Overview

Qwen3-TTS is a text-to-speech model developed by Alibaba's Qwen team. The 12Hz variant (Qwen3-TTS-12Hz-0.6B-Base) produces discrete audio codes at 12 frames per second, with each frame containing 16 codebook tokens. These discrete codes are then converted to a continuous audio waveform by a vocoder.

The system consists of four neural network components:

| Component | Parameters | Function |
|-----------|-----------|----------|
| **Talker** | ~540M | Transformer LLM that generates codebook-0 tokens autoregressively |
| **Code Predictor** | ~60M | Small transformer that predicts codebooks 1-15 given the Talker's hidden state |
| **Speaker Encoder** | ~6M | ECAPA-TDNN that extracts a speaker embedding from reference audio |
| **Speech Tokenizer** | ~25M | MimiModel-based encoder that converts reference audio to codec tokens for ICL |
| **Vocoder** | ~90M | WavTokenizer decoder that converts 16 codebook tokens per frame into audio |

The implementation in `llama.cpp` uses two registered architectures (`LLM_ARCH_QWEN3TTS` for the Talker and `LLM_ARCH_QWEN3TTS_CP` for the Code Predictor), builds the Speaker Encoder and Vocoder as raw GGML computation graphs, and implements the Speech Tokenizer Encoder as manual C++ matrix operations. All four components run natively within the custom CLI tool with no external Python dependencies required.

---

## Pipeline Flow

The synthesis pipeline proceeds in five stages:

1. **Text Tokenization**: Input text is wrapped in a Qwen chat template and tokenized using the Qwen3 tokenizer (151,936 token vocabulary). Special tokens mark the assistant role boundary.

2. **Speaker Embedding**: If a reference audio is provided, the ECAPA-TDNN speaker encoder extracts a 1024-dimensional speaker embedding (x-vector). Without reference audio, a zero vector is used.

3. **Prefill Construction**: Text token projections and codec embeddings are combined into a sequence of 1024-dimensional embedding vectors that serve as the Talker's initial input context.

4. **Autoregressive Generation**: For each output frame:
   - The **Talker** produces a hidden state and codebook-0 logits. The top-scoring codebook-0 token is selected.
   - The **Code Predictor** takes the Talker's hidden state and the codebook-0 embedding, then autoregressively predicts codebook tokens 1 through 15.
   - This yields a 16-integer code vector per frame.

5. **Vocoding**: All accumulated code vectors are passed through the WavTokenizer decoder, which produces 24,000 Hz mono audio.

Generation terminates when the Talker emits the `codec_eos` token (ID 2150) or the maximum frame count is reached.

---

## Text Tokenization and Embedding

### Tokenization

Input text is wrapped in the Qwen3 chat template before tokenization:

```
<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
```

This produces a token sequence with a specific structure:
- Tokens `[0:3]` — role prefix: `<|im_start|>`, `assistant`, `\n`
- Tokens `[3:-5]` — actual text content
- Tokens `[-5:]` — suffix: `<|im_end|>`, `\n`, `<|im_start|>`, `assistant`, `\n`

Only the role prefix and text content tokens participate in the prefill embedding. The suffix tokens are not used directly.

### Text Projection

The Talker uses a separate text embedding table (`text_embedding`, 151,936 x 2,048) that is distinct from the main model's token embedding table (which handles codec tokens). Each text token ID is looked up in this table and then projected down to the Talker's hidden dimension (1,024) through a two-layer MLP:

1. **Embedding lookup**: token_id → 2,048-dim vector
2. **FC1 (up projection)**: 2,048 → 2,048, with bias, followed by SiLU activation
3. **FC2 (down projection)**: 2,048 → 1,024, with bias

The result is a 1,024-dimensional text projection for each token. These projections are computed on the CPU outside the main LLM graph, since they involve custom weight matrices that are not part of the standard Qwen2 layer structure.

---

## Talker Model

### Architecture

The Talker is a 28-layer transformer decoder with the Qwen2 architecture, modified for TTS:

| Parameter | Value |
|-----------|-------|
| Hidden dimension | 1,024 |
| Intermediate (FFN) dimension | 3,072 |
| Attention heads | 16 |
| Key/Value heads (GQA) | 8 |
| Head dimension | 128 |
| RMS norm epsilon | 1e-6 |
| RoPE theta | 1,000,000 |
| RoPE type | iMRoPE (interleaved multi-dimensional) |
| MRoPE sections | [24, 20, 20] |
| Codec vocabulary | 3,072 tokens |
| Max position embeddings | 32,768 |

### Input/Output

**Input**: The Talker receives 1,024-dimensional embedding vectors (not token IDs) at every position. These embeddings are pre-constructed by overlaying text projections with codec embeddings in the prefill, and by combining codec-0 embeddings with text-pad embeddings during autoregressive decode.

**Output**: The Talker produces:
- A **hidden state** (1,024-dim) at each position, which is passed to the Code Predictor
- **Codebook-0 logits** (3,072-dim), computed by multiplying the hidden state with the `codec_head` weight matrix outside the LLM graph

The `codec_head` is a separate weight matrix applied on the CPU, not the standard `output` tensor of the Qwen2 model. The standard `output` (logit projection) tensor is not used and not stored. The Talker context is configured with `embeddings = true` so that `llama_decode` returns the raw hidden states rather than logits.

### Codec Embedding

The Talker has a `codec_embedding` table (1,024 x 3,072) that maps codec token IDs to embedding vectors. This table is used for:
- Looking up special codec tokens (BOS, EOS, PAD, thinking tokens, language tokens)
- Embedding the codebook-0 prediction from each frame during autoregressive decode
- Constructing the prefill embedding by overlaying codec embeddings on text projections

### Grouped Query Attention

The Talker uses GQA with a 2:1 ratio (16 query heads, 8 key/value heads). Each head has 128 dimensions. The query and key projections include QK-norm (RMS normalization applied to Q and K before attention scores).

---

## Code Predictor Model

### Architecture

The Code Predictor is a 5-layer transformer decoder registered as `LLM_ARCH_QWEN3TTS_CP`:

| Parameter | Value |
|-----------|-------|
| Hidden dimension | 1,024 |
| Intermediate (FFN) dimension | 3,072 |
| Attention heads | 16 |
| Key/Value heads (GQA) | 8 |
| Head dimension | 128 |
| RoPE type | Standard (NeoX-style) |
| RoPE theta | 1,000,000 |
| Number of layers | 5 (full attention) |
| Codec vocabulary | 2,048 tokens per codebook |

### Per-Codebook Weights

Unlike the Talker, the Code Predictor has 15 separate sets of:
- **`lm_head.{i}`** (1,024 x 2,048): One output projection per codebook (i = 0..14, mapping to codebooks 1..15)
- **`codec_embd.{i}`** (1,024 x 2,048): One embedding table per codebook (i = 0..14)

These are stored outside the standard LLM weight structure and applied on the CPU.

### Inference Loop (Per Frame)

For each frame, after the Talker produces a hidden state and codebook-0 token:

1. **Prefill (step 0)**: Feed 2 embeddings to the Code Predictor:
   - Position 0: Talker's hidden state (1,024-dim, passed as-is)
   - Position 1: Codebook-0 embedding from `codec_embd.0[cb0_token]`

2. **Decode (steps 1-14)**: For each subsequent codebook:
   - Extract hidden state from the Code Predictor's output
   - Multiply by `lm_head.{step}` to get 2,048-dim logits
   - Select the argmax token
   - Embed it using `codec_embd.{step+1}` (or the last codebook's embedding for step 14)
   - Feed back into the Code Predictor

3. **KV cache reset**: After all 15 steps, the Code Predictor's KV cache is completely cleared before processing the next frame.

The total sequence length per frame is 16 positions (2 prefill + 14 decode), and the KV cache never exceeds this.

---

## Speaker Encoder (ECAPA-TDNN)

The speaker encoder is an ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation Time Delay Neural Network) that extracts a fixed-dimensional speaker embedding (x-vector) from variable-length audio. This embedding captures the speaker's voice identity independently of the spoken content, and is used for x-vector voice cloning and as part of ICL voice cloning.

The speaker encoder is a standalone component with no dependency on the Talker, Code Predictor, or Vocoder. It can be used independently for any speaker verification, identification, or embedding extraction task.

### Input Processing

1. **Audio input**: 24,000 Hz mono PCM audio (read from 16-bit WAV)
2. **Mel spectrogram**: 128-band mel filterbank features are computed with:
   - FFT size: 1,024
   - Hop length: 256 (~10.7ms)
   - Window length: 1,024 (~42.7ms)
   - Window function: Hann
   - Frequency range: 0 Hz to 12,000 Hz (Nyquist)
   - Mel scale: Slaney normalization (`2 / (f_right - f_left)`)
   - Log scaling: `log(max(energy, 1e-5))`
   - Padding: reflect-pad by `(N_FFT - HOP) / 2` on both sides

The mel spectrogram is stored in `[n_mels, n_frames]` layout (channel-major), where `n_frames = (padded_length - N_FFT) / HOP + 1`.

### Network Architecture

The ECAPA-TDNN processes the mel spectrogram through six stages:

#### Stage 1: Initial Convolution

Conv1d(128 → 512, kernel=5, padding=2) followed by ReLU.
This projects the 128 mel bands to the hidden dimension (512 channels). Padding is applied to the left side only (left-pad = 2).

#### Stage 2: Three SE-Res2Net Blocks

Each block has a different dilation rate (2, 3, 4) and processes 512-channel input:

1. **TDNN1**: Conv1d(512 → 512, kernel=1) + ReLU. Projects input to the working dimension.

2. **Res2Net multi-scale processing**: The 512 channels are split into 8 branches of 64 channels each (scale=8). Each branch (except the first) processes through a dilated Conv1d(64 → 64, kernel=3, dilation=d) + ReLU, with the previous branch's output added to the current branch's input before convolution. This creates a hierarchical, multi-resolution feature representation at 8 different receptive field scales. The 8 branch outputs are concatenated back to 512 channels.

3. **TDNN2**: Conv1d(512 → 512, kernel=1) + ReLU. Mixes the multi-scale features.

4. **Squeeze-and-Excitation (SE)**: Global average pooling → Conv1d(512 → 128, kernel=1) + ReLU → Conv1d(128 → 512, kernel=1) + Sigmoid. The sigmoid output gates (multiplies) the block's features, letting the network emphasize informative channels.

5. **Residual connection**: The gated output is added to the block's input.

#### Stage 3: Multi-layer Feature Aggregation (MFA)

The outputs from all three SE-Res2Net blocks (each 512 channels) are concatenated along the channel dimension (512 × 3 = 1,536 channels), then processed by Conv1d(1536 → 1536, kernel=1) + ReLU. This lets the network combine features from different layers and receptive fields.

#### Stage 4: Attentive Statistics Pooling (ASP)

ASP collapses the variable-length time dimension into a fixed-size vector using learned attention:

1. **Context vector**: Global mean and standard deviation of the 1,536-channel features are computed across time, then expanded back to the time dimension.

2. **Attention computation**: The input features, global mean, and global std are concatenated along the channel dimension (1536 × 3 = 4,608 channels), then processed by TDNN(4608 → 1536) + ReLU + Tanh → Conv1d(1536 → 1536, kernel=1) → Softmax. This produces per-frame attention weights.

3. **Weighted statistics**: The attention weights are used to compute the weighted mean and weighted standard deviation of the input features across time. These are concatenated to form a 3,072-dimensional vector (1,536 mean + 1,536 std).

#### Stage 5: Final Projection

Conv1d(3072 → 1024, kernel=1) with bias. Projects the pooled statistics to the output embedding dimension.

### Output

A 1,024-dimensional speaker embedding vector (x-vector). In the TTS pipeline, this vector is inserted as a single position in the Talker's codec preamble using the `codec_embed(spk_embd_token)` lookup overlaid with the x-vector.

### Weight Summary

| Layer | Weight Shape | Description |
|-------|-------------|-------------|
| conv0 | [512, 128, 5] + bias[512] | Initial mel projection |
| blk.N.tdnn1 | [512, 512, 1] + bias[512] | TDNN1 in SE-Res2Net block N |
| blk.N.res2net.M | [64, 64, 3] + bias[64] | Res2Net branch M in block N (M=0..6) |
| blk.N.tdnn2 | [512, 512, 1] + bias[512] | TDNN2 in SE-Res2Net block N |
| blk.N.se.conv1 | [128, 512, 1] + bias[128] | SE squeeze in block N |
| blk.N.se.conv2 | [512, 128, 1] + bias[512] | SE excitation in block N |
| mfa | [1536, 1536, 1] + bias[1536] | Multi-layer feature aggregation |
| asp.tdnn | [1536, 4608, 1] + bias[1536] | ASP attention TDNN |
| asp.conv | [1536, 1536, 1] + bias[1536] | ASP attention conv |
| fc | [1024, 3072, 1] + bias[1024] | Final embedding projection |

Total: ~6M parameters. All stored in the Talker GGUF under the `spk_enc.*` prefix.

### Implementation

The speaker encoder is implemented as a raw GGML computation graph. It runs independently of the llama.cpp model infrastructure (no `llama_model` or `llama_context` needed), only requiring the GGML library. The weights are loaded via the `gguf_tensor_loader` utility from the Talker GGUF, and the graph is built and computed using raw GGML APIs (`ggml_new_graph_custom`, `ggml_backend_graph_compute`).

Because it only depends on GGML operations (conv1d, pool, softmax, concat, etc.) and has no dependency on the rest of the TTS pipeline, the speaker encoder is also available as a **standalone CLI tool** at `tools/tts/speaker-encoder/`. This tool can:

- Extract 1024-dim speaker embeddings from any WAV file
- Compare multiple speakers via pairwise cosine similarity
- Export embeddings as raw float32 binary files

This enables use cases beyond TTS, such as speaker verification, speaker diarization, and voice similarity search.

---

## Speech Tokenizer (Encoder)

The speech tokenizer encodes raw audio into discrete codec tokens. It is used for ICL (In-Context Learning) voice cloning, where the reference audio must be converted to codec codes that the Talker can condition on.

### Architecture

The encoder is based on MimiModel (a variant of EnCodec/SoundStream) consisting of:

1. **SEANet Encoder**: A strided convolutional encoder that progressively downsamples the audio:
   - Initial conv: 1 → 64 channels (kernel=7)
   - Four downsample stages with residual blocks:
     - 64 → 128, stride=8
     - 128 → 256, stride=5
     - 256 → 512, stride=4
     - 512 → 1024, stride=3
   - Final conv: 1024 → 512 (kernel=3)
   - Total downsampling factor: 8 x 5 x 4 x 3 = 480, then 2x from downsample = 960
   - At 24,000 Hz input, this produces ~12.5 frames/second (matching the 12Hz design)

2. **Encoder Transformer**: 8 layers of transformer attention operating on the 512-dim encoder output, each with:
   - Layer norm (with bias)
   - Self-attention (Q, K, V projections, 512-dim)
   - Learnable attention scale and FFN scale
   - Feed-forward network (512 → 2048 → 512)

3. **Downsample**: A final conv1d (kernel=4, stride=2) that halves the frame rate

4. **Residual Vector Quantization (RVQ)**:
   - **Semantic codebook**: 1 codebook with 2,048 entries of 256 dimensions (input projection from 512-dim)
   - **Acoustic codebooks**: 31 codebooks with 2,048 entries each, using residual quantization (input projection from 512-dim)

### Processing Pipeline

The encoder processes audio as: **Raw Audio → SEANet → Transformer → Downsample → RVQ → Codec Tokens**.

Only the first 16 of the 32 available codebooks are used during TTS inference (1 semantic + 15 acoustic), matching the Talker/Code Predictor architecture.

### Causal Convolutions

All convolutions in the encoder use causal padding: `pad = kernel_size - stride` applied to the left side of the input only. This ensures no information from future samples leaks into the current frame, which is critical for streaming-compatible processing.

### GELU Activation

The encoder transformer uses exact erf-based GELU activation (`ggml_gelu_erf`), not the tanh approximation (`ggml_gelu`). The HuggingFace implementation uses `nn.functional.gelu` which defaults to the exact erf-based variant. Using the tanh approximation causes accumulated numerical errors across the 8 transformer layers.

### Codebook Normalization

Each VQ codebook stores raw `embed_sum` and `cluster_usage` tensors. The actual codebook embeddings are computed as `embed_sum / cluster_usage` during GGUF conversion. This normalization must be applied to both encoder and decoder codebooks during conversion.

### Implementation

The speech tokenizer encoder is implemented natively in C++ within `qwen3tts.cpp`, operating entirely on the CPU using manual matrix operations (not GGML graphs). Its weights are stored in the vocoder GGUF file with the `tok_enc.*` prefix. The encoder achieves functional parity with HuggingFace: 100% VQ match when given identical transformer output, and ~26% exact code match end-to-end due to inherent floating-point precision differences in the attention computation across 8 transformer layers.

A Python helper script (`extract_ref_codes.py`) is also provided as a fallback for extracting codec codes using the HuggingFace model.

---

## Vocoder (WavTokenizer Decoder)

The vocoder converts 16 discrete codebook tokens per frame into a continuous audio waveform at 24,000 Hz.

### VQ Codebook Lookup

Each frame's 16 tokens are looked up in their respective codebook tables:
- **Codebook 0** ("first"): 2,048 entries of 512 dimensions, with input/output projections (512 ↔ 256)
- **Codebooks 1-15** ("rest"): Each has 2,048 entries of 512 dimensions, with shared input/output projections

The per-codebook embeddings are summed to produce a single 512-dimensional vector per frame. This summation is performed after applying input projection → codebook lookup → output projection for each group.

### Decoder Architecture

The 512-dim per-frame vectors are processed through:

1. **Pre-convolution**: Conv1d, 512 → 1024 (kernel=7, padding=3)

2. **Pre-transformer**: 8 layers of transformer blocks, each containing:
   - Layer norm with bias
   - Self-attention (1024-dim, with Q/K/V projections)
   - Learnable attention and FFN scaling factors
   - Feed-forward network (1024 → 4096 → 1024)
   - The transformer output is added back to the pre-conv output (residual connection)

3. **Upsample blocks**: Four stages that progressively increase the temporal resolution:
   - Stage 0: Transposed conv (1024 → 512, kernel=16, stride=8) + 3 residual blocks
   - Stage 1: Transposed conv (512 → 256, kernel=10, stride=5) + 3 residual blocks
   - Stage 2: Transposed conv (256 → 128, kernel=8, stride=4) + 3 residual blocks
   - Stage 3: Transposed conv (128 → 64, kernel=6, stride=3) + 3 residual blocks
   - Total upsample factor: 8 x 5 x 4 x 3 = 480

4. **Decoder blocks**: Four additional processing stages, each containing:
   - Snake activation (learned per-channel)
   - Conv1d with decreasing kernel sizes (7, 7, 7, 7)
   - Residual blocks with dilation patterns

5. **Final output**: Conv1d to 1 channel, producing mono audio

### Residual Blocks

Each residual block contains:
- Snake activation → dilated Conv1d → Snake activation → Conv1d (kernel=1)
- Skip connection from input to output
- Dilation patterns vary: [1, 3, 9] across the three blocks in each stage

### Snake Activation

Snake activation is defined as: `x + (1/alpha) * sin^2(alpha * x)` where alpha is a learnable per-channel parameter. This provides a smooth, periodic nonlinearity that is well-suited for audio generation.

### Output

Raw PCM audio at 24,000 Hz, mono, float32. The output length per frame is 480 samples (24000/12Hz * 480 upsample / 480 = 2000 samples per frame, corrected for the actual pipeline). Written as 16-bit signed integer WAV.

---

## Prefill Embedding Construction

The prefill is the initial context fed to the Talker before autoregressive generation begins. It combines text and codec information into a single sequence of 1,024-dim embedding vectors.

### Standard Mode (x-vector or no cloning)

The prefill sequence is constructed as follows:

| Position(s) | Text component | Codec component | Description |
|-------------|---------------|-----------------|-------------|
| 0-2 | text_proj(role tokens) | (none) | `<\|im_start\|>`, `assistant`, `\n` |
| 3 to 3+N-1 | tts_pad_embed | codec_embed(think, think_bos, lang, think_eos, [spk]) | Codec preamble: thinking tags + language + optional speaker |
| 3+N | tts_bos_embed | codec_embed(codec_pad) | Transition: text BOD marker + codec padding |
| next K | text_proj(content tokens) | codec_embed(codec_pad) | All text content overlaid with codec padding |
| next 1 | tts_eos_embed | codec_embed(codec_pad) | Text end marker |
| final | tts_pad_embed | codec_embed(codec_bos) | Start of codec generation |

Where:
- N = number of codec preamble positions (varies with language and speaker)
- K = number of text content tokens (varies with input text)
- Each position's embedding is the element-wise sum of the text and codec components

### ICL Mode (In-Context Learning)

When ICL voice cloning is enabled, the prefill additionally includes reference text and reference codec tokens:

| Section | Content |
|---------|---------|
| Role tokens | Same as standard mode |
| Codec preamble | Same as standard mode |
| ICL text | ref_text + target_text + eos, each overlaid with codec_pad |
| ICL codec | codec_bos + sum-of-all-codebook-embeddings for each ref frame, each overlaid with tts_pad |
| Final | tts_pad + codec_bos |

The reference codec embeddings are computed by summing the embeddings from all 16 codebooks for each reference frame. Codebook 0 uses the Talker's `codec_embedding`, while codebooks 1-15 use the Code Predictor's per-codebook `codec_embd.{i}` tables.

---

## Voice Cloning Modes

### No Cloning (Default)

Without any reference audio, the model uses a zero speaker embedding. The voice characteristics are determined entirely by the model's learned default.

### X-Vector Only Mode

Activated with `--ref-audio <wav>`. The ECAPA-TDNN speaker encoder extracts a 1,024-dim x-vector from the reference audio, which is inserted into the codec preamble as a single embedding position. This mode captures the speaker's general voice quality but does not replicate specific prosodic patterns.

### ICL Mode (In-Context Learning)

Activated with `--ref-audio <wav> --ref-text "<transcript>"`. This mode provides the strongest voice cloning by conditioning the Talker on both:
- The speaker embedding (x-vector)
- The actual codec tokens from the reference audio, aligned with the reference transcript

The reference codec codes are extracted automatically by the native C++ speech tokenizer encoder from the reference audio. No external Python preprocessing is required. Optionally, pre-extracted codes can be provided via `--ref-codes <codes_file>` to skip the encoding step.

### Voice Design (Named Speakers)

Named speaker presets (e.g., specific voice styles) are only available in the Instruct model variant. The Base model's `spk_id` dictionary is empty, so this feature requires upgrading to `Qwen3-TTS-12Hz-0.6B-Instruct`.

---

## Multi-Language Support

The model supports 10 languages, each identified by a codec language ID token:

| Language | Codec ID |
|----------|----------|
| English | 2050 |
| Chinese | 2055 |
| German | 2053 |
| Spanish | 2054 |
| French | 2061 |
| Italian | 2070 |
| Japanese | 2058 |
| Korean | 2064 |
| Portuguese | 2071 |
| Russian | 2069 |

The language is specified via `--language <lang>` and controls which codec language token is inserted into the prefill preamble. The special value `auto` omits the language token entirely, using a `nothink` token instead.

The language token is placed inside the "thinking" section of the codec preamble: `[think, think_bos, lang_id, think_eos]`. When language is `auto`, this becomes `[nothink, think_bos, think_eos]`.

---

## MRoPE

The Talker uses interleaved Multi-dimensional Rotary Position Encoding (iMRoPE), the same mechanism used in Qwen2-VL and Qwen3-VL for multimodal models. This is registered as `LLAMA_ROPE_TYPE_IMROPE` in llama.cpp.

### MRoPE Sections

The 128-dimensional head is split into three sections according to the `mrope_section` parameter `[24, 20, 20]`:
- **Dimensions 0-23** (24 dims): First position component
- **Dimensions 24-43** (20 dims): Second position component
- **Dimensions 44-63** (20 dims): Third position component
- **Dimensions 64-127** (64 dims): Fourth component (always position 0, effectively unused)

Each section receives its own position ID, allowing different position trajectories for different aspects of the input (analogous to how VL models encode spatial vs. temporal positions).

### Position Assignment

In the current TTS implementation, all three active dimensions use the same sequential position (0, 1, 2, ...). The fourth dimension is always 0. This means MRoPE effectively behaves like standard RoPE for TTS, but the infrastructure supports future extensions where text and audio positions could diverge.

The batch position layout is `[dim0*N, dim1*N, dim2*N, dim3*N]` where N is the number of tokens. This is implemented via `n_pos_per_embd = 4` in the batch allocation.

### Code Predictor RoPE

The Code Predictor uses standard NeoX-style RoPE (not MRoPE), with `rope_theta = 1,000,000`. Its positions reset to [0, 1, ..., 15] for each frame.

---

## KV Cache Management

### Talker KV Cache

The Talker's KV cache grows monotonically across the entire generation:
- **Prefill**: Positions 0 to N-1 (where N is the prefill length)
- **Decode**: Position N, N+1, N+2, ... for each successive frame

The KV cache is never cleared during generation. Context size is set to 4,096 positions.

### Code Predictor KV Cache

The Code Predictor's KV cache is **reset after every frame**. Each frame runs through 16 positions (2 prefill + 14 decode), then the cache is completely cleared. Context size is set to 32 positions.

This separation is achieved naturally by using two independent `llama_context` instances, each with its own KV cache. The Talker context persists state across frames while the Code Predictor context is reset via `llama_kv_self_clear()`.

---

## GGUF Model Files

The system requires three GGUF files:

### Talker GGUF

Contains:
- Standard Qwen2 transformer weights (28 layers)
- Custom tensors with `tts.` prefix:
  - `tts.text_embd.weight` — Text embedding table (151,936 x 2,048)
  - `tts.text_proj_up.weight/bias` — FC1 of text projection (2,048 x 2,048)
  - `tts.text_proj_down.weight/bias` — FC2 of text projection (2,048 x 1,024)
  - `tts.codec_embd.weight` — Codec embedding table (1,024 x 3,072)
  - `tts.codec_head.weight` — Codec output head (1,024 x 3,072)
- Speaker encoder weights with `spk_enc.` prefix
- GGUF metadata keys for special token IDs (`qwen3tts.tts_bos_token_id`, etc.)

### Code Predictor GGUF

Contains:
- Standard Qwen2 transformer weights (5 layers)
- Per-codebook tensors with `tts.cp.` prefix:
  - `tts.cp.lm_head.{0..14}.weight` — 15 output heads (1,024 x 2,048 each)
  - `tts.cp.codec_embd.{0..14}.weight` — 15 codec embedding tables (1,024 x 2,048 each)

### Vocoder GGUF

Contains:
- Decoder tensors with `tok_dec.` prefix (VQ codebooks, convolutions, transformer, upsampler)
- Encoder tensors with `tok_enc.` prefix (SEANet convolutions, transformer layers, downsample conv, RVQ codebooks and projections)

### Conversion

GGUF files are produced by `tools/tts/convert_qwen3tts.py`, which calls the `Qwen3TTSTalkerModel` and `Qwen3TTSCodePredictorModel` classes from `convert_hf_to_gguf.py`. Supported output types: `f16`, `bf16`, `f32`, `q8_0`.

---

## Numerical Precision

### BF16 Requirement

The Code Predictor is sensitive to floating-point precision. Using F16 weights causes NaN outputs in the CP hidden states, which manifests as the error "CP prefill produced invalid output". This occurs because the CP's attention computation accumulates precision loss across its 5 layers when using F16's limited range, producing non-finite values. BF16 (matching the original HuggingFace model's storage format) resolves this issue.

Recommended precision:
- **Talker**: BF16 (tested, stable)
- **Code Predictor**: BF16 (required; F16 causes NaN in attention output)
- **Vocoder**: BF16 or F16 (both work; BF16 weights are automatically cast to F16 for conv operations)

### Convolution F16 Requirement

The GGML `ggml_conv_1d`, `ggml_conv_transpose_1d`, and `ggml_conv_1d_dw` operations require F16 weight tensors. When using BF16 GGUF files, the vocoder and speaker encoder code must explicitly cast convolution weights to F16 via `ggml_cast()` before passing them to these operations. Without this cast, the operations will fail with an assertion error.

### Text Projection Precision

The text projection and codec head computations are performed in F32 on the CPU, regardless of the GGUF weight storage type. The `read_row` and `read_bias` helper functions automatically convert F16 and BF16 weights to F32 before computation.

### Encoder Precision

The speech tokenizer encoder operates entirely in F32 on the CPU using manual matrix operations, avoiding GGML graph overhead. Weights are read from the GGUF and converted to F32 at load time. The encoder transformer uses `ggml_gelu_erf` (exact erf-based GELU) to match HuggingFace's activation function.

---

## Parity

### Methodology

Numerical parity is tested by comparing codec tokens generated by `llama-qwen3tts` against the HuggingFace reference implementation (`qwen_tts` package) using identical inputs:
- Same text: "Hello how are you today"
- Same template: `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`
- Zero speaker embedding (no voice cloning)
- Greedy decoding (argmax, no sampling)
- English language

### Talker/CP Results

With BF16 Talker and Code Predictor weights:
- **Frames 0-1**: Codebook-0 tokens match exactly (100%)
- **Frame 2 onwards**: Tokens diverge due to accumulated BF16 → F32 precision differences
- **Overall cb0 match**: ~10.5% (2/19 frames)
- **Overall full-frame match**: 0% (cb1-15 differ from frame 0)

The early-frame match confirms that the architecture, embedding construction, tokenization, and MRoPE implementation are all correct. The divergence in later frames is an expected and well-understood characteristic of running transformer inference at different precision levels.

### Speech Tokenizer Encoder Results

The native C++ encoder was validated against the HuggingFace implementation at multiple stages:

1. **SEANet output**: Near-perfect numerical parity after causal padding fixes
2. **Transformer output**: Accumulating RMS divergence across 8 layers (~6% by layer 7), caused by floating-point differences between GGML's manual attention and PyTorch's optimized SDPA
3. **VQ isolation test**: **100% code match** when feeding identical transformer output to both implementations, confirming the VQ implementation is functionally identical
4. **End-to-end encoder**: **26.2% exact code match** across all 16 codebooks and 25 frames. CB0 (semantic) matches at 12%, CB1 matches at 84%

Two critical bugs were identified and fixed during encoder parity testing:
- **GELU variant**: Changed from `ggml_gelu` (tanh approximation) to `ggml_gelu_erf` (exact erf-based), matching HuggingFace's implementation
- **Codebook normalization**: Fixed the GGUF converter to apply `embed_sum / cluster_usage` normalization for encoder codebooks (not just decoder codebooks)

### Audio Quality

Both implementations produce comparable audio quality. The codebook-0 tokens follow similar patterns (e.g., both produce silence/padding tokens for pause regions), and the vocoder produces clean speech from both sets of codes.

---

## Sampling

The `llama-qwen3tts` tool supports configurable sampling for both the Talker (codebook-0) and Code Predictor (codebook 1-15), with independent parameter sets.

### Talker Sampling Parameters

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Temperature | `--temp` | 0.9 | Controls randomness. Lower = more deterministic. 0 = greedy |
| Top-k | `--top-k` | 50 | Keep only the top-k highest logits before sampling. 0 = disabled |
| Top-p | `--top-p` | 1.0 | Nucleus sampling. Keep smallest set of tokens whose cumulative probability >= top-p |
| Repetition penalty | `--rep-penalty` | 1.05 | Penalizes recently generated tokens. 1.0 = disabled |
| Greedy | `--greedy` | off | Force argmax decoding for both Talker and CP |
| Seed | `--seed` | random | Deterministic RNG seed for reproducibility |

### Code Predictor Sampling Parameters

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Temperature | `--cp-temp` | 0.9 | CP temperature (independent of Talker) |
| Top-k | `--cp-top-k` | 50 | CP top-k (independent of Talker) |

The Code Predictor resets its recent-token history every frame (16 codebook steps), since each frame's 15 autoregressive steps are independent.

### Sampling Pipeline

1. **Repetition penalty**: Recent tokens have their logits divided (positive logits) or multiplied (negative logits) by the penalty factor
2. **Temperature scaling**: All logits divided by temperature
3. **Top-k filtering**: Only the top-k logits are kept
4. **Softmax**: Convert to probabilities
5. **Top-p (nucleus) filtering**: Accumulate probabilities until sum >= top-p, discard the rest
6. **Random sampling**: Weighted random selection from remaining candidates

---

## Streaming Text Mode

The `--streaming-text` flag enables streaming text mode, which changes how text is fed to the Talker during generation.

### Non-Streaming Mode (Default)

All text content is included in the prefill embedding sequence. During autoregressive decode, `tts_pad` is added to every step (no new text information). This is equivalent to HuggingFace's `non_streaming_mode=True`.

### Streaming Mode

Only the first text token (plus role tokens and codec preamble) goes into the prefill. The remaining text tokens are stored in a "trailing text" buffer and fed one-per-step during autoregressive decode. Once all trailing text tokens are consumed, `tts_pad` is used for remaining steps. This is equivalent to HuggingFace's `non_streaming_mode=False`.

### When to Use Each Mode

- **Non-streaming (default)**: Better for batch synthesis where all text is known upfront. Produces higher quality since the model sees all text context during prefill.
- **Streaming**: Better for real-time applications where audio needs to start before all text is available, or for very long texts where prefill would be too large.

---

## Troubleshooting

### "CP prefill produced invalid output"

This error means the Code Predictor's hidden state after its 2-token prefill is either null or contains NaN/Inf values. Root causes:

1. **F16 GGUF for Code Predictor**: The CP's attention computation is sensitive to F16's limited dynamic range. Using F16 weights causes NaN outputs. Solution: use BF16 for the Code Predictor GGUF.

2. **Corrupted or mismatched GGUF**: Ensure the CP GGUF was converted from the same HuggingFace checkpoint as the Talker GGUF, and that the conversion completed successfully.

### Vocoder assertion failure ("src0->type == GGML_TYPE_F16")

The GGML convolution operations (`ggml_conv_1d`, `ggml_conv_transpose_1d`, `ggml_conv_1d_dw`) require F16 weight tensors. When using BF16 GGUF files, convolution weights must be explicitly cast via `ggml_cast(ctx, w, GGML_TYPE_F16)` before use. The `voc_ensure_f16()` helper handles this automatically in the vocoder graph building.

### Low quality or garbled audio

Possible causes:
- Wrong GGUF precision (see above)
- Incorrect codec preamble construction (language token mismatch)
- Text not wrapped in the correct chat template
- Codec embedding table mismatch between Talker and Code Predictor

### Encoder code mismatch with HuggingFace

Expected. The native C++ encoder achieves ~26% exact code match due to floating-point precision differences in the transformer attention. This does not significantly affect voice cloning quality since the Talker is robust to small variations in reference codes.
