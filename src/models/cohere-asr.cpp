#include "models.h"

// CohereLabs/cohere-transcribe-03-2026 : NeMo-style Conformer encoder (48L,
// d=1280, rel-pos MHSA) -> Linear proj -> Transformer decoder (8L, d=1024,
// self-attn + cross-attn + ReLU FFN, fixed sinusoidal position embedding).
//
// The encoder's rel-pos attention (Transformer-XL matrix_ac/matrix_bd +
// rel_shift) does not fit the generic build_attn() helper (same reason noted
// in tools/mtmd/models/conformer.cpp: matrix_ac and matrix_bd need to be
// computed and combined before any masking/softmax), so it is built here with
// raw ggml ops, mirroring that file's proven op sequence. The decoder reuses
// the standard KV-cache causal self-attention path for its self-attention,
// and T5's existing cross-attention machinery (build_attn_inp_cross /
// build_inp_cross_embd) for cross-attention.

void llama_model_cohere_asr::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

    hparams.dec_n_layer = hparams.n_layer();
    ml.get_key(LLM_KV_DECODER_BLOCK_COUNT, hparams.dec_n_layer, false);

    uint32_t dec_start_token_id = 0;
    if (ml.get_key(LLM_KV_DECODER_START_TOKEN_ID, dec_start_token_id, false)) {
        hparams.dec_start_token_id = dec_start_token_id;
    }

    ml.get_key(LLM_KV_ENCODER_EMBEDDING_LENGTH,       hparams.n_embd_enc);
    ml.get_key(LLM_KV_ENCODER_FEED_FORWARD_LENGTH,    hparams.n_ff_enc);
    ml.get_key(LLM_KV_ENCODER_ATTENTION_HEAD_COUNT,   hparams.n_head_enc);
    ml.get_key(LLM_KV_ENCODER_ATTENTION_KEY_LENGTH,   hparams.n_embd_head_k_enc);
    ml.get_key(LLM_KV_ENCODER_ATTENTION_VALUE_LENGTH, hparams.n_embd_head_v_enc);
    ml.get_key(LLM_KV_ENCODER_CONV_KERNEL_SIZE,       hparams.n_conv_kernel_enc);
    ml.get_key(LLM_KV_ENCODER_SUBSAMPLING_FACTOR,     hparams.n_subsampling_factor_enc);
    ml.get_key(LLM_KV_ENCODER_SUBSAMPLING_CHANNELS,   hparams.n_subsampling_channels_enc);
    ml.get_key(LLM_KV_ENCODER_FEATURES_COUNT,         hparams.n_mel_bins_enc);

    // encoder input is raw log-mel features, not token ids
    hparams.n_embd_inp_enc_impl = hparams.n_mel_bins_enc;

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_cohere_asr::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_enc        = hparams.n_embd_enc;
    const int64_t n_head_enc        = hparams.n_head_enc;
    const int64_t n_embd_head_enc   = hparams.n_embd_head_k_enc;
    const int64_t n_ff_enc          = hparams.n_ff_enc;
    const int64_t n_conv_kernel     = hparams.n_conv_kernel_enc;
    const int64_t n_conv_channels   = hparams.n_subsampling_channels_enc;
    const int64_t n_mel             = hparams.n_mel_bins_enc;
    const int64_t subsampling       = hparams.n_subsampling_factor_enc;
    const int64_t n_fft             = 512; // fixed: mel filterbank is [n_fft/2+1, n_mel]
    const int64_t dec_n_layer       = hparams.dec_n_layer;
    const int64_t max_seq_len       = 1024; // decoder fixed position table length

    // --- token embedding / output head (decoder side, generic fields) ---
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        // weights are tied to tok_embd in the original model
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }
    output_b = create_tensor(tn(LLM_TENSOR_OUTPUT, "bias"), {n_vocab}, TENSOR_NOT_REQUIRED);

    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);

    // --- decoder embedding: fixed sinusoidal position table + embedding LN ---
    pos_embd = create_tensor(tn(LLM_TENSOR_POS_EMBD, "weight"), {n_embd, max_seq_len}, 0);

    dec_embd_norm   = create_tensor(tn(LLM_TENSOR_DEC_EMBD_NORM, "weight"), {n_embd}, 0);
    dec_embd_norm_b = create_tensor(tn(LLM_TENSOR_DEC_EMBD_NORM, "bias"),   {n_embd}, 0);

    // --- mel frontend (fixed buffers copied from the checkpoint) ---
    mel_fb     = create_tensor(tn(LLM_TENSOR_ENC_MEL_FB,     "weight"), {n_fft / 2 + 1, n_mel}, 0);
    mel_window = create_tensor(tn(LLM_TENSOR_ENC_MEL_WINDOW, "weight"), {400}, 0);

    // --- Conformer subsampling stem (dw_striding, 3x stride-2) ---
    // conv indices: 0 = std conv (1->C), 1/3 = depthwise (C,k,k), 2/4 = pointwise (C->C,1,1)
    subsample_conv[0]   = create_tensor(tn(LLM_TENSOR_ENC_SUBSAMPLE_CONV, "weight", 0), {3, 3, 1, n_conv_channels}, 0);
    subsample_conv_b[0] = create_tensor(tn(LLM_TENSOR_ENC_SUBSAMPLE_CONV, "bias",   0), {n_conv_channels}, 0);
    subsample_conv[1]   = create_tensor(tn(LLM_TENSOR_ENC_SUBSAMPLE_CONV, "weight", 1), {3, 3, 1, n_conv_channels}, 0);
    subsample_conv_b[1] = create_tensor(tn(LLM_TENSOR_ENC_SUBSAMPLE_CONV, "bias",   1), {n_conv_channels}, 0);
    subsample_conv[2]   = create_tensor(tn(LLM_TENSOR_ENC_SUBSAMPLE_CONV, "weight", 2), {1, 1, n_conv_channels, n_conv_channels}, 0);
    subsample_conv_b[2] = create_tensor(tn(LLM_TENSOR_ENC_SUBSAMPLE_CONV, "bias",   2), {n_conv_channels}, 0);
    subsample_conv[3]   = create_tensor(tn(LLM_TENSOR_ENC_SUBSAMPLE_CONV, "weight", 3), {3, 3, 1, n_conv_channels}, 0);
    subsample_conv_b[3] = create_tensor(tn(LLM_TENSOR_ENC_SUBSAMPLE_CONV, "bias",   3), {n_conv_channels}, 0);
    subsample_conv[4]   = create_tensor(tn(LLM_TENSOR_ENC_SUBSAMPLE_CONV, "weight", 4), {1, 1, n_conv_channels, n_conv_channels}, 0);
    subsample_conv_b[4] = create_tensor(tn(LLM_TENSOR_ENC_SUBSAMPLE_CONV, "bias",   4), {n_conv_channels}, 0);

    const int64_t freq_sub = n_mel / subsampling;
    subsample_out   = create_tensor(tn(LLM_TENSOR_ENC_SUBSAMPLE_OUT, "weight"), {n_conv_channels * freq_sub, n_embd_enc}, 0);
    subsample_out_b = create_tensor(tn(LLM_TENSOR_ENC_SUBSAMPLE_OUT, "bias"),   {n_embd_enc}, 0);

    enc_output_proj   = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_PROJ, "weight"), {n_embd_enc, n_embd}, 0);
    enc_output_proj_b = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_PROJ, "bias"),   {n_embd}, 0);

    // --- Conformer encoder layers ---
    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.ffn1_norm_enc   = create_tensor(tn(LLM_TENSOR_ENC_FFN1_NORM, "weight", i), {n_embd_enc}, 0);
        layer.ffn1_norm_enc_b = create_tensor(tn(LLM_TENSOR_ENC_FFN1_NORM, "bias",   i), {n_embd_enc}, 0);
        layer.ffn1_up_enc     = create_tensor(tn(LLM_TENSOR_ENC_FFN1_UP,   "weight", i), {n_embd_enc, n_ff_enc}, 0);
        layer.ffn1_up_enc_b   = create_tensor(tn(LLM_TENSOR_ENC_FFN1_UP,   "bias",   i), {n_ff_enc}, 0);
        layer.ffn1_down_enc   = create_tensor(tn(LLM_TENSOR_ENC_FFN1_DOWN, "weight", i), {n_ff_enc, n_embd_enc}, 0);
        layer.ffn1_down_enc_b = create_tensor(tn(LLM_TENSOR_ENC_FFN1_DOWN, "bias",   i), {n_embd_enc}, 0);

        layer.attn_norm_enc   = create_tensor(tn(LLM_TENSOR_ENC_ATTN_NORM, "weight", i), {n_embd_enc}, 0);
        layer.attn_norm_enc_b = create_tensor(tn(LLM_TENSOR_ENC_ATTN_NORM, "bias",   i), {n_embd_enc}, 0);

        layer.wq_enc   = create_tensor(tn(LLM_TENSOR_ENC_ATTN_Q,   "weight", i), {n_embd_enc, n_embd_enc}, 0);
        layer.bq_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_Q,   "bias",   i), {n_embd_enc}, 0);
        layer.wk_enc   = create_tensor(tn(LLM_TENSOR_ENC_ATTN_K,   "weight", i), {n_embd_enc, n_embd_enc}, 0);
        layer.bk_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_K,   "bias",   i), {n_embd_enc}, 0);
        layer.wv_enc   = create_tensor(tn(LLM_TENSOR_ENC_ATTN_V,   "weight", i), {n_embd_enc, n_embd_enc}, 0);
        layer.bv_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_V,   "bias",   i), {n_embd_enc}, 0);
        layer.wo_enc   = create_tensor(tn(LLM_TENSOR_ENC_ATTN_OUT, "weight", i), {n_embd_enc, n_embd_enc}, 0);
        layer.bo_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_OUT, "bias",   i), {n_embd_enc}, 0);

        layer.attn_pos        = create_tensor(tn(LLM_TENSOR_ENC_ATTN_POS,        "weight", i), {n_embd_enc, n_embd_enc}, 0);
        layer.attn_pos_bias_u = create_tensor(tn(LLM_TENSOR_ENC_ATTN_POS_BIAS_U, "weight", i), {n_embd_head_enc, n_head_enc}, 0);
        layer.attn_pos_bias_v = create_tensor(tn(LLM_TENSOR_ENC_ATTN_POS_BIAS_V, "weight", i), {n_embd_head_enc, n_head_enc}, 0);

        layer.conv_ln_enc   = create_tensor(tn(LLM_TENSOR_ENC_CONV_LN, "weight", i), {n_embd_enc}, 0);
        layer.conv_ln_enc_b = create_tensor(tn(LLM_TENSOR_ENC_CONV_LN, "bias",   i), {n_embd_enc}, 0);
        layer.conv_pw1      = create_tensor(tn(LLM_TENSOR_ENC_CONV_PW1, "weight", i), {n_embd_enc, 2 * n_embd_enc}, 0);
        layer.conv_pw1_b    = create_tensor(tn(LLM_TENSOR_ENC_CONV_PW1, "bias",   i), {2 * n_embd_enc}, 0);
        layer.conv_dw       = create_tensor(tn(LLM_TENSOR_ENC_CONV_DW, "weight", i), {n_conv_kernel, n_embd_enc}, 0);
        layer.conv_dw_b     = create_tensor(tn(LLM_TENSOR_ENC_CONV_DW, "bias",   i), {n_embd_enc}, 0);
        layer.conv_bn_w     = create_tensor(tn(LLM_TENSOR_ENC_CONV_BN, "weight", i), {n_embd_enc}, 0);
        layer.conv_bn_b     = create_tensor(tn(LLM_TENSOR_ENC_CONV_BN, "bias",   i), {n_embd_enc}, 0);
        layer.conv_pw2      = create_tensor(tn(LLM_TENSOR_ENC_CONV_PW2, "weight", i), {n_embd_enc, n_embd_enc}, 0);
        layer.conv_pw2_b    = create_tensor(tn(LLM_TENSOR_ENC_CONV_PW2, "bias",   i), {n_embd_enc}, 0);

        layer.ffn_norm_enc   = create_tensor(tn(LLM_TENSOR_ENC_FFN_NORM, "weight", i), {n_embd_enc}, 0);
        layer.ffn_norm_enc_b = create_tensor(tn(LLM_TENSOR_ENC_FFN_NORM, "bias",   i), {n_embd_enc}, 0);
        layer.ffn_up_enc     = create_tensor(tn(LLM_TENSOR_ENC_FFN_UP,   "weight", i), {n_embd_enc, n_ff_enc}, 0);
        layer.ffn_up_enc_b   = create_tensor(tn(LLM_TENSOR_ENC_FFN_UP,   "bias",   i), {n_ff_enc}, 0);
        layer.ffn_down_enc   = create_tensor(tn(LLM_TENSOR_ENC_FFN_DOWN, "weight", i), {n_ff_enc, n_embd_enc}, 0);
        layer.ffn_down_enc_b = create_tensor(tn(LLM_TENSOR_ENC_FFN_DOWN, "bias",   i), {n_embd_enc}, 0);

        layer.norm_out_enc   = create_tensor(tn(LLM_TENSOR_ENC_NORM_OUT, "weight", i), {n_embd_enc}, 0);
        layer.norm_out_enc_b = create_tensor(tn(LLM_TENSOR_ENC_NORM_OUT, "bias",   i), {n_embd_enc}, 0);
    }

    // --- Transformer decoder layers ---
    for (int i = 0; i < dec_n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_DEC_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_DEC_ATTN_NORM, "bias",   i), {n_embd}, 0);

        layer.wq   = create_tensor(tn(LLM_TENSOR_DEC_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
        layer.bq = create_tensor(tn(LLM_TENSOR_DEC_ATTN_Q,   "bias",   i), {n_embd}, 0);
        layer.wk   = create_tensor(tn(LLM_TENSOR_DEC_ATTN_K,   "weight", i), {n_embd, n_embd}, 0);
        layer.bk = create_tensor(tn(LLM_TENSOR_DEC_ATTN_K,   "bias",   i), {n_embd}, 0);
        layer.wv   = create_tensor(tn(LLM_TENSOR_DEC_ATTN_V,   "weight", i), {n_embd, n_embd}, 0);
        layer.bv = create_tensor(tn(LLM_TENSOR_DEC_ATTN_V,   "bias",   i), {n_embd}, 0);
        layer.wo   = create_tensor(tn(LLM_TENSOR_DEC_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
        layer.bo = create_tensor(tn(LLM_TENSOR_DEC_ATTN_OUT, "bias",   i), {n_embd}, 0);

        layer.attn_norm_cross   = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.attn_norm_cross_b = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_NORM, "bias",   i), {n_embd}, 0);

        layer.wq_cross   = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
        layer.bq_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_Q,   "bias",   i), {n_embd}, 0);
        layer.wk_cross   = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_K,   "weight", i), {n_embd, n_embd}, 0);
        layer.bk_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_K,   "bias",   i), {n_embd}, 0);
        layer.wv_cross   = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_V,   "weight", i), {n_embd, n_embd}, 0);
        layer.bv_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_V,   "bias",   i), {n_embd}, 0);
        layer.wo_cross   = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
        layer.bo_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_OUT, "bias",   i), {n_embd}, 0);

        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_DEC_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_DEC_FFN_NORM, "bias",   i), {n_embd}, 0);
        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_DEC_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_DEC_FFN_UP,   "bias",   i), {n_ff}, 0);
        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_DEC_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_DEC_FFN_DOWN, "bias",   i), {n_embd}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_cohere_asr::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            return std::make_unique<graph<false>>(*this, params);
        default:
            GGML_ABORT("invalid graph type");
    };
}

template <>
ggml_tensor * llama_model_cohere_asr::graph<true>::build_inp_embd_enc() const {
    auto inp = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp_enc());

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp_enc(), n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * cur = inp->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp));

    return cur;
}

namespace {

// Fills the Transformer-XL sinusoidal relative-position table needed by the
// Conformer encoder's rel-pos attention. Sized [n_embd, 2*n_tokens_enc - 1]
// (n_tokens_enc = the post-subsampling encoder sequence length), computed the
// same way as RelPositionalEncoding._create_pe() in modeling_cohere_asr.py.
class llm_graph_input_pos_emb_enc : public llm_graph_input_i {
public:
    llm_graph_input_pos_emb_enc(int64_t n_embd, int64_t n_tokens_enc)
        : n_embd(n_embd), n_tokens_enc(n_tokens_enc) {}
    virtual ~llm_graph_input_pos_emb_enc() = default;

    void set_input(const llama_ubatch *) override {
        const int64_t n_pos = 2 * n_tokens_enc - 1;

        GGML_ASSERT(ggml_backend_buffer_is_host(pos_emb->buffer));

        float * data = (float *) pos_emb->data;

        for (int64_t p = 0; p < n_pos; ++p) {
            // positions run from (n_tokens_enc - 1) down to -(n_tokens_enc - 1)
            const double position = (double) (n_tokens_enc - 1 - p);
            for (int64_t i = 0; i < n_embd; i += 2) {
                const double div_term = std::exp(i * (-std::log(10000.0) / (double) n_embd));
                data[p * n_embd + i]     = (float) std::sin(position * div_term);
                if (i + 1 < n_embd) {
                    data[p * n_embd + i + 1] = (float) std::cos(position * div_term);
                }
            }
        }
    }

    ggml_tensor * pos_emb = nullptr; // F32 [n_embd, 2*n_tokens_enc - 1]

    const int64_t n_embd;
    const int64_t n_tokens_enc;
};

} // namespace

template <>
ggml_tensor * llama_model_cohere_asr::graph<true>::build_inp_pos_emb_enc(int64_t n_tokens_enc) const {
    auto inp = std::make_unique<llm_graph_input_pos_emb_enc>(hparams.n_embd_enc, n_tokens_enc);

    auto & cur = inp->pos_emb;
    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_enc, 2 * n_tokens_enc - 1);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

template <>
llama_model_cohere_asr::graph<true>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_enc      = hparams.n_embd_enc;
    const int64_t n_head_enc      = hparams.n_head_enc;
    const int64_t d_head          = hparams.n_embd_head_k_enc;
    const int64_t n_conv_kernel   = hparams.n_conv_kernel_enc;
    const int64_t n_conv_channels = hparams.n_subsampling_channels_enc;
    const int64_t n_mel           = hparams.n_mel_bins_enc;
    const int64_t subsampling     = hparams.n_subsampling_factor_enc;

    ggml_tensor * cur = build_inp_embd_enc(); // [n_mel, T_mel]

    const int64_t t_mel = cur->ne[1];

    // conv subsampling: [n_mel, T, 1, 1] (ggml W=freq, H=time, C=1, N=1)
    cur = ggml_reshape_4d(ctx0, cur, n_mel, t_mel, 1, 1);

    cur = ggml_conv_2d_direct(ctx0, model.subsample_conv[0], cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, ggml_reshape_4d(ctx0, model.subsample_conv_b[0], 1, 1, n_conv_channels, 1));
    cur = ggml_relu(ctx0, cur);

    cur = ggml_conv_2d_dw_direct(ctx0, model.subsample_conv[1], cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, ggml_reshape_4d(ctx0, model.subsample_conv_b[1], 1, 1, n_conv_channels, 1));
    cur = ggml_conv_2d_direct(ctx0, model.subsample_conv[2], cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, ggml_reshape_4d(ctx0, model.subsample_conv_b[2], 1, 1, n_conv_channels, 1));
    cur = ggml_relu(ctx0, cur);

    cur = ggml_conv_2d_dw_direct(ctx0, model.subsample_conv[3], cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, ggml_reshape_4d(ctx0, model.subsample_conv_b[3], 1, 1, n_conv_channels, 1));
    cur = ggml_conv_2d_direct(ctx0, model.subsample_conv[4], cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, ggml_reshape_4d(ctx0, model.subsample_conv_b[4], 1, 1, n_conv_channels, 1));
    cur = ggml_relu(ctx0, cur);

    const int64_t t_enc = cur->ne[1];

    // flatten (freq, channel) -> feature axis, matching PyTorch's
    // x.transpose(1,2).reshape(b,t,-1) on a (B,C,T,F) tensor
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3)); // [F, C, T, 1]
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], t_enc);

    cur = build_lora_mm(model.subsample_out, cur);
    cur = ggml_add(ctx0, cur, model.subsample_out_b);
    cb(cur, "subsample_out", -1);

    ggml_tensor * pos_emb = build_inp_pos_emb_enc(t_enc); // [n_embd_enc, 2*t_enc-1]

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        // --- macaron half-step FFN 1 ---
        {
            ggml_tensor * ffn_inp = cur;
            cur = build_norm(cur, layer.ffn1_norm_enc, layer.ffn1_norm_enc_b, LLM_NORM, il);
            cur = build_ffn(cur,
                    layer.ffn1_up_enc, layer.ffn1_up_enc_b, nullptr,
                    nullptr, nullptr, nullptr,
                    layer.ffn1_down_enc, layer.ffn1_down_enc_b, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_SEQ, il);
            cur = ggml_add(ctx0, ffn_inp, ggml_scale(ctx0, cur, 0.5f));
        }

        // --- rel-pos self-attention ---
        {
            ggml_tensor * attn_inp = cur;
            cur = build_norm(cur, layer.attn_norm_enc, layer.attn_norm_enc_b, LLM_NORM, il);

            ggml_tensor * Qcur = ggml_add(ctx0, build_lora_mm(layer.wq_enc, cur), layer.bq_enc);
            ggml_tensor * Kcur = ggml_add(ctx0, build_lora_mm(layer.wk_enc, cur), layer.bk_enc);
            ggml_tensor * Vcur = ggml_add(ctx0, build_lora_mm(layer.wv_enc, cur), layer.bv_enc);

            Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head_enc, t_enc);
            Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head_enc, t_enc);
            Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head_enc, t_enc);

            ggml_tensor * q_with_u = ggml_add(ctx0, Qcur, layer.attn_pos_bias_u); // [d_head, n_head, T]
            q_with_u = ggml_permute(ctx0, q_with_u, 0, 2, 1, 3);                 // [d_head, T, n_head]
            ggml_tensor * q_with_v = ggml_add(ctx0, Qcur, layer.attn_pos_bias_v);
            q_with_v = ggml_permute(ctx0, q_with_v, 0, 2, 1, 3);

            ggml_tensor * k = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3)); // [d_head, T, n_head]
            ggml_tensor * v = ggml_cont(ctx0, ggml_permute(ctx0, Vcur, 1, 2, 0, 3)); // [T, n_head, d_head]

            ggml_tensor * matrix_ac = ggml_mul_mat(ctx0, k, q_with_u); // [T, T, n_head]

            ggml_tensor * p = build_lora_mm(layer.attn_pos, pos_emb);  // [n_embd_enc, 2T-1]
            p = ggml_reshape_3d(ctx0, p, d_head, n_head_enc, p->ne[1]);
            p = ggml_cont(ctx0, ggml_permute(ctx0, p, 0, 2, 1, 3));    // [d_head, 2T-1, n_head]

            ggml_tensor * matrix_bd = ggml_mul_mat(ctx0, p, q_with_v); // [2T-1, T, n_head]

            // rel_shift: (B,H,T,2T-1) -> (B,H,T,T), matching
            // RelPositionMultiHeadAttention.rel_shift() in modeling_cohere_asr.py
            {
                const int64_t pos_len = matrix_bd->ne[0];
                const int64_t q_len   = matrix_bd->ne[1];
                const int64_t h       = matrix_bd->ne[2];
                matrix_bd = ggml_pad(ctx0, matrix_bd, 1, 0, 0, 0);
                matrix_bd = ggml_reshape_3d(ctx0, matrix_bd, q_len, pos_len + 1, h);
                matrix_bd = ggml_view_3d(ctx0, matrix_bd, q_len, pos_len, h,
                        matrix_bd->nb[1], matrix_bd->nb[2], matrix_bd->nb[1]);
                matrix_bd = ggml_cont_3d(ctx0, matrix_bd, pos_len, q_len, h);
            }
            matrix_bd = ggml_view_3d(ctx0, matrix_bd, matrix_ac->ne[0], matrix_bd->ne[1], matrix_bd->ne[2],
                    matrix_bd->nb[1], matrix_bd->nb[2], 0);

            ggml_tensor * scores = ggml_add(ctx0, matrix_ac, matrix_bd);
            scores = ggml_scale(ctx0, scores, 1.0f / sqrtf((float) d_head));

            ggml_tensor * attn = ggml_soft_max(ctx0, scores);
            ggml_tensor * kqv  = ggml_mul_mat(ctx0, v, attn); // [d_head, T, n_head]
            kqv = ggml_permute(ctx0, kqv, 0, 2, 1, 3);        // [d_head, n_head, T]
            kqv = ggml_cont_2d(ctx0, kqv, d_head * n_head_enc, t_enc);

            cur = ggml_add(ctx0, build_lora_mm(layer.wo_enc, kqv), layer.bo_enc);
            cur = ggml_add(ctx0, attn_inp, cur);
        }

        // --- convolution module ---
        {
            ggml_tensor * conv_inp = cur;
            cur = build_norm(cur, layer.conv_ln_enc, layer.conv_ln_enc_b, LLM_NORM, il);

            cur = ggml_add(ctx0, build_lora_mm(layer.conv_pw1, cur), layer.conv_pw1_b); // [2*n_embd_enc, T]

            // GLU (sigmoid gate): ggml_glu doesn't expose the sigmoid variant, so
            // split + sigmoid + mul manually (same as tools/mtmd/models/conformer.cpp)
            {
                ggml_tensor * value = ggml_view_2d(ctx0, cur, n_embd_enc, t_enc, cur->nb[1], 0);
                ggml_tensor * gate  = ggml_view_2d(ctx0, cur, n_embd_enc, t_enc, cur->nb[1], n_embd_enc * cur->nb[0]);
                gate = ggml_sigmoid(ctx0, gate);
                cur  = ggml_mul(ctx0, ggml_cont(ctx0, value), gate);
            }

            // depthwise conv, symmetric SAME padding (non-causal, unlike the
            // streaming variant in tools/mtmd/models/conformer.cpp)
            cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // [T, n_embd_enc]
            const int64_t pad = (n_conv_kernel - 1) / 2;
            cur = ggml_pad_ext(ctx0, cur, (int) pad, (int) pad, 0, 0, 0, 0, 0, 0);
            cur = ggml_ssm_conv(ctx0, cur, layer.conv_dw);
            cur = ggml_add(ctx0, cur, layer.conv_dw_b);

            // folded BatchNorm1d affine (scale/shift precomputed at conversion time)
            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.conv_bn_w), layer.conv_bn_b);
            cur = ggml_silu(ctx0, cur);

            cur = ggml_add(ctx0, build_lora_mm(layer.conv_pw2, cur), layer.conv_pw2_b);
            cur = ggml_add(ctx0, conv_inp, cur);
        }

        // --- macaron half-step FFN 2 ---
        {
            ggml_tensor * ffn_inp = cur;
            cur = build_norm(cur, layer.ffn_norm_enc, layer.ffn_norm_enc_b, LLM_NORM, il);
            cur = build_ffn(cur,
                    layer.ffn_up_enc, layer.ffn_up_enc_b, nullptr,
                    nullptr, nullptr, nullptr,
                    layer.ffn_down_enc, layer.ffn_down_enc_b, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_SEQ, il);
            cur = ggml_add(ctx0, ffn_inp, ggml_scale(ctx0, cur, 0.5f));
        }

        cur = build_norm(cur, layer.norm_out_enc, layer.norm_out_enc_b, LLM_NORM, il);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
    }

    cb(cur, "result_embd", -1);

    // encoder_decoder_proj: n_embd_enc -> n_embd (applied once, not per-layer)
    cur = ggml_add(ctx0, build_lora_mm(model.enc_output_proj, cur), model.enc_output_proj_b);
    cb(cur, "result_embd_proj", -1);
    res->t_embd = cur;

    ggml_build_forward_expand(gf, cur);
}

template <>
llama_model_cohere_asr::graph<false>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * pos_emb = ggml_get_rows(ctx0, model.pos_embd, inp_pos);
    inpL = ggml_add(ctx0, inpL, pos_emb);
    inpL = build_norm(inpL, model.dec_embd_norm, model.dec_embd_norm_b, LLM_NORM, -1);

    ggml_tensor * embd_enc = build_inp_cross_embd();
    const int64_t n_outputs_enc = embd_enc->ne[1];

    auto * inp_attn_self  = build_attn_inp_kv();
    auto * inp_attn_cross = build_attn_inp_cross();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < (int) hparams.dec_n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // pre-LN self-attention (causal, KV-cached)
        cur = build_norm(inpL, model.layers[il].attn_norm, model.layers[il].attn_norm_b, LLM_NORM, il);
        {
            ggml_tensor * Qcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wq, cur), model.layers[il].bq);
            ggml_tensor * Kcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wk, cur), model.layers[il].bk);
            ggml_tensor * Vcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wv, cur), model.layers[il].bv);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k, n_head, n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head_v, n_head_kv, n_tokens);

            cur = build_attn(inp_attn_self,
                    model.layers[il].wo, model.layers[il].bo, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    1.0f / sqrtf((float) n_embd_head_k), il);
        }
        cur = ggml_add(ctx0, cur, inpSA);

        ggml_tensor * inpCA = cur;

        // pre-LN cross-attention (K/V from the fixed encoder output)
        cur = build_norm(cur, model.layers[il].attn_norm_cross, model.layers[il].attn_norm_cross_b, LLM_NORM, il);
        {
            ggml_tensor * Qcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wq_cross, cur), model.layers[il].bq_cross);
            ggml_tensor * Kcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wk_cross, embd_enc), model.layers[il].bk_cross);
            ggml_tensor * Vcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wv_cross, embd_enc), model.layers[il].bv_cross);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_kv, n_outputs_enc);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head_v, n_head_kv, n_outputs_enc);

            cur = build_attn(inp_attn_cross,
                    model.layers[il].wo_cross, model.layers[il].bo_cross, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    1.0f / sqrtf((float) n_embd_head_k), il);
        }
        if (il == (int) hparams.dec_n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpCA = ggml_get_rows(ctx0, inpCA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpCA);

        // pre-LN FFN (plain ReLU, ungated)
        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, model.layers[il].ffn_norm_b, LLM_NORM, il);
        cur = build_ffn(cur,
                model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   nullptr,
                nullptr, nullptr, nullptr,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, nullptr,
                nullptr, LLM_FFN_RELU, LLM_FFN_SEQ, il);
        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }
    cur = inpL;
    cb(cur, "result_embd", -1);

    cur = build_norm(cur, model.output_norm, model.output_norm_b, LLM_NORM, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    if (model.output_b) {
        cur = ggml_add(ctx0, cur, model.output_b);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
