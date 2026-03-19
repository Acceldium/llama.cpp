#include "models.h"

void llama_model_qwen3tts::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, true);
    ml.get_key(LLM_KV_TTS_TEXT_VOCAB_SIZE,         hparams.tts_text_vocab_size);
    ml.get_key(LLM_KV_TTS_TEXT_EMBEDDING_LENGTH,   hparams.tts_text_embd_size);
    ml.get_key(LLM_KV_TTS_NUM_CODE_GROUPS,         hparams.tts_num_code_groups, false);
    ml.get_key(LLM_KV_TTS_POSITION_ID_PER_SECONDS, hparams.tts_position_id_per_s, false);
    switch (hparams.n_layer()) {
        case 28: type = LLM_TYPE_0_6B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen3tts::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_text_vocab = hparams.tts_text_vocab_size;
    const int64_t n_text_embd  = hparams.tts_text_embd_size;

    uint32_t n_codec_vocab_u32 = 0;
    ml.get_key(LLM_KV_VOCAB_SIZE, n_codec_vocab_u32, false);
    const int64_t n_codec_vocab = n_codec_vocab_u32 > 0 ? n_codec_vocab_u32 : 3072;

    // codec embedding used as tok_embd for the graph builder
    tok_embd = create_tensor(tn(LLM_TENSOR_TTS_CODEC_EMBD, "weight"), {n_embd, n_codec_vocab}, 0);

    // text embedding + projection
    tts_text_embd        = create_tensor(tn(LLM_TENSOR_TTS_TEXT_EMBD,      "weight"), {n_text_embd, n_text_vocab}, 0);
    tts_text_proj_up     = create_tensor(tn(LLM_TENSOR_TTS_TEXT_PROJ_UP,   "weight"), {n_text_embd, n_text_embd},  0);
    tts_text_proj_up_b   = create_tensor(tn(LLM_TENSOR_TTS_TEXT_PROJ_UP,   "bias"),   {n_text_embd},               TENSOR_NOT_REQUIRED);
    tts_text_proj_gate   = create_tensor(tn(LLM_TENSOR_TTS_TEXT_PROJ_GATE, "weight"), {n_text_embd, n_text_embd},  TENSOR_NOT_REQUIRED);
    tts_text_proj_down   = create_tensor(tn(LLM_TENSOR_TTS_TEXT_PROJ_DOWN, "weight"), {n_text_embd, n_embd},       0);
    tts_text_proj_down_b = create_tensor(tn(LLM_TENSOR_TTS_TEXT_PROJ_DOWN, "bias"),   {n_embd},                    TENSOR_NOT_REQUIRED);

    tts_codec_embd = tok_embd;

    output_norm    = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,   "weight"), {n_embd},             0);
    tts_codec_head = create_tensor(tn(LLM_TENSOR_TTS_CODEC_HEAD,"weight"), {n_embd, n_codec_vocab}, 0);
    output = tts_codec_head;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa},             0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa},             0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd},   0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
    }

    // Speaker encoder tensors are loaded separately by the CLI; count them so
    // done_getting_tensors() passes without error.
    for (const auto & kv : ml.weights_map) {
        if (kv.first.rfind("spk_enc.", 0) == 0) {
            ml.n_created++;
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen3tts::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen3tts::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // The Talker uses codec_embedding as tok_embd (stored there during load_tensors).
    // During prefill, the caller pre-computes the mixed embedding
    // (text_proj + codec + speaker) and passes it via ubatch.embd.
    // During decode, the caller passes the summed codec embeddings via ubatch.embd.
    // For simple token-only use, tok_embd (= codec_embd) is used directly.
    inpL = build_inp_embd(model.tok_embd);

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_multi(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_multi(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);

    // Output the hidden state as embedding. The codec_head logit projection
    // is applied externally because the output vocab (3072 codec tokens)
    // differs from the text tokenizer vocab (151936). Applying it here
    // would cause a size mismatch when llama_context copies logits.
    res->t_embd = cur;

    ggml_build_forward_expand(gf, cur);
}
