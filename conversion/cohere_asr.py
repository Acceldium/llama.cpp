from __future__ import annotations

import json
from typing import Iterable

import torch
from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger

# Raw HF Sequential index (encoder.pre_encode.conv.{idx}) -> our compact GGUF
# conv index (0=std, 1/3=depthwise, 2/4=pointwise). See ConvSubsampling in
# modeling_cohere_asr.py: indices 1/4/7 are ReLU (no weights).
SUBSAMPLE_CONV_IDX_MAP = {0: 0, 2: 1, 3: 2, 5: 3, 6: 4}

BATCH_NORM_EPS = 1e-5


@ModelBase.register("CohereAsrForConditionalGeneration")
class CohereAsrModel(TextModel):
    model_arch = gguf.MODEL_ARCH.COHERE_ASR

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bn_buffer: dict[int, dict[str, Tensor]] = {}

    # cohere-asr's config.json nests almost everything under "encoder",
    # "transf_decoder"/"config_dict" and "head" instead of flat top-level
    # keys, so bridge the handful of lookups TextModel.__init__ itself makes
    # (block_count, context length, etc) before our own set_gguf_parameters
    # takes over and reads the nested dicts directly.
    def find_hparam(self, keys: Iterable[str], optional: bool = False):
        keys = list(keys)
        for key in keys:
            if key in self.hparams:
                return self.hparams[key]

        dec_cfg = self.hparams["transf_decoder"]["config_dict"]
        bridged = {
            "n_layers": self.hparams["encoder"]["n_layers"],
            "num_hidden_layers": self.hparams["encoder"]["n_layers"],
            "hidden_size": dec_cfg["hidden_size"],
            "max_position_embeddings": dec_cfg["max_sequence_length"],
            "max_sequence_length": dec_cfg["max_sequence_length"],
        }
        for key in keys:
            if key in bridged:
                return bridged[key]

        if optional:
            return None
        raise KeyError(f"could not find any of: {keys}")

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        enc = self.hparams["encoder"]
        dec = self.hparams["transf_decoder"]["config_dict"]
        head = self.hparams["head"]
        pre = self.hparams["preprocessor"]

        n_embd_dec = dec["hidden_size"]
        n_head_dec = dec["num_attention_heads"]

        self.gguf_writer.add_context_length(dec["max_sequence_length"])
        self.gguf_writer.add_embedding_length(n_embd_dec)
        self.gguf_writer.add_feed_forward_length(dec["inner_size"])
        self.gguf_writer.add_block_count(enc["n_layers"])
        self.gguf_writer.add_decoder_block_count(dec["num_layers"])
        self.gguf_writer.add_head_count(n_head_dec)
        self.gguf_writer.add_head_count_kv(n_head_dec)
        self.gguf_writer.add_key_length(n_embd_dec // n_head_dec)
        self.gguf_writer.add_value_length(n_embd_dec // n_head_dec)
        self.gguf_writer.add_layer_norm_eps(1e-5)
        self.gguf_writer.add_vocab_size(head["num_classes"])
        self.gguf_writer.add_file_type(self.ftype)

        decoder_start_token_id = self.hparams.get("decoder_start_token_id")
        if decoder_start_token_id is None:
            gen_config_path = self.dir_model / "generation_config.json"
            if gen_config_path.is_file():
                with open(gen_config_path, "r", encoding="utf-8") as f:
                    decoder_start_token_id = json.load(f).get("decoder_start_token_id")
        if decoder_start_token_id is not None:
            self.gguf_writer.add_decoder_start_token_id(decoder_start_token_id)

        arch = self.gguf_writer.arch
        n_head_enc = enc["n_heads"]
        d_model_enc = enc["d_model"]
        self.gguf_writer.add_uint32(gguf.Keys.Encoder.EMBEDDING_LENGTH.format(arch=arch), d_model_enc)
        self.gguf_writer.add_uint32(gguf.Keys.Encoder.FEED_FORWARD_LENGTH.format(arch=arch), d_model_enc * enc["ff_expansion_factor"])
        self.gguf_writer.add_uint32(gguf.Keys.Encoder.HEAD_COUNT.format(arch=arch), n_head_enc)
        self.gguf_writer.add_uint32(gguf.Keys.Encoder.KEY_LENGTH.format(arch=arch), d_model_enc // n_head_enc)
        self.gguf_writer.add_uint32(gguf.Keys.Encoder.VALUE_LENGTH.format(arch=arch), d_model_enc // n_head_enc)
        self.gguf_writer.add_uint32(gguf.Keys.Encoder.CONV_KERNEL_SIZE.format(arch=arch), enc["conv_kernel_size"])
        self.gguf_writer.add_uint32(gguf.Keys.Encoder.SUBSAMPLING_FACTOR.format(arch=arch), enc["subsampling_factor"])
        self.gguf_writer.add_uint32(gguf.Keys.Encoder.SUBSAMPLING_CHANNELS.format(arch=arch), enc["subsampling_conv_channels"])
        self.gguf_writer.add_uint32(gguf.Keys.Encoder.FEATURES_COUNT.format(arch=arch), pre["features"])

    def _flush_batch_norm(self, bid: int) -> Iterable[tuple[str, Tensor]]:
        buf = self._bn_buffer.get(bid)
        if buf is None or len(buf) < 4:
            return []

        gamma = buf["weight"].float()
        beta = buf["bias"].float()
        mean = buf["running_mean"].float()
        var = buf["running_var"].float()

        scale = gamma / torch.sqrt(var + BATCH_NORM_EPS)
        shift = beta - mean * scale

        del self._bn_buffer[bid]

        return [
            (self.format_tensor_name(gguf.MODEL_TENSOR.ENC_CONV_BN, bid, ".weight"), scale),
            (self.format_tensor_name(gguf.MODEL_TENSOR.ENC_CONV_BN, bid, ".bias"), shift),
        ]

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # re-derived below; the generic (first-decimal-part) extraction
        # doesn't fit the subsample conv's non-contiguous raw indices {0,2,3,5,6}

        # -- mel frontend (fixed buffers, not learned) --
        if name == "preprocessor.featurizer.fb":
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.ENC_MEL_FB), data_torch.squeeze(0))]
        if name == "preprocessor.featurizer.window":
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.ENC_MEL_WINDOW), data_torch)]

        # -- subsampling stem --
        if name.startswith("encoder.pre_encode.conv."):
            parts = name.split(".")
            raw_idx = int(parts[3])
            suffix = "." + parts[4]
            new_idx = SUBSAMPLE_CONV_IDX_MAP[raw_idx]
            # all subsample stem convs are real 2D convs (std/depthwise/pointwise
            # all keep their native 4D (OC,IC,KH,KW) shape -- only the per-layer
            # Conformer conv module below has 1D convs that need squeezing)
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.ENC_SUBSAMPLE_CONV, new_idx, suffix), data_torch)]
        if name.startswith("encoder.pre_encode.out."):
            suffix = "." + name.rsplit(".", 1)[1]
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.ENC_SUBSAMPLE_OUT, None, suffix), data_torch)]

        if name.startswith("encoder_decoder_proj."):
            suffix = "." + name.rsplit(".", 1)[1]
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.ENC_OUTPUT_PROJ, None, suffix), data_torch)]

        # -- Conformer encoder layers --
        if name.startswith("encoder.layers."):
            parts = name.split(".")
            bid = int(parts[2])
            rest = ".".join(parts[3:])
            suffix = "." + parts[-1]

            mapping = {
                "norm_feed_forward1": gguf.MODEL_TENSOR.ENC_FFN1_NORM,
                "feed_forward1.linear1": gguf.MODEL_TENSOR.ENC_FFN1_UP,
                "feed_forward1.linear2": gguf.MODEL_TENSOR.ENC_FFN1_DOWN,
                "norm_self_att": gguf.MODEL_TENSOR.ENC_ATTN_NORM,
                "self_attn.linear_q": gguf.MODEL_TENSOR.ENC_ATTN_Q,
                "self_attn.linear_k": gguf.MODEL_TENSOR.ENC_ATTN_K,
                "self_attn.linear_v": gguf.MODEL_TENSOR.ENC_ATTN_V,
                "self_attn.linear_out": gguf.MODEL_TENSOR.ENC_ATTN_OUT,
                "self_attn.linear_pos": gguf.MODEL_TENSOR.ENC_ATTN_POS,
                "norm_conv": gguf.MODEL_TENSOR.ENC_CONV_LN,
                "norm_feed_forward2": gguf.MODEL_TENSOR.ENC_FFN_NORM,
                "feed_forward2.linear1": gguf.MODEL_TENSOR.ENC_FFN_UP,
                "feed_forward2.linear2": gguf.MODEL_TENSOR.ENC_FFN_DOWN,
                "norm_out": gguf.MODEL_TENSOR.ENC_NORM_OUT,
            }

            for prefix, tensor_kind in mapping.items():
                if rest == f"{prefix}.weight" or rest == f"{prefix}.bias":
                    return [(self.format_tensor_name(tensor_kind, bid, suffix), data_torch)]

            if rest == "self_attn.pos_bias_u":
                return [(self.format_tensor_name(gguf.MODEL_TENSOR.ENC_ATTN_POS_BIAS_U, bid, ".weight"), data_torch)]
            if rest == "self_attn.pos_bias_v":
                return [(self.format_tensor_name(gguf.MODEL_TENSOR.ENC_ATTN_POS_BIAS_V, bid, ".weight"), data_torch)]

            if rest == "conv.depthwise_conv.weight":
                data_torch = data_torch.squeeze(1)  # (C,1,K) -> (C,K)
                return [(self.format_tensor_name(gguf.MODEL_TENSOR.ENC_CONV_DW, bid, ".weight"), data_torch)]
            if rest == "conv.depthwise_conv.bias":
                return [(self.format_tensor_name(gguf.MODEL_TENSOR.ENC_CONV_DW, bid, ".bias"), data_torch)]
            if rest.startswith("conv.pointwise_conv1.") or rest.startswith("conv.pointwise_conv2."):
                tensor_kind = gguf.MODEL_TENSOR.ENC_CONV_PW1 if "conv1" in rest else gguf.MODEL_TENSOR.ENC_CONV_PW2
                if rest.endswith(".weight"):
                    data_torch = data_torch.squeeze(-1)  # (OC,IC,1) -> (OC,IC)
                return [(self.format_tensor_name(tensor_kind, bid, suffix), data_torch)]

            if rest.startswith("conv.batch_norm."):
                bn_key = rest.rsplit(".", 1)[1]
                if bn_key == "num_batches_tracked":
                    return []
                self._bn_buffer.setdefault(bid, {})[bn_key] = data_torch
                return self._flush_batch_norm(bid)

            raise ValueError(f"cohere-asr: unhandled encoder tensor {name!r}")

        # -- Transformer decoder --
        if name.startswith("transf_decoder._embedding."):
            if name == "transf_decoder._embedding.token_embedding.weight":
                return [(self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD), data_torch)]
            if name == "transf_decoder._embedding.position_embedding.pos_enc":
                return [(self.format_tensor_name(gguf.MODEL_TENSOR.POS_EMBD), data_torch)]
            if name.startswith("transf_decoder._embedding.layer_norm."):
                suffix = "." + name.rsplit(".", 1)[1]
                return [(self.format_tensor_name(gguf.MODEL_TENSOR.DEC_EMBD_NORM, None, suffix), data_torch)]

        if name == "transf_decoder._decoder.final_layer_norm.weight" or name == "transf_decoder._decoder.final_layer_norm.bias":
            suffix = "." + name.rsplit(".", 1)[1]
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT_NORM, None, suffix), data_torch)]

        if name.startswith("transf_decoder._decoder.layers."):
            parts = name.split(".")
            bid = int(parts[3])
            rest = ".".join(parts[4:])
            suffix = "." + parts[-1]

            mapping = {
                "layer_norm_1": gguf.MODEL_TENSOR.DEC_ATTN_NORM,
                "first_sub_layer.query_net": gguf.MODEL_TENSOR.DEC_ATTN_Q,
                "first_sub_layer.key_net": gguf.MODEL_TENSOR.DEC_ATTN_K,
                "first_sub_layer.value_net": gguf.MODEL_TENSOR.DEC_ATTN_V,
                "first_sub_layer.out_projection": gguf.MODEL_TENSOR.DEC_ATTN_OUT,
                "layer_norm_2": gguf.MODEL_TENSOR.DEC_CROSS_ATTN_NORM,
                "second_sub_layer.query_net": gguf.MODEL_TENSOR.DEC_CROSS_ATTN_Q,
                "second_sub_layer.key_net": gguf.MODEL_TENSOR.DEC_CROSS_ATTN_K,
                "second_sub_layer.value_net": gguf.MODEL_TENSOR.DEC_CROSS_ATTN_V,
                "second_sub_layer.out_projection": gguf.MODEL_TENSOR.DEC_CROSS_ATTN_OUT,
                "layer_norm_3": gguf.MODEL_TENSOR.DEC_FFN_NORM,
                "third_sub_layer.dense_in": gguf.MODEL_TENSOR.DEC_FFN_UP,
                "third_sub_layer.dense_out": gguf.MODEL_TENSOR.DEC_FFN_DOWN,
            }
            for prefix, tensor_kind in mapping.items():
                if rest == f"{prefix}.weight" or rest == f"{prefix}.bias":
                    return [(self.format_tensor_name(tensor_kind, bid, suffix), data_torch)]

            raise ValueError(f"cohere-asr: unhandled decoder tensor {name!r}")

        if name == "log_softmax.mlp.layer0.weight" or name == "log_softmax.mlp.layer0.bias":
            suffix = "." + name.rsplit(".", 1)[1]
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT, None, suffix), data_torch)]

        raise ValueError(f"cohere-asr: unhandled tensor {name!r}")
