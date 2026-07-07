from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("Qwen3TTSForConditionalGeneration", "Qwen3TTSModel")
class Qwen3TTSTalkerModel(TextModel):
    """Converter for the Qwen3-TTS Talker model.

    Produces a GGUF for the 28-layer Talker with interleaved MRoPE.
    The Code Predictor is converted by Qwen3TTSCodePredictorModel.

    Usage:
        python convert_hf_to_gguf.py <model_dir>
        python convert_hf_to_gguf.py <model_dir> --model-type Qwen3TTSCodePredictor
    """
    model_arch = gguf.MODEL_ARCH.QWEN3TTS

    def __init__(self, dir_model: Path, *args, **kwargs):
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            with open(dir_model / "config.json", "r", encoding="utf-8") as f:
                hparams = json.load(f)
        talker_cfg = hparams.get("talker_config", hparams)
        hparams["num_hidden_layers"] = talker_cfg.get("num_hidden_layers", 28)
        hparams["hidden_size"] = talker_cfg.get("hidden_size", 1024)
        hparams["num_attention_heads"] = talker_cfg.get("num_attention_heads", 16)
        hparams["num_key_value_heads"] = talker_cfg.get("num_key_value_heads", 8)
        hparams["vocab_size"] = talker_cfg.get("text_vocab_size", 151936)
        super().__init__(dir_model, *args, hparams=hparams, **kwargs)
        self.talker_hparams = talker_cfg
        self.block_count = talker_cfg.get("num_hidden_layers", 28)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        # TextModel.filter_tensors() drops any tensor whose name contains "talker."
        # as a generic guard against bundled TTS output heads inside otherwise
        # text-only multimodal chat models. Every tensor in a Qwen3-TTS checkpoint
        # is legitimately prefixed "talker." (it *is* the talker model), so that
        # guard would silently filter out 100% of tensors here. Bypass it and go
        # straight to ModelBase's generic filtering.
        return ModelBase.filter_tensors(item)

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        cfg = self.talker_hparams
        n_embd = cfg.get("hidden_size", 1024)
        n_head = cfg.get("num_attention_heads", 16)
        n_head_kv = cfg.get("num_key_value_heads", 8)
        n_ff = cfg.get("intermediate_size", 3072)
        n_layer = cfg.get("num_hidden_layers", 28)
        n_ctx = cfg.get("max_position_embeddings", 32768)
        vocab_size = cfg.get("vocab_size", 3072)
        rms_eps = cfg.get("rms_norm_eps", 1e-6)
        rope_theta = cfg.get("rope_theta", 1000000.0)
        text_vocab_size = cfg.get("text_vocab_size", 151936)
        text_embd_size = cfg.get("text_hidden_size", 2048)
        num_code_groups = cfg.get("num_code_groups", 16)
        position_id_per_s = cfg.get("position_id_per_seconds", 13)

        head_dim = cfg.get("head_dim", n_embd // n_head)

        self.gguf_writer.add_block_count(n_layer)
        self.gguf_writer.add_embedding_length(n_embd)
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head_kv)
        self.gguf_writer.add_key_length(head_dim)
        self.gguf_writer.add_value_length(head_dim)
        self.gguf_writer.add_feed_forward_length(n_ff)
        self.gguf_writer.add_context_length(n_ctx)
        self.gguf_writer.add_vocab_size(vocab_size)
        self.gguf_writer.add_layer_norm_rms_eps(rms_eps)
        self.gguf_writer.add_rope_freq_base(rope_theta)

        # MRoPE sections
        rope_scaling = cfg.get("rope_scaling", {})
        mrope_section = rope_scaling.get("mrope_section", [24, 20, 20])
        while len(mrope_section) < 4:
            mrope_section.append(0)
        self.gguf_writer.add_rope_dimension_sections(mrope_section[:4])

        # TTS-specific metadata
        self.gguf_writer.add_uint32(
            gguf.Keys.LLM.TTS_TEXT_VOCAB_SIZE.format(arch=self.gguf_writer.arch),
            text_vocab_size)
        self.gguf_writer.add_uint32(
            gguf.Keys.LLM.TTS_TEXT_EMBEDDING_LENGTH.format(arch=self.gguf_writer.arch),
            text_embd_size)
        self.gguf_writer.add_uint32(
            gguf.Keys.LLM.TTS_NUM_CODE_GROUPS.format(arch=self.gguf_writer.arch),
            num_code_groups)
        self.gguf_writer.add_uint32(
            gguf.Keys.LLM.TTS_POSITION_ID_PER_SECONDS.format(arch=self.gguf_writer.arch),
            position_id_per_s)

        # Speaker encoder parameters
        spk_cfg = self.hparams.get("speaker_encoder_config", {})
        spk_enc_dim = spk_cfg.get("enc_dim", 1024)
        spk_sample_rate = spk_cfg.get("sample_rate", 24000)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.speaker_encoder.embedding_length", spk_enc_dim)
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.speaker_encoder.sample_rate", spk_sample_rate)

        # Special codec token IDs
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.codec.pad_id", cfg.get("codec_pad_id", 2148))
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.codec.bos_id", cfg.get("codec_bos_id", 2149))
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.codec.eos_id", cfg.get("codec_eos_token_id", 2150))
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.codec.think_id", cfg.get("codec_think_id", 2154))
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.codec.nothink_id", cfg.get("codec_nothink_id", 2155))
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.codec.think_bos_id", cfg.get("codec_think_bos_id", 2156))
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.codec.think_eos_id", cfg.get("codec_think_eos_id", 2157))

        # TTS special tokens
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.tts_bos_token_id",
                                    self.hparams.get("tts_bos_token_id", 151672))
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.tts_eos_token_id",
                                    self.hparams.get("tts_eos_token_id", 151673))
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.tts_pad_token_id",
                                    self.hparams.get("tts_pad_token_id", 151671))

        # Language IDs for codec
        codec_lang = cfg.get("codec_language_id", {})
        if codec_lang:
            self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.codec.lang.english", codec_lang.get("english", 2050))

        # CustomVoice named speaker presets: spk_id maps a name to a reserved index in
        # the *existing* codec_embedding table (tts.codec_embd.weight) - not a separate
        # tensor. Base checkpoints have this empty/None; CustomVoice checkpoints populate
        # it with ~9 names. spk_is_dialect optionally overrides the codec language id for
        # a speaker regardless of --language (e.g. "eric" always speaks Sichuan dialect).
        spk_id = cfg.get("spk_id") or {}
        if spk_id:
            spk_is_dialect = cfg.get("spk_is_dialect") or {}
            names = sorted(spk_id.keys())
            ids = [spk_id[n] for n in names]
            dialects = [spk_is_dialect.get(n) or "" for n in names]
            self.gguf_writer.add_array(f"{self.gguf_writer.arch}.spk_id.names", names)
            self.gguf_writer.add_array(f"{self.gguf_writer.arch}.spk_id.ids", ids)
            self.gguf_writer.add_array(f"{self.gguf_writer.arch}.spk_id.dialects", dialects)

        # M-RoPE section array
        self.gguf_writer.add_array(f"{self.gguf_writer.arch}.rope.mrope_section",
                                   mrope_section[:3])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Skip code predictor tensors (they go in a separate GGUF)
        if "code_predictor" in name:
            return

        # Speaker encoder tensors: remap to spk_enc.* namespace
        if name.startswith("speaker_encoder."):
            spk_name = self._remap_speaker_encoder(name)
            if spk_name is not None:
                yield (spk_name, data_torch)
            return

        # Strip the "talker." prefix
        if name.startswith("talker."):
            name = name[len("talker."):]

        # Remap TTS-specific tensors that don't follow standard Qwen naming
        tts_remap = {
            "model.codec_embedding.weight": "tts.codec_embd.weight",
            "model.text_embedding.weight":  "tts.text_embd.weight",
            "codec_head.weight":            "tts.codec_head.weight",
            "text_projection.linear_fc1.weight": "tts.text_proj_up.weight",
            "text_projection.linear_fc1.bias":   "tts.text_proj_up.bias",
            "text_projection.linear_fc2.weight": "tts.text_proj_down.weight",
            "text_projection.linear_fc2.bias":   "tts.text_proj_down.bias",
        }
        if name in tts_remap:
            yield (tts_remap[name], data_torch)
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # tools/tts/qwen3tts-lib.cpp reads these tensors with CPU-side row-stride
        # arithmetic (read_row/read_bias) outside the ggml graph, which only
        # understands F32/F16/BF16 layouts. Q8_0's block layout breaks that
        # arithmetic (out-of-bounds reads -> segfault), so keep them
        # un-quantized regardless of --outtype.
        if self.ftype == gguf.LlamaFileType.MOSTLY_Q8_0 and new_name in (
            "tts.text_embd.weight",
            "tts.text_proj_up.weight",
            "tts.text_proj_down.weight",
            "tts.codec_embd.weight",
            "tts.codec_head.weight",
        ):
            return gguf.GGMLQuantizationType.F16
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    @staticmethod
    def _remap_speaker_encoder(name: str) -> str | None:
        direct_map = {
            "speaker_encoder.blocks.0.conv.weight": "spk_enc.conv0.weight",
            "speaker_encoder.blocks.0.conv.bias": "spk_enc.conv0.bias",
            "speaker_encoder.asp.conv.weight": "spk_enc.asp.conv.weight",
            "speaker_encoder.asp.conv.bias": "spk_enc.asp.conv.bias",
            "speaker_encoder.asp.tdnn.conv.weight": "spk_enc.asp.tdnn.weight",
            "speaker_encoder.asp.tdnn.conv.bias": "spk_enc.asp.tdnn.bias",
            "speaker_encoder.mfa.conv.weight": "spk_enc.mfa.weight",
            "speaker_encoder.mfa.conv.bias": "spk_enc.mfa.bias",
            "speaker_encoder.fc.weight": "spk_enc.fc.weight",
            "speaker_encoder.fc.bias": "spk_enc.fc.bias",
        }
        if name in direct_map:
            return direct_map[name]

        patterns = [
            (r"speaker_encoder\.blocks\.(\d+)\.res2net_block\.blocks\.(\d+)\.conv\.weight",
             "spk_enc.blk.{}.res2net.{}.weight"),
            (r"speaker_encoder\.blocks\.(\d+)\.res2net_block\.blocks\.(\d+)\.conv\.bias",
             "spk_enc.blk.{}.res2net.{}.bias"),
            (r"speaker_encoder\.blocks\.(\d+)\.se_block\.conv1\.weight",
             "spk_enc.blk.{}.se.conv1.weight"),
            (r"speaker_encoder\.blocks\.(\d+)\.se_block\.conv1\.bias",
             "spk_enc.blk.{}.se.conv1.bias"),
            (r"speaker_encoder\.blocks\.(\d+)\.se_block\.conv2\.weight",
             "spk_enc.blk.{}.se.conv2.weight"),
            (r"speaker_encoder\.blocks\.(\d+)\.se_block\.conv2\.bias",
             "spk_enc.blk.{}.se.conv2.bias"),
            (r"speaker_encoder\.blocks\.(\d+)\.tdnn1\.conv\.weight",
             "spk_enc.blk.{}.tdnn1.weight"),
            (r"speaker_encoder\.blocks\.(\d+)\.tdnn1\.conv\.bias",
             "spk_enc.blk.{}.tdnn1.bias"),
            (r"speaker_encoder\.blocks\.(\d+)\.tdnn2\.conv\.weight",
             "spk_enc.blk.{}.tdnn2.weight"),
            (r"speaker_encoder\.blocks\.(\d+)\.tdnn2\.conv\.bias",
             "spk_enc.blk.{}.tdnn2.bias"),
        ]
        for pattern, template in patterns:
            m = re.match(pattern, name)
            if m:
                groups = m.groups()
                return template.format(*groups)

        return None


@ModelBase.register("Qwen3TTSCodePredictor")
class Qwen3TTSCodePredictorModel(TextModel):
    """Converter for the Qwen3-TTS Code Predictor model.

    Produces a GGUF for the 5-layer Code Predictor with standard RoPE.
    Run from the same model directory as the Talker using the architecture
    override: --model-type Qwen3TTSCodePredictor (or use convert_qwen3tts.py).
    """
    model_arch = gguf.MODEL_ARCH.QWEN3TTS_CP

    def __init__(self, dir_model: Path, *args, **kwargs):
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            with open(dir_model / "config.json", "r", encoding="utf-8") as f:
                hparams = json.load(f)
        talker_cfg = hparams.get("talker_config", hparams)
        cp_cfg = talker_cfg.get("code_predictor_config", talker_cfg)
        hparams["num_hidden_layers"] = cp_cfg.get("num_hidden_layers", 5)
        hparams["hidden_size"] = cp_cfg.get("hidden_size", 1024)
        hparams["num_attention_heads"] = cp_cfg.get("num_attention_heads", 16)
        hparams["num_key_value_heads"] = cp_cfg.get("num_key_value_heads", 8)
        hparams["vocab_size"] = cp_cfg.get("vocab_size", 2048)
        super().__init__(dir_model, *args, hparams=hparams, **kwargs)
        self.cp_hparams = cp_cfg
        self.talker_hparams = talker_cfg
        self.block_count = cp_cfg.get("num_hidden_layers", 5)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        # See Qwen3TTSTalkerModel.filter_tensors: bypass TextModel's blanket
        # "talker." exclusion, which would otherwise drop every tensor here too
        # (they're all under "talker.code_predictor.").
        return ModelBase.filter_tensors(item)

    def set_vocab(self):
        self.gguf_writer.add_tokenizer_model("no_vocab")

    def set_gguf_parameters(self):
        cfg = self.cp_hparams
        n_embd = cfg.get("hidden_size", 1024)
        n_head = cfg.get("num_attention_heads", 16)
        n_head_kv = cfg.get("num_key_value_heads", 8)
        n_ff = cfg.get("intermediate_size", 3072)
        n_layer = cfg.get("num_hidden_layers", 5)
        vocab_size = cfg.get("vocab_size", 2048)
        rms_eps = cfg.get("rms_norm_eps", 1e-6)
        rope_theta = cfg.get("rope_theta", 1000000.0)
        num_code_groups = self.talker_hparams.get("num_code_groups", 16)

        head_dim = cfg.get("head_dim", n_embd // n_head)

        self.gguf_writer.add_block_count(n_layer)
        self.gguf_writer.add_embedding_length(n_embd)
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head_kv)
        self.gguf_writer.add_key_length(head_dim)
        self.gguf_writer.add_value_length(head_dim)
        self.gguf_writer.add_feed_forward_length(n_ff)
        self.gguf_writer.add_context_length(32)
        self.gguf_writer.add_vocab_size(vocab_size)
        self.gguf_writer.add_layer_norm_rms_eps(rms_eps)
        self.gguf_writer.add_rope_freq_base(rope_theta)

        self.gguf_writer.add_uint32(
            gguf.Keys.LLM.TTS_NUM_CODE_GROUPS.format(arch=self.gguf_writer.arch),
            num_code_groups)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Only process code predictor tensors
        cp_prefix = "talker.code_predictor."
        if not name.startswith(cp_prefix):
            return

        name = name[len(cp_prefix):]

        # lm_head.{idx}.weight -> tts.cp.lm_head.{idx}.weight
        if name.startswith("lm_head."):
            parts = name.split(".")
            cb_idx = int(parts[1])
            suffix = ".".join(parts[2:])
            new_name = f"tts.cp.lm_head.{cb_idx}.{suffix}"
            yield (new_name, data_torch)
            return

        # model.codec_embedding.{idx}.weight -> tts.cp.codec_embd.{idx}.weight
        if name.startswith("model.codec_embedding."):
            parts = name.split(".")
            cb_idx = int(parts[2])
            suffix = ".".join(parts[3:])
            new_name = f"tts.cp.codec_embd.{cb_idx}.{suffix}"
            yield (new_name, data_torch)
            return

        # Strip "model." prefix for standard layer tensors
        if name.startswith("model."):
            name = name[len("model."):]

        # small_to_mtp projection
        if name == "small_to_mtp_projection.weight":
            yield ("tts.cp.small_to_mtp.weight", data_torch)
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # Same constraint as Qwen3TTSTalkerModel.tensor_force_quant: these
        # per-codebook tensors (and, at 1.7B, small_to_mtp) are read via
        # CPU-side read_row()/read_bias() in tools/tts/qwen3tts-lib.cpp,
        # which requires F32/F16/BF16 layout.
        if self.ftype == gguf.LlamaFileType.MOSTLY_Q8_0 and (
            new_name.startswith("tts.cp.lm_head.")
            or new_name.startswith("tts.cp.codec_embd.")
            or new_name == "tts.cp.small_to_mtp.weight"
        ):
            return gguf.GGMLQuantizationType.F16
        return super().tensor_force_quant(name, new_name, bid, n_dims)
