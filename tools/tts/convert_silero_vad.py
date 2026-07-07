#!/usr/bin/env python3
"""
Convert the vendored Silero VAD ONNX model (16kHz path only) to a small GGUF
file, so the C++ CLI can run real Silero VAD natively via ggml - no torch,
no onnxruntime, same reasoning as silero_vad_lite.py but for the CLI instead
of the Streamlit app. See examples/meeting-voice-viz/HOW_IT_WORKS.md.

The ONNX graph branches on sample rate at the top (an `If` node); this script
only extracts the sr==16000 branch's weights, since that's the only rate the
CLI's --live-mic uses.

Usage:
    python tools/tts/convert_silero_vad.py \
        --input examples/meeting-voice-viz/models/silero_vad.onnx \
        --output models/silero-vad.gguf

Requires: pip install onnx (conversion-time only; not needed to run the CLI)
"""

import argparse
import sys

import numpy as np
import onnx
from onnx import numpy_helper

import gguf

ARCH = "silero-vad"

# Maps the ONNX tensor's short name (suffix after the branch/inline prefix)
# to the GGUF tensor name this repo's C++ code expects.
TENSOR_MAP = {
    "stft.forward_basis_buffer": "silero_vad.stft.basis",
    "encoder.0.reparam_conv.weight": "silero_vad.encoder.0.weight",
    "encoder.0.reparam_conv.bias": "silero_vad.encoder.0.bias",
    "encoder.1.reparam_conv.weight": "silero_vad.encoder.1.weight",
    "encoder.1.reparam_conv.bias": "silero_vad.encoder.1.bias",
    "encoder.2.reparam_conv.weight": "silero_vad.encoder.2.weight",
    "encoder.2.reparam_conv.bias": "silero_vad.encoder.2.bias",
    "encoder.3.reparam_conv.weight": "silero_vad.encoder.3.weight",
    "encoder.3.reparam_conv.bias": "silero_vad.encoder.3.bias",
    "decoder.rnn.weight_ih": "silero_vad.rnn.weight_ih",
    "decoder.rnn.weight_hh": "silero_vad.rnn.weight_hh",
    "decoder.rnn.bias_ih": "silero_vad.rnn.bias_ih",
    "decoder.rnn.bias_hh": "silero_vad.rnn.bias_hh",
    "decoder.decoder.2.weight": "silero_vad.decoder.weight",
    "decoder.decoder.2.bias": "silero_vad.decoder.bias",
}


def extract_16k_branch_weights(onnx_path):
    model = onnx.load(onnx_path)
    if_node = model.graph.node[2]
    if if_node.op_type != "If":
        raise RuntimeError("unexpected graph structure - expected node[2] to be the sr==16000/8000 If branch")

    const0 = numpy_helper.to_array(model.graph.node[0].attribute[0].t)
    then_branch = else_branch = None
    for attr in if_node.attribute:
        if attr.name == "then_branch":
            then_branch = attr.g
        elif attr.name == "else_branch":
            else_branch = attr.g
    branch = then_branch if int(const0) == 16000 else else_branch

    found = {}
    for node in branch.node:
        if node.op_type != "Constant":
            continue
        suffix = node.output[0].split("__")[-1]
        if suffix in TENSOR_MAP:
            found[suffix] = numpy_helper.to_array(node.attribute[0].t)

    missing = set(TENSOR_MAP) - set(found)
    if missing:
        raise RuntimeError(f"missing expected tensors in ONNX graph: {missing}")
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Path to silero_vad.onnx")
    ap.add_argument("--output", required=True, help="Output GGUF path")
    args = ap.parse_args()

    weights = extract_16k_branch_weights(args.input)

    writer = gguf.GGUFWriter(path=None, arch=ARCH)
    writer.add_name("Silero-VAD-16k")
    writer.add_type(gguf.GGUFType.MODEL)
    writer.add_file_type(gguf.LlamaFileType.ALL_F32)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_uint32(f"{ARCH}.sample_rate", 16000)
    writer.add_uint32(f"{ARCH}.chunk_samples", 512)
    writer.add_uint32(f"{ARCH}.context_samples", 64)
    writer.add_uint32(f"{ARCH}.hidden_size", 128)

    for onnx_name, gguf_name in TENSOR_MAP.items():
        arr = np.ascontiguousarray(weights[onnx_name].astype(np.float32))
        writer.add_tensor(gguf_name, arr)
        print(f"  {gguf_name}: shape={list(arr.shape)}")

    writer.write_header_to_file(path=args.output)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    sys.exit(main())
