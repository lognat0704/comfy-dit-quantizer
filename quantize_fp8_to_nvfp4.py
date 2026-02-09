"""
FP8 to NVFP4 Quantization Script

This script extends the original quantize.py to accept FP8 source weights,
which it upcasts to float32 before quantizing to NVFP4.

Usage:
    python quantize_fp8_to_nvfp4.py configs/qwen-image-edit-2511.json \
        /path/to/fp8_model.safetensors \
        /path/to/output_nvfp4.safetensors
"""
import argparse
import json
import os
from typing import Any

import torch
import comfy_kitchen as ck
from comfy_kitchen.float_utils import F8_E4M3_MAX, F4_E2M1_MAX
from safetensors.torch import load_file, save_file
import utils
from utils import print_layer_metrics, print_layer_header

# Extended to include FP8 dtypes for source
QUANTIZABLE_WEIGHT_DTYPES = (
    torch.bfloat16, torch.float16, torch.float32,
    torch.float8_e4m3fn, torch.float8_e5m2,
)
ALLOWED_QTYPES = {"float8_e4m3fn", "nvfp4"}

device = utils.get_device()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        prog="quantize_fp8_to_nvfp4.py",
        description="Quantize FP8/BF16/FP32 safetensors weights to NVFP4.",
    )
    p.add_argument("json", help="Quant config JSON path")
    p.add_argument("src", nargs="*", help="Source safetensors path(s)")
    p.add_argument("dst", help="Target safetensors path")
    p.add_argument(
        "-d", "--downcast-fp32", choices=("fp16", "bf16"), default=None,
        metavar="{fp16,bf16}",
        help="Cast fp32 tensors to the selected dtype (default: keep FP32)."
    )
    p.add_argument(
        "-m", "--method", choices=("amax", "mse"), default="mse",
        metavar="{amax, mse}",
        help="Set calibration method (default: mse)."
    )
    p.add_argument("-n", "--n-samples", default=None, type=int,
                   help="num of samples for calibration method")
    p.add_argument("-q", "--quiet", action="store_true", help="no verbose.")
    p.add_argument("-t", "--test", action="store_true",
                   help="does not save output")
    return p.parse_args()


def quantize_weight(
    weight: torch.Tensor,
    key: str,
    quantized_state_dict: dict[str, torch.Tensor],
    quantization_layers: dict[str, Any],
    qtype: str,
    qformat: str,
    method: str,
    n_samples: int | None,
    verbose: bool = True,
) -> None:
    """Quantize a single weight tensor to the specified format."""
    layer_name = key.replace(".weight", "")

    # Upcast FP8 to float32 for calibration
    if weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        weight = weight.to(torch.float32)

    if qtype == "nvfp4":
        if method == "mse":
            weight_scale_2 = utils.scale_mse_nvfp4(weight, n_samples=n_samples)
        else:
            weight_scale_2 = utils.scale_amax_nvfp4(weight)
        with ck.use_backend("triton"):  # triton supports conversion from fp32
            weight_quantized, weight_scale = ck.quantize_nvfp4(
                weight, weight_scale_2
            )
        if verbose:
            print_layer_metrics(
                layer_name, weight, weight_quantized,
                weight_scale_2, weight_scale
            )
        quantized_state_dict[key] = weight_quantized.cpu()
        quantized_state_dict[f"{layer_name}.weight_scale"] = weight_scale.cpu()
        quantized_state_dict[f"{layer_name}.weight_scale_2"] = (
            weight_scale_2.cpu()
        )
    else:  # fp8
        if method == "mse":
            weight_scale = utils.scale_mse_fp8(weight, n_samples=n_samples)
        else:
            weight_scale = utils.scale_amax_fp8(weight)
        with ck.use_backend("triton"):
            weight_quantized = ck.quantize_per_tensor_fp8(weight, weight_scale)
        if verbose:
            print_layer_metrics(layer_name, weight, weight_quantized, weight_scale)
        quantized_state_dict[key] = weight_quantized.cpu()
        quantized_state_dict[f"{layer_name}.weight_scale"] = weight_scale.cpu()

    if qformat == "comfy_quant":
        quantized_state_dict[f"{layer_name}.comfy_quant"] = torch.tensor(
            list(json.dumps({"format": qtype}).encode("utf-8")), dtype=torch.uint8
        )
    else:  # 1.0
        quantization_layers[layer_name] = {"format": qtype}


def store_with_optional_downcast(
    tensor: torch.Tensor,
    key: str,
    quantized_state_dict: dict[str, torch.Tensor],
    cast_to: torch.dtype | None,
    verbose: bool = True,
) -> None:
    """Store tensor with optional dtype downcasting."""
    if tensor.dtype == torch.float32 and cast_to is not None:
        casted_weight = tensor.to(dtype=cast_to)
        quantized_state_dict[key] = casted_weight.cpu()

        if verbose and ".weight" in key:
            layer_name = key.replace(".weight", "")
            print_layer_metrics(layer_name, tensor, casted_weight)
    else:
        quantized_state_dict[key] = tensor.cpu()


def first_matching_qtype_for_key(key: str, rules: list[dict]) -> str | None:
    """Find the first matching quantization type for a key."""
    for r in rules:
        if any(p in key for p in r.get("match", [])):
            qtype = r.get("policy")
            return qtype if qtype in ALLOWED_QTYPES else None
    return None


def main() -> None:
    """Main entry point."""
    args = parse_args()
    cast_to = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
        args.downcast_fp32, None
    )

    assert ".json" in args.json, f"{args.json} is not .json file."
    with open(args.json, "r", encoding="utf-8") as f:
        config = json.load(f)

    qformat = config.get("format", "1.0")
    block_names = config.get("block_names", ["block", "transformer", "layer"])
    rules = config.get("rules", [])

    quantized_state_dict: dict[str, torch.Tensor] = {}
    quantization_layers: dict[str, Any] = {}

    for src_file in args.src:
        print(f"Processing: {src_file}")
        state_dict = load_file(src_file)

        if not args.quiet:
            print_layer_header()

        for key, tensor in state_dict.items():
            # Check if this is a quantizable weight
            is_target_layer = (
                any(b in key for b in block_names)
                and key.endswith(".weight")
                and tensor.dtype in QUANTIZABLE_WEIGHT_DTYPES
                and tensor.ndim == 2
            )

            if not is_target_layer:
                store_with_optional_downcast(
                    tensor, key, quantized_state_dict, cast_to,
                    verbose=not args.quiet
                )
                continue

            qtype = first_matching_qtype_for_key(key, rules)
            if qtype is None:
                store_with_optional_downcast(
                    tensor, key, quantized_state_dict, cast_to,
                    verbose=not args.quiet
                )
            else:
                quantize_weight(
                    tensor.to(device), key, quantized_state_dict,
                    quantization_layers, qtype, qformat, args.method,
                    args.n_samples, verbose=not args.quiet
                )

    metadata = (
        {"_quantization_metadata": json.dumps({
            "format_version": "1.0",
            "layers": quantization_layers
        })}
        if qformat != "comfy_quant" else None
    )

    if not args.test:
        save_file(quantized_state_dict, args.dst, metadata=metadata)
        total_bytes = os.path.getsize(args.dst)
        print(f"Output: {args.dst} ({round(total_bytes / (1024**3), 2)}GB)")


if __name__ == "__main__":
    main()
