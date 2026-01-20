import argparse
import json
import torch
from safetensors.torch import load_file, save_file
from comfy_kitchen import quantize_nvfp4, quantize_per_tensor_fp8, float_utils

QUANTIZABLE_WEIGHT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
ALLOWED_QTYPES = {"float8_e4m3fn", "nvfp4"}

def parse_args():
    p = argparse.ArgumentParser(
        prog="quantize.py",
        description="Quantize safetensors weights with rule-based policies.",
    )
    p.add_argument("json", help="Quant config JSON path")
    p.add_argument("src", help="Source safetensors path")
    p.add_argument("dst", help="Target safetensors path")
    return p.parse_args()

def quantize_layer(tensor, key, quantized_state_dict, quantization_layers, qtype):
    layer_name = key.replace(".weight", "")
    amax = torch.amax(tensor.abs()).to(torch.float32)
    
    if qtype == "nvfp4":
        weight_scale_2 = amax / (float_utils.F8_E4M3_MAX * float_utils.F4_E2M1_MAX)
        weight_quantized, weight_scale = quantize_nvfp4(tensor, weight_scale_2, epsilon=1e-6)
        quantized_state_dict[key] = weight_quantized.cpu()
        quantized_state_dict[f"{layer_name}.weight_scale"] = weight_scale.cpu()
        quantized_state_dict[f"{layer_name}.weight_scale_2"] = weight_scale_2.cpu()
    else: # fp8
        weight_scale = amax / float_utils.F8_E4M3_MAX
        quantized_state_dict[key] = quantize_per_tensor_fp8(tensor, weight_scale).cpu()
        quantized_state_dict[f"{layer_name}.weight_scale"] = weight_scale.cpu()

    quantization_layers[layer_name] = {"format": qtype}

    if qtype == "nvfp4":
        fmt = "nvfp4"
        tail = f"scale=max:{weight_scale.float().abs().max()}, mean:{weight_scale.float().mean():.4f} scale_2={weight_scale_2:.8f}"
    else:  # fp8
        fmt = "fp8"
        tail = f"scale={weight_scale:.8f}"
    print(f"[{layer_name.partition('.')[2]:<25} {fmt:>5}] amax:{amax.item():.4f}", tail)

def first_matching_qtype_for_key(key, rules):
    for r in rules:
        if any(p in key for p in r.get("match", [])):
            qtype = r.get("policy")
            return qtype if qtype in ALLOWED_QTYPES else None
    return None

def main():
    args = parse_args()
    with open(args.json, "r", encoding="utf-8") as f:
        config = json.load(f)
    block_name = config.get("block_name", "block")
    rules = config.get("rules", [])
    
    state_dict = load_file(args.src)
    quantized_state_dict, quantization_layers = {}, {}
    
    for key, tensor in state_dict.items():
        if not (block_name in key and key.endswith(".weight") and tensor.dtype in QUANTIZABLE_WEIGHT_DTYPES and tensor.ndim == 2):
            quantized_state_dict[key] = tensor
            continue
        
        qtype = first_matching_qtype_for_key(key, rules)
        if qtype is None:
            quantized_state_dict[key] = tensor
        else:
            quantize_layer(tensor.cuda(), key, quantized_state_dict, quantization_layers, qtype)

    metadata = {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": quantization_layers})}
    save_file(quantized_state_dict, args.dst, metadata=metadata)

if __name__ == "__main__":
    main()
