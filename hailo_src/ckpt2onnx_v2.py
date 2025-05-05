#!/usr/bin/env python3
import os
import argparse
from collections import OrderedDict

import torch
import onnx
from onnxsim import simplify

# 1) Import your modified ResNet18 (with torch.flatten)
from model.resnet18 import ResNet18
from model.mobilenet import mobilenet

from torchvision.models import mobilenet_v2

def load_checkpoint1(pth_path: str):
    # 1) instantiate the exact torchvision MobileNetV2
    model = mobilenet_v2(weights=None, num_classes=2)
    model.eval()

    # 2) load your checkpoint (it may be a dict with 'state_dict' inside)
    ckpt = torch.load(pth_path, map_location="cpu")
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    # 3) strip off any "module." prefixes and unwanted keys
    clean = OrderedDict()
    for k, v in sd.items():
        # remove DataParallel prefix
        nk = k.replace("module.", "")
        # drop any extra keys your training pipeline may have added
        if nk == "n_averaged":
            continue
        clean[nk] = v

    # 4) load into the model
    model.load_state_dict(clean)
    return model

def load_checkpoint(pth_path: str) -> torch.nn.Module:
    model = mobilenet(num_classes=2)
    sd = torch.load(pth_path, map_location="cpu")
    clean = OrderedDict()
    for k, v in sd.items():
        nk = k.replace("module.", "")
        if nk == "n_averaged":
            continue
        clean[nk] = v
    model.load_state_dict(clean)
    model.eval()
    return model

def export_and_simplify(model: torch.nn.Module, onnx_path: str, opset: int):
    # Dummy input to drive the trace
    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    # Direct ONNX export (uses tracing internally)
    torch.onnx.export(
        model, dummy, onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,           # fuse constants
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input":  {0: "batch_size"},
                      "output": {0: "batch_size"}},
        training=torch.onnx.TrainingMode.EVAL,
    )
    print(f"✅ ONNX exported to: {onnx_path}")

    # Simplify to clean up any leftover shape ops
    print("🔄 Running onnx-simplifier…")
    model_proto = onnx.load(onnx_path)
    simp, check = simplify(model_proto, skip_fuse_bn=False)
    if not check:
        raise RuntimeError("❌ onnx-simplifier failed to validate the model")
    onnx.save(simp, onnx_path)
    print(f"✅ Simplified ONNX saved → {onnx_path}")

def main_onnx(exp_name, run_id, out):
    # Make sure you have:
    #   pip install "numpy<2.0" torch onnx onnx-simplifier
    
    model = load_checkpoint1(os.path.join(exp_name, run_id, 'model_state_dict', 'model.pth'))
    export_and_simplify(model, out, opset=13)

    print("\nDone! Now run your STM32Cube.AI MCU flow:")
    print("  python3 stm32ai_main.py \\")
    print("    --config-path config_file_examples \\")
    print("    --config-name chain_mc_config.yaml")
