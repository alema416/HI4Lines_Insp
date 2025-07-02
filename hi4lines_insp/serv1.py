#!/usr/bin/env python3

# first run: sudo ip route add 192.168.1.11/32 via 192.168.1.1 dev wlp2s0
import os
import sys
import json
from flask import Flask, request, jsonify
from datetime import datetime
import subprocess
import base64
import os
from hydra import initialize, compose
import base64
import requests
with initialize(config_path="../configs/"):
    cfg = compose(config_name="base")  # exp1.yaml with defaults key
#!/usr/bin/env python3
import os
import argparse
from collections import OrderedDict

import torch
import onnx
from onnxsim import simplify

# 1) Import your modified ResNet18 (with torch.flatten)
#from model.resnet18 import ResNet18
#from model.mobilenet import mobilenet
from torchvision.models import resnet18
from torchvision.models import mobilenet_v2
from torchvision.models import efficientnet_b0

import subprocess
import re

def patch_json(run_id, dev):
    json_file = f'../sample_params_{dev}.json'
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Update the model_name field
    data['model_name'] = f"model_{run_id}"
    
    # Write back the updated JSON
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Patched JSON with model_name: model_{run_id}")
    
def get_quantized_models_dir(run_id: int, dev: str) -> str:
    patch_json(run_id, dev)
    print('command run start')
    cmd = [
        "python3",
        '/app/orca_src/dg_compiler_api_usage.py',
        "--json_file", f'../sample_params_{dev}.json',
        "--model_file", f'../models/model_{run_id}.onnx',
        "--class_file", '../labels.yaml',
        '--calib_images_folder', '../data/processed/IDID_cropped_224/val/'
    ]
    print('command run ok')
    # run and capture both stdout and stderr as text
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False
    )
    print(proc)
    if dev == 'orca':
        return f'/app/models/model_{run_id}--224x224_quant_n2x_orca1_1/model_{run_id}--224x224_quant_n2x_orca1_1.n2x'
    elif dev == 'coral':
        return f'/app/models/model_{run_id}--224x224_quant_tflite_edgetpu_1/model_{run_id}--224x224_quant_tflite_edgetpu_1.tflite'
def load_checkpoint1(pth_path: str):
    # 1) instantiate the exact torchvision MobileNetV2
    model = mobilenet_v2(num_classes=2) #mobilenet(num_classes=2)
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

def main_onnx(run_id, pth_file, out, dev):
    print('main onnx enter')
    model = load_checkpoint1(pth_file)
    export_and_simplify(model, out, opset=13)
    print('pth --> onnx done')
    
    quant_dir = get_quantized_models_dir(run_id, dev)
    print("Quantized models directory:", quant_dir)
    return quant_dir

# fix so it sends folder and not model

# validation_service.py inside the Docker container

def validate(data):
    try:
        run_id = data.get('run_id')
        encoded_file = data.get('file')  # Get the Base64 encoded file
        if encoded_file:
            file_content = base64.b64decode(encoded_file)
            file_path = os.path.join('../models', 'ckpt.pth')
            with open(file_path, 'wb') as f:
                f.write(file_content)
            print(f"File {file_path} received and saved.")

        print(f'received run_id {run_id}')
        # pth --> tflite
        tflite_path = main_onnx(run_id, file_path, os.path.join('../models', f'model_{run_id}.onnx'), data.get('target_dev'))
        print(tflite_path)
        return {'tflite_path': tflite_path, 'run_id': run_id}
        #respon = send_file(, ) #send_file(os.path.join(cfg.training.save_path, str(run_id), 'tflite', 'model.tflite'), run_id)
    except Exception as e:
        #return jsonify({"error": str(e)}), 400
        with open('sasaa.txt', "w", encoding="utf-8") as f:
            f.write(str(e))
        return {'error': str(e)}

def main():
    pass
if __name__ == '__main__':
    main()
