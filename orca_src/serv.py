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
from model.resnet18 import ResNet18
from model.mobilenet import mobilenet

from torchvision.models import mobilenet_v2
from torchvision.models import efficientnet_b0

import subprocess
import re

def get_quantized_models_dir(run_id: int) -> str:
    cmd = [
        "python3",
        '/app/yolov5-8_cloud_api/dg_compiler_api_usage.py',
        "--json_file", '../../output_files/sample_params.json',
        "--model_file", f'../../output_files/models/model_{run_id}.onnx',
        "--class_file", '../../output_files/labels.yaml',
        '--calib_images_folder', '../../output_files/data/processed/IDID_cropped_224/val/broken/'
    ]
    # run and capture both stdout and stderr as text
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False
    )
    print(proc)
    return f'/output_files/model_{run_id}--224x224_quant_n2x_orca1_1'

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

def main_onnx(run_id, pth_file, out):
    print('main onnx enter')
    model = load_checkpoint1(pth_file)
    export_and_simplify(model, out, opset=13)
    print('pth --> onnx done')
    
    quant_dir = get_quantized_models_dir(run_id)
    print("Quantized models directory:", quant_dir)
    return quant_dir

# fix so it sends folder and not model
def send_file(filename, run_id, url=f"http://{cfg.training.orca_dev_ip}:{cfg.training.orca_port}/validate"):
    # Read and encode the file
    print('send file enter')

    with open(filename, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "file": encoded,
        "run_id": run_id
    }
    headers = {"Content-Type": "application/json"}
    
    # Send the request
    print('sending file to rpi...')   
    
    response = requests.post(url, json=payload, headers=headers)
    
    # Print out status and response
    print(f"Status: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except ValueError:
        print(response.text)
    if response.status_code == 200:
        augrc_hw_train = response.json()['augrc_hw_train']
        acc_hw_train = response.json()['acc_hw_train']
        
        augrc_hw_val = response.json()['augrc_hw_val']
        acc_hw_val = response.json()['acc_hw_val']
        
        augrc_hw_test = response.json()['augrc_hw_test']
        acc_hw_test = response.json()['acc_hw_test']
        
        print("Server Response:", response.json())  # Assuming the response is JSON
    else:
        print("Error:", response.status_code, response.text)
    print('send file ok')
    return jsonify({'augrc_hw_train': float(augrc_hw_train), 'acc_hw_train': float(acc_hw_train), 'augrc_hw_val': float(augrc_hw_val), 'acc_hw_val': float(acc_hw_val), 'augrc_hw_test': float(augrc_hw_test), 'acc_hw_test': float(acc_hw_test)})

# validation_service.py inside the Docker container

app = Flask(__name__)
@app.route('/validate', methods=['POST'])
def validate():
    try:
        data = request.get_json()
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
        tflite_path = main_onnx(run_id, file_path, os.path.join('../models', f'model_{run_id}.onnx'))
        print(tflite_path)
        respon = send_file(tflite_path, run_id) #send_file(os.path.join(cfg.training.save_path, str(run_id), 'tflite', 'model.tflite'), run_id)
    except Exception as e:
        #return jsonify({"error": str(e)}), 400
        with open('sasaa.txt', "w", encoding="utf-8") as f:
            f.write(str(e))
        return jsonify({'error': str(e)}), 400
    return respon, 200

#### FOR REAL TIME QUANT

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)

#### FOR POST TRAINING QUANT
'''
if __name__ == '__main__':
    run_id = 56
    pth_file = f'/home/alema416/BASELINE_{run_id}.pth'
    model = load_checkpoint1(pth_file)
    export_and_simplify(model, f'bsln_{run_id}.onnx', opset=13)
    print('pth --> onnx done')
    #app.run(host='0.0.0.0', port=5002)
    script = "/home/alema416/dev/work/ST_stm32ai-modelzoo-services/image_classification/src/stm32ai_main.py"
    cfg_path = "/home/alema416/dev/work/HI4Lines_Insp/configs/"
    cfg_name = "quantization_config"
    mdl_path = f"/home/alema416/dev/work/HI4Lines_Insp/st_src/bsln_{run_id}.onnx"
    print(get_quantized_models_dir(run_id, script, cfg_path, cfg_name, mdl_path))
'''