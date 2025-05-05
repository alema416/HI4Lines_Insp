#!/usr/bin/env python3
import os
import sys
import json
import base64
import requests

def send_file(filename, run_id, url="http://192.168.1.11:5001/validate"):
    # Read and encode the file
    with open(filename, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "file": encoded,
        "run_id": run_id
    }
    headers = {"Content-Type": "application/json"}
    
    # Send the request
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
    #print({'acc_emu': float(acc_emu), 'augrc_emu': float(augrc_emu), 'augrc_hw': float(augrc_hw), 'acc_hw': float(acc_hw)})
    #return jsonify({'acc_emu': float(1.2), 'augrc_emu': float(1.2), 'augrc_hw': float(1.2), 'acc_hw': float(1.2)})
    return jsonify({'augrc_hw_train': float(augrc_hw_train), 'acc_hw_train': float(acc_hw_train), 'augrc_hw_val': float(augrc_hw_val), 'acc_hw_val': float(acc_hw_val), 'augrc_hw_test': float(augrc_hw_test), 'acc_hw_test': float(acc_hw_test)})

# validation_service.py inside the Docker container
from flask import Flask, request, jsonify
from datetime import datetime
import subprocess
import base64
import os
from hydra import initialize, compose
with initialize(config_path="./configs/"):
    cfg = compose(config_name="base")  # exp1.yaml with defaults key

app = Flask(__name__)
@app.route('/validate', methods=['POST'])
def validate():
    try:
        data = request.get_json()
        run_id = data.get('run_id')
        print(f'received run_id {run_id}')
        # pth --> tflite
        respon = send_file(os.path.join(cfg.training.save_path, run_id, 'tflite', 'model.tflite'), run_id)
    except Exception as e:
        #return jsonify({"error": str(e)}), 400
        with open('sasaa.txt', "w", encoding="utf-8") as f:
            f.write(str(e))
        return jsonify({'error': str(e)}), 400
    return respon, 200
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
