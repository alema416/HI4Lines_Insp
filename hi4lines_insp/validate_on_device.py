import os
import base64
import requests
from .serv import validate
'''
from .ckpt2onnx import toONNX

def validate_on_hailo(cfg, RUN_ID):
    toONNX()
    ccc = 0
    hailo_ip = cfg.training.ds_device_ip_hailo
    port = cfg.training.hailo_ip
    while ccc < 10:
        try:
            response = requests.post(f"http://{hailo_ip}:{port}/validate", json={"run_id": RUN_ID})
            response.raise_for_status()
            break
        except requests.RequestException as e:
            print(e)
            print(f"================ERROR #{ccc}================")
            ccc += 1
            continue

    result = response.json()
    return result
'''
def validate_on_orca(cfg, RUN_ID):
    stmz = cfg.training.ds_device_ip_orca
    ckpt_loc = os.path.join(cfg.training.save_path, str(RUN_ID), 'model_state_dict', 'model.pth')
    with open(ckpt_loc, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "file": encoded,
        "run_id": str(RUN_ID)
    }
    response = validate(payload)    
    result = response
    return result

def validate_on_coral(cfg, RUN_ID):
    stmz = cfg.training.ds_device_ip_coral
    ckpt_loc = os.path.join(cfg.training.save_path, str(RUN_ID), 'model_state_dict', 'model.pth')
    with open(ckpt_loc, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "file": encoded,
        "run_id": str(RUN_ID),
        "target_dev": cfg.training.target_dev
    }
    response = validate(payload)
    result = response
    return result

def validate_on_device(device_name, cfg, RUN_ID):
    if device_name == 'orca':
        return validate_on_orca(cfg, RUN_ID)
    elif device_name == 'coral':
        return validate_on_coral(cfg, RUN_ID)
    elif device_name == 'hailo':
        return validate_on_hailo(cfg, RUN_ID)
    else:
        print('error')
        return -1
