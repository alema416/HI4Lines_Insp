import os
import base64
import requests
from .serv import validate

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
    else:
        print('error')
        return -1
