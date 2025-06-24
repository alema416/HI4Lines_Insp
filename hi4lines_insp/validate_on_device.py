import os
import base64
import requests
from .serv import validate
def validate_on_device(device_name, cfg, RUN_ID):
    # ORCA CODE
    ccc = 0
    stmz = cfg.training.ds_device_ip_orca
    ckpt_loc = os.path.join(cfg.training.save_path, str(RUN_ID), 'model_state_dict', 'model.pth')
    with open(os.path.join('../models/coral_optim_mbln_wed', str(10), 'model_state_dict', 'model.pth'), "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "file": encoded,
        "run_id": str(RUN_ID)
    }
    response = validate(payload)    
    result = response
    return result