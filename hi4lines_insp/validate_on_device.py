import os
import base64
import requests
from serv1 import validate
import time

def send_file(filename, run_id, cfg):
    url=f"http://{cfg.training.orca_dev_ip}:{cfg.training.orca_port}/validate"
    # Read and encode the file
    print(f'send file enter to {url}')
    print(f'filename={filename}')
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
    return {'augrc_hw_train': float(augrc_hw_train), 'acc_hw_train': float(acc_hw_train), 'augrc_hw_val': float(augrc_hw_val), 'acc_hw_val': float(acc_hw_val), 'augrc_hw_test': float(augrc_hw_test), 'acc_hw_test': float(acc_hw_test)}


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
        "run_id": str(RUN_ID), 
        "target_dev": cfg.training.target_dev
    }
    response = validate(payload)        
    print(response)
    #time.sleep(30)
    return response

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
    print(response)
    return response

def validate_on_device(device_name, cfg, RUN_ID):
    if device_name == 'orca':
        response = validate_on_orca(cfg, RUN_ID)
        return send_file(response['tflite_path'], response['run_id'], cfg)
    elif device_name == 'coral':
        response = validate_on_coral(cfg, RUN_ID)
        return send_file(response['tflite_path'], response['run_id'], cfg)
    elif device_name == 'hailo':
        return validate_on_hailo(cfg, RUN_ID)
    else:
        print('error')
        return -1

def main():
    #pass
    from hydra import initialize, compose
    with initialize(config_path="../configs/"):
        cfg = compose(config_name="fmfp")  # exp1.yaml with defaults key

    validate_on_device('coral', cfg, 11)

if __name__ == '__main__':
    main()