# validation_service.py inside the Docker container
from flask import Flask, request, jsonify
from datetime import datetime
import time
import subprocess
import os
import base64
import requests
from hydra import initialize, compose

app = Flask(__name__)
with initialize(config_path="../configs/"):
    cfg = compose(config_name="optimizer")  # exp1.yaml with defaults key

exp_name = cfg.optimizer.exp_name
online = True
@app.route('/validate', methods=['POST'])
def validate(): #run_id_given):
    scripts = []
    if online:
        data = request.get_json()
        run_id = data.get('run_id')
        print(f'received run_id {run_id}')
    else:
        run_id = run_id_given
        print(f'received run_id {run_id}')

    for j in ['ckpt2onnx', 'parser', 'optimizer']:
        scripts.append({"file": f"{j}.py", "args": ["--run_id", str(run_id)]})
    scripts.append({"file": f"compiler.py", "args": ["--run_id", str(run_id)]})
    
    augrc_emu = 9999.999
    acc_emu = 9999.999
    for script in scripts:
        command = ["python", script["file"]] + script["args"]
        start_time = time.time()
        print(f'running {command[1]}')
        
        try:
            # Run the command; check=True raises an error if the command fails
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            keyword = "SPECIAL_PRINTquantizedvalaugrc"
            specific_line = next((line for line in result.stdout.splitlines() if keyword in line), None)
            
            # If the specific line was found, print it.
            if specific_line:
                parts = specific_line.split(keyword)
                if len(parts) > 1:
                    augrc_emu = parts[1].strip().split()[0]
            
            keyword = "SPECIAL_PRINTquantizedvalacc"
            specific_line = next((line for line in result.stdout.splitlines() if keyword in line), None)
            
            # If the specific line was found, print it.
            if specific_line:
                parts = specific_line.split(keyword)
                if len(parts) > 1:
                    acc_emu = parts[1].strip().split()[0]       
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error running {script['file']}:")
            print(e.stderr)
        
        print(f'done in {((time.time() - start_time)/60):.1f} min') 
    print(f'emulator AUGRC: {augrc_emu}')
    print(f'emulator acc: {acc_emu}')
    
    # Define the server URL (change if running on a different host)
    rpi_ip = cfg.optimizer.rpi_ip #input('RPI IP address: ')
    SERVER_URL = f"http://{rpi_ip}:5001/validate"  # Update with actual server address 
    hef_dir = f'../models/{exp_name}/{run_id}'
    # File to send 
    FILE_PATH = os.path.join(hef_dir, f'model_{run_id}.hef')  # Change this to your actual file path
    
    # Open the file in binary mode and send it
    res = 400
    ccc = 0
    while ccc < 10:
        with open(FILE_PATH, "rb") as file:
            file_content = file.read()
            encoded_file = base64.b64encode(file_content).decode('utf-8')
            print(f'sending {FILE_PATH}')
            payload = {
              'run_id': str(run_id),
              'file': encoded_file
            }
    
            # Send the POST request with the JSON payload
            headers = {'Content-Type': 'application/json'}
            try:
                response = requests.post(SERVER_URL, json=payload, headers=headers)
                res = response.status_code # and maybe response.raise_for_status()
                print(res)
                if res == 200:
                    print("Successfully received response from device.")
                    break  # Exit the loop on success
            #files = {"file": (FILE_PATH, file)}
            #json = {'run_id': run_id}
            #response = requests.post(SERVER_URL, files=files)
            except requests.RequestException as e:
                print(e)
                print(f"================ERROR #{ccc}================")
                time.sleep(60)
                ccc += 1
                continue
    # Check the response
    
        
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
    
    if float(augrc_emu) > float(1000):
        a = input('problem: ')
    return jsonify({'acc_emu': float(acc_emu), 'augrc_emu': float(augrc_emu), 'augrc_hw_train': augrc_hw_train, 'acc_hw_train': acc_hw_train, 'augrc_hw_val': augrc_hw_val, 'acc_hw_val': acc_hw_val, 'augrc_hw_test': augrc_hw_test, 'acc_hw_test': acc_hw_test}), 200
    #return {'acc_emu': float(acc_emu), 'augrc_emu': float(augrc_emu)}
if __name__ == '__main__':
    
    server2 = False
    port = 5001 if server2 else 5000
    app.run(host='0.0.0.0', port=port)
    #validate()
    '''
    for i in [1, 2, 3, 4, 8, 9, 12, 13, 14, 17, 22, 26, 33, 35, 40, 44, 48, 51, 54, 69, 74, 75, 83, 88]:
        print(validate(i))
    '''
