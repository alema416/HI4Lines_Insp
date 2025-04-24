# validation_service.py inside the Docker container
from flask import Flask, request, jsonify
from datetime import datetime
import subprocess
import base64
import os
from hydra import initialize, compose

app = Flask(__name__)
@app.route('/validate', methods=['POST'])
def validate():
    with initialize(config_path="../../configs/"):
        cfg = compose(config_name="hw_eval_server")  # exp1.yaml with defaults key

    UPLOAD_FOLDER = cfg.server.upload_dir
    LOCK_FILE = cfg.server.lock_file
    
    scripts = []
    if os.path.exists(LOCK_FILE):
	      return jsonify({"error": 'device locked'}), 400
    with open(LOCK_FILE, 'w') as lock_file:
        lock_file.write("locked")
    try:
        data = request.get_json()
        encoded_file = data.get('file')  # Get the Base64 encoded file
        run_id = data.get('run_id')
        print(f'received run_id {run_id}')
        
    except Exception as e:
        #return jsonify({"error": str(e)}), 400
        with open('sasaa.txt', "w", encoding="utf-8") as f:
            f.write(str(e))
        return jsonify({'error': str(e)}), 400
    if encoded_file:
        file_content = base64.b64decode(encoded_file)
        file_path = os.path.join(UPLOAD_FOLDER, cfg.server.temp_model_flnm)
        with open(file_path, 'wb') as f:
            f.write(file_content)
        print(f"File {file_path} received and saved.")

    for j in ['class_eval', 'metric']:
        scripts.append({"file": f"{j}.py", "args": ["--model", cfg.server.temp_model_flnm]})
    
    augrc_hw_train = None
    acc_hw_train = None
    augrc_hw_val = None
    acc_hw_val = None
    augrc_hw_test = None
    acc_hw_test = None
    
    for script in scripts:
        command = ["python", script["file"]] + script["args"]
        try:
            # Run the command; check=True raises an error if the command fails
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            keyword = "SPECIAL_PRINTaugrctrain"
            specific_line = next((line for line in result.stdout.splitlines() if keyword in line), None)
            
            # If the specific line was found, print it.
            if specific_line:
                parts = specific_line.split(keyword)
                if len(parts) > 1:
                    augrc_hw_train = parts[1].strip().split()[0]
            
            keyword = "SPECIAL_PRINTaugrcval"
            specific_line = next((line for line in result.stdout.splitlines() if keyword in line), None)
            
            # If the specific line was found, print it.
            if specific_line:
                parts = specific_line.split(keyword)
                if len(parts) > 1:
                    augrc_hw_val = parts[1].strip().split()[0]
            
            keyword = "SPECIAL_PRINTaugrctest"
            specific_line = next((line for line in result.stdout.splitlines() if keyword in line), None)
            
            # If the specific line was found, print it.
            if specific_line:
                parts = specific_line.split(keyword)
                if len(parts) > 1:
                    augrc_hw_test = parts[1].strip().split()[0]
            
            keyword = "SPECIAL_PRINTacctrain"
            specific_line = next((line for line in result.stdout.splitlines() if keyword in line), None)
            
            # If the specific line was found, print it.
            if specific_line:
                parts = specific_line.split(keyword)
                if len(parts) > 1:
                    acc_hw_train = parts[1].strip().split()[0]
            
            
            keyword = "SPECIAL_PRINTaccval"
            specific_line = next((line for line in result.stdout.splitlines() if keyword in line), None)
            
            # If the specific line was found, print it.
            if specific_line:
                parts = specific_line.split(keyword)
                if len(parts) > 1:
                    acc_hw_val = parts[1].strip().split()[0]
            keyword = "SPECIAL_PRINTacctest"
            specific_line = next((line for line in result.stdout.splitlines() if keyword in line), None)
            
            # If the specific line was found, print it.
            if specific_line:
                parts = specific_line.split(keyword)
                if len(parts) > 1:
                    acc_hw_test = parts[1].strip().split()[0]
            
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error running {script['file']}:")
            print(e.stderr)
            os.remove(LOCK_FILE)
            with open('sasaa.txt', "w", encoding="utf-8") as f:
                f.write(str(e.stderr))         # write the string
                f.write("\n")         # optionally add a newline
    os.remove(LOCK_FILE)
    return jsonify({'acc_hw_train': acc_hw_train, 'augrc_hw_train': augrc_hw_train, 'acc_hw_val': acc_hw_val, 'augrc_hw_val': augrc_hw_val, 'acc_hw_test': acc_hw_test, 'augrc_hw_test': augrc_hw_test}), 200

if __name__ == '__main__':
    app.run(host='192.168.2.98', port=5001)
