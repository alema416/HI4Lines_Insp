from flask import Flask, request, jsonify
from datetime import datetime
import subprocess
import glob
import os
app = Flask(__name__)

@app.route('/validate', methods=['POST'])

def validate():
    scripts = []
    data = request.get_json()
    threshold = data.get('threshold')
    split = data.get('split')
    print(f'received threshold: {threshold}')
    
    script = {"file": f"pipeline_1.py", "args": ["--THRES", str(threshold), "--SPLIT", split]}
    
    command = ["python", script["file"]] + script["args"]
    results = {}
    
    try:
        output_dir = f'./runs/threshold_{threshold}/{split}'
        print(output_dir)
        a = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(a)
        # the above command creates a directory of txt files. CHATGPT: how can i return these txt files to the server?
        txt_files = glob.glob(os.path.join(output_dir, "*.txt"))
        txt_contents = {}
        
        for txt_file in txt_files:
            with open(txt_file, 'r') as file:
                filename = os.path.basename(txt_file)
                txt_contents[filename] = file.read()

        results[script["file"]] = txt_contents
    except subprocess.CalledProcessError as e:
        print(f"Error running {script['file']}:")
        print(e.stderr)

    return jsonify(results)

if __name__ == '__main__':
    ip = input('IP address: ') #192.168.2.53
    app.run(host=ip, port=5005)