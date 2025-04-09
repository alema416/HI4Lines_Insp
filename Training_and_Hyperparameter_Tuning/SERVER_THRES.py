import requests
import json
import os
import numpy as np
import shutil
import pandas as pd
from evaluator import get_mAP

url = "http://192.168.2.53:5005/validate"  # Adjust if your Flask API runs elsewhere
rows = []
for threshold in np.arange(0.5, 1.01, 0.05):
    info = []
    for split in ['train', 'val', 'test']:

        payload = {
            "threshold": f'{threshold:.2f}',
            'split': split
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
        
            # Choose where to save the output
            save_dir = f"./reconstructed_txt_files_{split}/thres_{threshold:.2f}"
            print(save_dir)
            os.makedirs(save_dir, exist_ok=True)
        
            for script, files in data.items():
                for filename, content in files.items():
                    file_path = os.path.join(save_dir, filename)
                    with open(file_path, 'w') as f:
                        f.write(content)
        else:
            print(f"Error {response.status_code}: {response.text}")
        num = None
        if split == 'train':
            num = 1024
        elif split == 'val':
            num = 256
        else:
            num = 320
        offloadedsamples = num - len(os.listdir(save_dir))
        print(f'{offloadedsamples} samples offloaded')
        info.append(offloadedsamples)
        info.append(offloadedsamples/num)
        print(f'{threshold:.2f} done!')
        
        # Set your directories
        #save_dir = f"./reconstructed_txt_files_{split}/thres_{threshold:.2f}"
        dir_a = f'../inference_tests/edge_server_{split}_swapped/labels/' # edge server inference results
        dir_b = save_dir # drone inference results
        
        # Make sure target directory exists
        os.makedirs(dir_b, exist_ok=True)
        
        # Loop through files in directory A
        for filename in os.listdir(dir_a):
            source_file = os.path.join(dir_a, filename)
            target_file = os.path.join(dir_b, filename)
        
            # Check if it's a file (not a subdirectory), and not already in B
            if os.path.isfile(source_file) and not os.path.exists(target_file):
                shutil.copy2(source_file, target_file)
        num = 320 if split == 'test' else 256
        num = 1024 if split == 'train' else 0
    row = {'conf_thres': f'{threshold:.2f}', 'offloaded samples train': info[0], 'offloading rate train': info[1], 'offloaded samples val': info[2], 'offloading rate val': info[3], 'offloaded samples test': info[4], 'offloading rate test': info[5], 'mAP cooperative train': get_mAP('train', f'{threshold:.2f}'), 'mAP cooperative val': get_mAP('val', f'{threshold:.2f}'), 'mAP cooperative test': get_mAP('test', f'{threshold:.2f}')}
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv('exp_0.csv', index=False)
print(df)

print(f"COOP RESULTS (OK) AT: ./reconstructed_txt_files_split")
print(f"EDGE SERVER RESULTS (OK) AT: ../inference_tests/edge_server_split/labels/")
print(f"DRONE RESULTS (PENDING) AT: ../inference_tests/drone_split/")