import subprocess
import re
import sys
import ast
import numpy as np
import pandas as pd
import os
import shutil
                
def get_mAP(split, thres, off):
    dir_path = f'./newgt_{thres}_{split}{off}'
    dir_drone_path = f'/home/amax/GitHub/hailo_examples/runs/threshold_{thres}{off}/{split}/'
    dir_server_path = f'./newserpred_{thres}_{split}{off}'
    
    offname = 'offloaders' if off == '_offloaders' else 'keepers'
    out = []
    script = {"file": f"/home/amax/GitHub/object_detection_metrics_calculation/main.py", "args": ["--path_to_gt", dir_path, '--path_to_png', f'./assets/yolo_m1_JOIN_1280_debug/{split}/images/', '--path_to_pred', dir_drone_path, '--path_to_results', './', '--filename', f'./afternoon1/persample_{thres}_{split}_{offname}_drone']}
    command = ["python", script["file"]] + script["args"]
    
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)    
    if result.returncode != 0:
        print("Error:", result.stderr) 
        sys.exit(result.returncode)

    # Try to extract the dictionary printed to stdout
    match = re.search(r"\{.*\}", result.stdout, re.DOTALL)
    if not match:
        print("ERROR: Couldn't find output dictionary")
        sys.exit(1)

    try:
        cleaned = match.group(0).replace("nan", "'nan'")
        metrics = ast.literal_eval(cleaned)
        #metrics = ast.literal_eval(match.group(0))  # Safely convert string to dict
        ap50 = metrics.get("AP50")
        if ap50 is None:
            raise ValueError("AP50 not found")
        out.append(ap50)
    except Exception as e:
        print("ERROR parsing output:", e)
        sys.exit(1)

    
    script = {"file": f"/home/amax/GitHub/object_detection_metrics_calculation/main.py", "args": ["--path_to_gt", dir_path, '--path_to_png', f'./assets/yolo_m1_JOIN_1280_debug/{split}/images/', '--path_to_pred', dir_server_path, '--path_to_results', './', '--filename', f'./afternoon1/persample_{thres}_{split}_{offname}_server']}
    command = ["python", script["file"]] + script["args"]
    
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)    
    if result.returncode != 0:
        print("Error:", result.stderr) 
        sys.exit(result.returncode)

    # Try to extract the dictionary printed to stdout
    match = re.search(r"\{.*\}", result.stdout, re.DOTALL)
    if not match:
        print("ERROR  : Couldn't find output dictionary")
        sys.exit(1)

    try:
        cleaned = match.group(0).replace("nan", "'nan'")
        metrics = ast.literal_eval(cleaned)
        #metrics = ast.literal_eval(match.group(0))  # Safely convert string to dict
        ap50 = metrics.get("AP50")
        if ap50 is None:
            raise ValueError("AP50 not found")
        out.append(ap50)
    except Exception as e:
        print("ERROR parsing output:", e)
        sys.exit(1)
    print(f'{split} {thres} {off} gt: {len(os.listdir(dir_path))} drone pred: {len(os.listdir(dir_drone_path))} server pred: {len(os.listdir(dir_server_path))} drone {out[0]:.2f} server {out[1]:.2f}')
    return out

    
def main():
    rows = []
    for thres in [f'{i:.2f}' for i in np.arange(0.55, 0.951, 0.05)]:
        trainoff = get_mAP('train', thres, '_offloaders')
        valoff = get_mAP('val', thres, '_offloaders')
        testoff = get_mAP('test', thres, '_offloaders')
        trainkeep = get_mAP('train', thres, '')
        valkeep = get_mAP('val', thres, '')
        testkeep = get_mAP('test', thres, '')
        row = {'conf_thres': thres, 'mAP train offloaders drone': trainoff[0], 'mAP train offloaders server': trainoff[1], 'mAP val offloaders drone': valoff[0], 'mAP val offloaders server': valoff[1], 'mAP test offloaders drone': testoff[0], 'mAP test offloaders server': testoff[1], 'mAP train keepers drone': trainkeep[0], 'mAP train keepers server': trainkeep[1], 'mAP val keepers drone': valkeep[0], 'mAP val keepers server': valkeep[1], 'mAP test keepers drone': testkeep[0], 'mAP test keepers server': testkeep[1]}
        
        rows.append(row)
    df = pd.DataFrame(rows)
    print(df)
    df.to_csv('./afternoon1/exp_0_persample.csv', index=False)
if __name__ == "__main__":
    main()
