import subprocess
import re
import sys
import numpy as np
import pandas as pd
import os
import shutil

def copy_matching_files(dir_a, dir_b, dir_c):
    """
    For each file in directory A, copies the file with the same filename from directory B to directory C.
    
    Parameters:
    - dir_a (str): Path to the directory A (source of filenames).
    - dir_b (str): Path to the directory B (source of files to copy).
    - dir_c (str): Path to the directory C (destination directory).
    """
    # Ensure the destination directory exists
    os.makedirs(dir_c, exist_ok=True)
    cp = 0
    # Iterate over each file in directory A
    for filename in os.listdir(dir_a):
        # Construct the full path for the file in A
        file_a_path = os.path.join(dir_a, filename)
        # Only consider files (skip subdirectories)
        #print(f'{file_a_path}')
        if os.path.isfile(file_a_path):
            # Construct the expected source file path in directory B
            file_b_path = os.path.join(dir_b, filename)
            # Check if the file exists in directory B
            #print(f'{file_b_path}')
            if os.path.isfile(file_b_path):
                # Define the destination file path in directory C
                file_c_path = os.path.join(dir_c, filename)
                # Copy the file from B to C
                shutil.copy(file_b_path, file_c_path)
                #print(f"Copied {filename} from {dir_b} to {dir_c}.")
                
                cp += 1
                #print(f"File '{filename}' not found in {dir_b}.")
    print(f'copy {cp}')
    print(f'did not copy {len(os.listdir(dir_b)) - cp}')
                
def get_mAP(split, thres, off):
    SERVER = f'/home/amax/GitHub/hailo_examples/assets/edge_server_{split}_swapped/'
    dir_path = f'./newgt_{thres}_{split}{off}'
    dir_server_path = f'./newserpred_{thres}_{split}{off}'
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    
    GT = './assets/yolo_m1_JOIN_1280_debug/'
    copy_matching_files(f'/home/amax/GitHub/hailo_examples/runs/threshold_{thres}{off}/{split}/', os.path.join(GT, split, 'labels'), dir_path)
    copy_matching_files(f'/home/amax/GitHub/hailo_examples/runs/threshold_{thres}{off}/{split}/', os.path.join(SERVER, 'labels'), dir_server_path)
    
    if len(os.listdir(dir_path)) == 0:
        return [None, None]
    out = []
    
    script = {"file": f"../review_object_detection_metrics/cli.py", "args": ["--img", f'./assets/yolo_m1_JOIN_1280_debug/{split}/images/', "--anno_gt", dir_path, '--anno_det', f'/home/amax/GitHub/hailo_examples/runs/threshold_{thres}{off}/{split}/', '--format_gt', 'yolo', '--format_det', 'xcycwh', '--coord_det', 'rel', '--metric', 'voc2012', '--name', 'class.txt', '--threshold', '0.5', '--save_path', './', '--plot']}
    command = ["python", script["file"]] + script["args"]
    
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)    
    print(result)
    if result.returncode != 0:
        print("Error:", result.stderr) 
        sys.exit(result.returncode)

    match = re.search(r'mAP:\s*([0-9.]+)', result.stdout)
    if not match:
        print("ERROR")
    
    out.append(match.group(1))
        
    script = {"file": f"../review_object_detection_metrics/cli.py", "args": ["--img", f'./assets/yolo_m1_JOIN_1280_debug/{split}/images/', "--anno_gt", dir_path, '--anno_det', dir_server_path, '--format_gt', 'yolo', '--format_det', 'xcycwh', '--coord_det', 'rel', '--metric', 'voc2012', '--name', 'class.txt', '--threshold', '0.5', '--save_path', './', '--plot']}
    command = ["python", script["file"]] + script["args"]
    
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)    
    print(result)
    if result.returncode != 0:
        print("Error:", result.stderr) 
        sys.exit(result.returncode)

    match = re.search(r'mAP:\s*([0-9.]+)', result.stdout)
    if not match:
        print("ERROR")
        
    print(f"{thres} {split} {'keepers' if off == '' else 'offloaders'} done")
    out.append(match.group(1))
    return out

    
def main():
    rows = []
    for thres in [f'{i:.2f}' for i in np.arange(0.5, 1.01, 0.05)]:
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
    df.to_csv('exp_0.csv', index=False)
if __name__ == "__main__":
    main()
