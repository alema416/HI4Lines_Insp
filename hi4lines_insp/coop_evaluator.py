import subprocess
import re
import sys
import numpy as np
import pandas as pd
def get_mAP(split, thres):
    
    script = {"file": f"../test/review_object_detection_metrics/cli.py", "args": ["--img", f'/home/amax/machairas/FMFP-edge-idid/yolo_m1_JOIN_1280_debug/{split}/images/', "--anno_gt", f'/home/amax/machairas/FMFP-edge-idid/yolo_m1_JOIN_1280_debug/{split}/labels/', '--anno_det', f'/home/amax/machairas/FMFP-edge-idid/reconstructed_txt_files_{split}/thres_{thres}/', '--format_gt', 'yolo', '--format_det', 'xcycwh', '--coord_det', 'rel', '--metric', 'voc2012', '--name', '/home/amax/machairas/FMFP-edge-idid/class.txt', '--threshold', '0.5', '--save_path', './', '--plot']}
    command = ["python", script["file"]] + script["args"]
    
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)    

    if result.returncode != 0:
        print("Error:", result.stderr)
        sys.exit(result.returncode)

    match = re.search(r'mAP:\s*([0-9.]+)', result.stdout)
    if not match:
        print("ERROR")
    return match.group(1)

def main():
    rows = []
    for thres in [f'{i:.2f}' for i in np.arange(0.5, 1.01, 0.05)]:
        print(thres)
        row = {'conf_thres', thres, 'mAP cooperative train', get_mAP('train', thres), 'mAP cooperative val', get_mAP('val', thres), 'mAP cooperative test', get_mAP('test', thres)}
        rows.append(row)
    df = pd.DataFrame(rows)
    print(df)
if __name__ == "__main__":
    main()