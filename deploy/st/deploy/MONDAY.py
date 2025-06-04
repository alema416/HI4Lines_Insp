import pandas as pd
from nmetr import AUGRC
import argparse
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd
import seaborn as sns
import tqdm
import matplotlib.pyplot as plt
import argparse
from PIL import Image
import onnxruntime as ort
import os
from timeit import default_timer as timer

def preprocess_image(path: str, input_shape, input_dtype):
    dims = [d if isinstance(d, int) else 1 for d in input_shape]  
    # Determine channel ordering:
    #  - if dims[1] is 1 or 3 → channels-first  (N,C,H,W)
    #  - otherwise           → channels-last   (N,H,W,C)
    nchw = (dims[1] in (1,3))
    
    if nchw:
        _, C, H, W = dims
        img = Image.open(path).convert('RGB').resize((W, H))
        arr = np.asarray(img, dtype=input_dtype)           # (H,W,3)
        scale  = np.array([0.0213538, 0.0208032, 0.0197237], dtype=np.float32)
        offset = np.array([-2.7478418, -2.4465773, -2.1585231], dtype=np.float32)
        arr = arr * scale + offset            # broadcast per-channel
        arr = arr.transpose(2, 0, 1)                        # → (3,H,W)
    else:
        _, H, W, C = dims
        img = Image.open(path).convert('RGB').resize((W, H))
        arr = np.asarray(img, dtype=input_dtype)           # (H,W,3)

    # add batch dimension
    return np.expand_dims(arr, axis=0)  # → (1,C,H,W) or (1,H,W,C)


def run_eval(id: int, model_path: str, data_root: str):
    # --- 1) Create ONNX Runtime session with VSI NPU EP ---
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        model_path,
        sess_options=sess_opts,
        providers=['VSINPUExecutionProvider']
    )
    sess_opts.enable_profiling = True
    sess_opts.log_severity_level = 0  # VERBOSE
    sess_opts.profile_file_prefix = "/tmp/kadu_model_profile" 
    # --- 2) Discover input/output metadata ---
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    
    input_name  = inp.name
    input_shape = inp.shape       # e.g. [1,224,224,3]
    input_type = inp.type
    # Map ONNX dtype string to numpy dtype
    type_map = {
        'tensor(float)': np.float32,
        'tensor(float16)': np.float16,
        'tensor(double)': np.float64,
        'tensor(int32)': np.int32,
        'tensor(int64)': np.int64,
        'tensor(int8)':  np.int8,
        'tensor(uint8)': np.uint8,
        'tensor(bool)':  np.bool_
    }
    input_dtype = type_map.get(input_type, np.float32)

    labels = ['broken', 'healthy']
    latencies = []
    
    # --- 3) Loop over splits ---
    ACC_dict   = {}      # accuracy per split
    AUGRC_dict = {}      # AUGRC per split
    mean       = {}      # mean confidences
    std        = {}      # std confidences
    cc         = {}      # counts of successes/errors
    for split in ['train', 'val', 'test']:
        total = correct = 0
        file_lbls = []
        file_confs = []
        rows = []
        for cls_idx, cls_name in enumerate(labels):
            folder = os.path.join(data_root, split, cls_name)
            if not os.path.isdir(folder):
                continue

            for fn in os.listdir(folder):
                t0 = timer()
                img_path = os.path.join(folder, fn)
                # Preprocess
                x = preprocess_image(img_path, input_shape, input_dtype)
                # run inference
                
                raw_out = session.run([out.name], {input_name: x})[0]  # -> e.g. shape (1,) or (1,C)
                
                arr = np.squeeze(raw_out, axis=0)                      # -> shape () or (C,)

                # decide prediction & confidence
                if arr.ndim == 0:
                    #print('single')
                    # single score in [0,1]
                    score = float(arr)
                    pred  = 1 if score > 0.5 else 0
                    conf  = score if pred == 1 else 1.0 - score
                else:
                    # Multiclass classification with logits
                    def softmax(x):
                        e_x = np.exp(x - np.max(x))
                        return e_x / e_x.sum()
                    probs = softmax(arr)
                    pred = int(np.argmax(probs))
                    conf = float(probs[pred])
                    ##print(conf)
                '''
                else:
                    #print('multi')
                    # multiclass: pick the highest-scoring class
                    # (if these are logits, you may want to softmax first)
                    pred = int(np.argmax(arr))
                    conf = float(arr[pred])
                '''
                total += 1
                if pred == cls_idx:
                    correct += 1
                    file_lbls.append(1)
                else:
                    file_lbls.append(0)

                file_confs.append(conf)
                t1 = timer()
                latencies.append((t1 - t0) * 1000.0)
                row = {'filename': os.path.basename(img_path), 'confidence': conf, 'prediction': pred, 'ground_truth': cls_idx, 'correct': 1 if pred == cls_idx else 0}
                rows.append(row)

        # 4) Report
        acc = correct / total if total else 0.0
        ACC_dict[split] = acc * 100.0
        successes_conf = [c for c,lbl in zip(file_confs, file_lbls) if lbl == 1]
        errors_conf    = [c for c,lbl in zip(file_confs, file_lbls) if lbl == 0]
        
        if split == 'train':
            cc['succ_tr'] = len(successes_conf)
            cc['err_tr']  = len(errors_conf)
        elif split == 'val':
            cc['succ_val'] = len(successes_conf)
            cc['err_val']  = len(errors_conf)
        else:  # split == 'test'
            cc['succ_test'] = len(successes_conf)
            cc['err_test']  = len(errors_conf)
        
        mean[f's_{split}'] = np.mean(successes_conf) if len(successes_conf) > 0 else 0.0
        std[f's_{split}']  = np.std(successes_conf)  if len(successes_conf) > 0 else 0.0
        mean[f'e_{split}'] = np.mean(errors_conf)    if len(errors_conf) > 0 else 0.0
        std[f'e_{split}']  = np.std(errors_conf)     if len(errors_conf) > 0 else 0.0
        
        #print(f"Evaluated {total} images in {split}")
        #print(f"SPECIAL_#printacc{split} {acc * 100:.2f}")
        #print(f"Accuracy : {correct}/{total} = {acc * 100:.2f}%")
        ide = ''
        # 5) Dump to files
        with open(f"{ide}labels_{id}_{split}.txt", "w") as f_lbl:
            for v in file_lbls:
                f_lbl.write(f"{v}\n")
        with open(f"{ide}confs_{id}_{split}.txt", "w") as f_conf:
            for v in file_confs:
                f_conf.write(f"{v:.6f}\n")
        with open(f'labels_{id}_{split}.txt', "r") as f_lbl:
            labels_list = [int(line.strip()) for line in f_lbl]
        with open(f'confs_{id}_{split}.txt', "r") as f_conf:
            confs_list = [float(line.strip()) for line in f_conf]

        probs_tensor  = torch.tensor(confs_list, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)
        augrc_metric = AUGRC()
        augrc_metric.update(probs_tensor, labels_tensor)
        augrc_value = augrc_metric.compute()
        AUGRC_dict[split] = 1000.0 * augrc_value.item()        
        
        
        
        df = pd.DataFrame(rows)
        df.to_csv(f'per_sample_{split}.csv', index=False)

    prof_file = session.end_profiling()
    
    print(f"ID={id}  →  ACC(train)={ACC_dict['train']:.2f}%,  ACC(val)={ACC_dict['val']:.2f}%,  ACC(test)={ACC_dict['test']:.2f}%")
    print(f"ID={id}  →  AUGRC(train)={AUGRC_dict['train']:.2f},  AUGRC(val)={AUGRC_dict['val']:.2f},  AUGRC(test)={AUGRC_dict['test']:.2f}")
    print(mean)
    print(std)
    print(cc)
    # ───────────────────────────────────────────────────────────────────
    # ─── (3) Combine train & val, then call custom_seaborn ──────────────
    df_tr_confs  = pd.read_csv(f'confs_{id}_train.txt', header=None, names=['confidence'])
    df_tr_lbls   = pd.read_csv(f'labels_{id}_train.txt', header=None, names=['correct'])
    df_val_confs = pd.read_csv(f'confs_{id}_val.txt',   header=None, names=['confidence'])
    df_val_lbls  = pd.read_csv(f'labels_{id}_val.txt',   header=None, names=['correct'])

    df_tr  = pd.concat([df_tr_confs,  df_tr_lbls],  axis=1)
    df_val = pd.concat([df_val_confs, df_val_lbls], axis=1)
    df_combined = pd.concat([df_tr, df_val], axis=0, ignore_index=True)

    df_zero = df_combined[df_combined['correct'] == 0]
    df_one  = df_combined[df_combined['correct'] == 1]

    custom_seaborn(
        df_one, df_zero,
        model_id = id,
        split    = 'train-val',
        prefix   = '',
        AUGRC    = AUGRC_dict,
        mean     = mean,
        std      = std,
        cc       = cc
    )

    # ─── (4) Now do the test split plot ──────────────
    df_test_confs = pd.read_csv(f'confs_{id}_test.txt',  header=None, names=['confidence'])
    df_test_lbls  = pd.read_csv(f'labels_{id}_test.txt',  header=None, names=['correct'])
    df_test       = pd.concat([df_test_confs, df_test_lbls], axis=1)

    df_zero_t = df_test[df_test['correct'] == 0]
    df_one_t  = df_test[df_test['correct'] == 1]

    custom_seaborn(
        df_one_t, df_zero_t,
        model_id = id,
        split    = 'test',
        prefix   = '',
        AUGRC    = AUGRC_dict,
        mean     = mean,
        std      = std,
        cc       = cc
    )
    #print("Profiling data written to", prof_file)

    # 6) Overall latency
    #print(f"Avg latency: {np.mean(latencies):.1f} ms  (over {len(latencies)} inferences)")

'''

parser = argparse.ArgumentParser(description='Rethinking CC for FP')
parser.add_argument('--run_id', required=True, type=int, help='')
parser.add_argument('--idel', required=True, type=str, help='')
args = parser.parse_args()
'''
def custom_seaborn(df_pos, df_neg, model_id, split, prefix, AUGRC, mean, std, cc):
    if len(df_pos) < 2 or len(df_neg) < 2:
        return

    # extract
    pos = df_pos['confidence']
    neg = df_neg['confidence']

    plt.figure(figsize=(6,4))
    # `cut=3` extends each KDE 3 bandwidths beyond the data min/max
    sns.kdeplot(pos, 
                fill=True, 
                color='green', 
                alpha=0.4, 
                bw_method='scott', 
                cut=3,
                label='correct predictions')
    sns.kdeplot(neg, 
                fill=True, 
                color='red', 
                alpha=0.4, 
                bw_method='scott', 
                cut=3,
                label='misclassifications')
    modelname = 'mobilenet' #input('Model Name: ') #'ResNet18'
    plt.xlabel('confidence')
    plt.ylabel('density')
    plt.xlim(0.0, 1.2)
    
    info = f"Model: {modelname}\nID: {id}\n"
    if split == 'test':
        info += f"AUGRC: {AUGRC['test']:.2f}\n"
        info += f"success mean: {mean['s_test']:.2f}, success st: {std['s_test']:.2f}\n"
        info += f"error mean: {mean['e_test']:.2f}, error std: {std['e_test']:.2f}\n"

    else:
        info += f"AUGRC: {AUGRC['train']:.2f}, {AUGRC['val']:.2f}\n"
        # given mu1, sigma1, n1, mu2, sigma2, n2:
        n1 = cc['succ_tr']
        n2 = cc['succ_val']

        mu1 = mean['s_train']
        mu2 = mean['s_val']
        sigma1 = std['s_train']
        sigma2 = std['s_val']
    
        mu_tot = (n1*mu1 + n2*mu2) / (n1 + n2)

        num = n1*(sigma1**2 + mu1**2) + n2*(sigma2**2 + mu2**2)
        sigma_tot_sq = num/(n1+n2) - mu_tot**2
        sigma_tot = np.sqrt(sigma_tot_sq)
        info += f"success mean: {mu_tot:.2f}, success std: {sigma_tot:.2f}\n"

        n1 = cc['err_tr']
        n2 = cc['err_val']

        mu1 = mean['e_train']
        mu2 = mean['e_val']
        sigma1 = std['e_train']
        sigma2 = std['e_val']
    
        mu_tot = (n1*mu1 + n2*mu2) / (n1 + n2)

        num = n1*(sigma1**2 + mu1**2) + n2*(sigma2**2 + mu2**2)
        sigma_tot_sq = num/(n1+n2) - mu_tot**2
        sigma_tot = np.sqrt(sigma_tot_sq)
        info += f"error mean: {mu_tot:.2f}, error std: {sigma_tot:.2f}\n"
    
    #print(prefix)	
    name = 'Baseline' if prefix=='b_' else 'FMFP'
    #print(name)
    
    plt.gca().text(
        0.02, 0.68, info,
        transform=plt.gca().transAxes,
        va='top', ha='left',
        fontsize=7,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6)
    )
    
    if split == 'test':
        plt.title(f'{name} - {modelname} - id={model_id} - {split} set')
    else:
        plt.title(f'{name} - {modelname} - id={model_id} - {split} sets')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f'./MONDAY/post_withtitles/custom_{prefix}{model_id}_{split}.png', dpi=300)
    plt.close()

'''
for id in range(60):
    #print(id)
    if os.path.isfile(f'./MONDAY/model_{id}_opset17_quant_qdq_pc.onnx'):
        run_eval(id, f'./MONDAY/model_{id}_opset17_quant_qdq_pc.onnx', '../../../data/processed/IDID_cropped_224')

for id in tqdm.tqdm():
    #print(id)
    idel = '' #args.idel
    if os.path.isfile(f'./MONDAY/model_{id}_opset17_quant_qdq_pc.onnx'):
        if True:
            if True:
                for ide in [idel]:
                    # Replace 'confidences.txt' and 'labels.txt' with your actual file paths
                    confidences = pd.read_csv(f'{idel}confs_{id}_train.txt', header=None, names=['confidence'])
                    labels = pd.read_csv(f'{idel}labels_{id}_train.txt', header=None, names=['correct'])

                    # Combine into a single DataFrame
                    df1 = pd.concat([confidences, labels], axis=1)

                    confidences = pd.read_csv(f'{idel}confs_{id}_val.txt', header=None, names=['confidence'])
                    labels = pd.read_csv(f'{idel}labels_{id}_val.txt', header=None, names=['correct'])

                    # Combine into a single DataFrame
                    df2 = pd.concat([confidences, labels], axis=1)


                    ##print(len(df1))
                    ##print(len(df2))
                    df = pd.concat([df1, df2], axis=0, ignore_index=True)
                    #print(len(df))
                    df_zero = df[df['correct'] == 0]
                    df_one = df[df['correct'] == 1]
                    #plt.xlim(x_min, x_max)

                    #plt.tight_layout()
                    custom_seaborn(df_one, df_zero, id, 'train-val', ide, AUGRC_dict, mean, std, cc)
                    df_zero.to_csv(f'./MONDAY/post_withtitles/output_{idel}{id}_valtrain_error.csv', index=False)
                    df_one.to_csv(f'./MONDAY/post_withtitles/output_{idel}{id}_valtrain_success.csv', index=False)

        if True:
            if True:
                for ide in [idel]:
                    # Replace 'confidences.txt' and 'labels.txt' with your actual file paths
                    confidences = pd.read_csv(f'{idel}confs_{id}_test.txt', header=None, names=['confidence'])
                    labels = pd.read_csv(f'{idel}labels_{id}_test.txt', header=None, names=['correct'])

                    # Combine into a single DataFrame
                    df = pd.concat([confidences, labels], axis=1)
                    #print(len(df))
                    df_zero = df[df['correct'] == 0]
                    df_one = df[df['correct'] == 1]

                    custom_seaborn(df_one, df_zero, id, 'test', ide, AUGRC_dict, mean, std, cc)

                    df_zero.to_csv(f'./MONDAY/post_withtitles/output_{idel}{id}_test_error.csv', index=False)
                    df_one.to_csv(f'./MONDAY/post_withtitles/output_{idel}{id}_test_success.csv', index=False)
'''
for id in tqdm.tqdm([0]):
    run_eval(id, f'./MONDAY/model_{id}_opset17_quant_qdq_pc.onnx', '../../../data/processed/IDID_cropped_224')
