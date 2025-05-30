from PIL import Image
import argparse
import numpy as np
import pandas as pd
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
                    print('single')
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
                    #print(conf)
                '''
                else:
                    print('multi')
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
        print(f"Evaluated {total} images in {split}")
        print(f"SPECIAL_PRINTacc{split} {acc * 100:.2f}")
        print(f"Accuracy : {correct}/{total} = {acc * 100:.2f}%")

        # 5) Dump to files
        with open(f"b_labels_{id}_{split}.txt", "w") as f_lbl:
            for v in file_lbls:
                f_lbl.write(f"{v}\n")
        with open(f"b_confs_{id}_{split}.txt", "w") as f_conf:
            for v in file_confs:
                f_conf.write(f"{v:.6f}\n")
        df = pd.DataFrame(rows)
        df.to_csv(f'per_sample_{split}.csv', index=False)

    prof_file = session.end_profiling()
    print("Profiling data written to", prof_file)

    # 6) Overall latency
    print(f"Avg latency: {np.mean(latencies):.1f} ms  (over {len(latencies)} inferences)")

if __name__ == "__main__":
    for id in [32, 49, 56]:
        run_eval(id, f'MODEL_{id}.ONNX', '../../../data/processed/IDID_cropped_224')
