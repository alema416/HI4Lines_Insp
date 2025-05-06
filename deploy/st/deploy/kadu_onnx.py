from PIL import Image
import argparse
import numpy as np
import onnxruntime as ort
import os
from timeit import default_timer as timer

def preprocess_image1(path: str, input_shape, input_dtype):
    """
    Load an image from disk, resize to (W,H) and return a batched numpy array.
    """
    # ONNX shapes are typically [N,H,W,C] or [N,C,H,W]; detect channel order by length
    # We'll assume [N,H,W,C] here for simplicity.
    _, height, width, channels = input_shape
    img = Image.open(path).convert('RGB').resize((width, height))
    arr = np.asarray(img, dtype=input_dtype)
    # add batch dimension
    return np.expand_dims(arr, axis=0)
def preprocess_image(path: str, input_shape, input_dtype):
    """
    Load an image, resize it, and return a batched array in the correct
    channel order (NCHW vs NHWC) for the ONNX session.
    """
    # Replace any dynamic dims (None) in the shape with 1
    dims = [d if isinstance(d, int) else 1 for d in input_shape]  
    # Determine channel ordering:
    #  - if dims[1] is 1 or 3 → channels-first  (N,C,H,W)
    #  - otherwise           → channels-last   (N,H,W,C)
    nchw = (dims[1] in (1,3))

    if nchw:
        _, C, H, W = dims
        img = Image.open(path).convert('RGB').resize((W, H))
        arr = np.asarray(img, dtype=input_dtype)           # (H,W,3)
        arr = arr.transpose(2, 0, 1)                        # → (3,H,W)
    else:
        _, H, W, C = dims
        img = Image.open(path).convert('RGB').resize((W, H))
        arr = np.asarray(img, dtype=input_dtype)           # (H,W,3)

    # add batch dimension
    return np.expand_dims(arr, axis=0)  # → (1,C,H,W) or (1,H,W,C)

def run_eval(model_path: str, data_root: str):
    # --- 1) Create ONNX Runtime session with VSI NPU EP ---
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        model_path,
        sess_options=sess_opts,
        providers=['VSINPUExecutionProvider']
    )

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

        for cls_idx, cls_name in enumerate(labels):
            folder = os.path.join(data_root, split, cls_name)
            if not os.path.isdir(folder):
                continue

            for fn in os.listdir(folder):
                img_path = os.path.join(folder, fn)
                # Preprocess
                x = preprocess_image(img_path, input_shape, input_dtype)
                
                # Inference + timing
                t0 = timer()
                # session.run returns a list of outputs; here we expect one scalar array
                out_arr = session.run([out.name], {input_name: x})[0]
                t1 = timer()
                latencies.append((t1 - t0) * 1000.0)

                # Assume out_arr shape is (1,1) or (1,) giving a score in [0,1]
                print(out_arr)
                score = float(np.squeeze(out_arr))
                pred = 1 if score > 0.5 else 0

                total += 1
                if pred == cls_idx:
                    correct += 1
                    file_lbls.append(1)
                else:
                    file_lbls.append(0)

                # store confidence (probability of predicted class)
                conf = score if pred == 1 else 1.0 - score
                file_confs.append(conf)

        # 4) Report
        acc = correct / total if total else 0.0
        print(f"Evaluated {total} images in {split}")
        print(f"SPECIAL_PRINTacc{split} {acc * 100:.2f}")
        print(f"Accuracy : {correct}/{total} = {acc * 100:.2f}%")

        # 5) Dump to files
        with open(f"labels_{split}.txt", "w") as f_lbl:
            for v in file_lbls:
                f_lbl.write(f"{v}\n")
        with open(f"confs_{split}.txt", "w") as f_conf:
            for v in file_confs:
                f_conf.write(f"{v:.6f}\n")

    # 6) Overall latency
    print(f"Avg latency: {np.mean(latencies):.1f} ms  (over {len(latencies)} inferences)")

if __name__ == "__main__":
    run_eval('../../../models/temporal_optimization_model_st/temporal_optimization_model.onnx', '../../../data/processed/IDID_cropped_224')
