#!/usr/bin/env python3
# usage: python3 kadu.py /usr/local/bin/tflite-vx-delegate-example/model.tflite ../../../data/processed/IDID_cropped_224
import argparse, os
from timeit import default_timer as timer
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflr
from tqdm import tqdm

def preprocess_image(path, input_details):
    # load, resize, RGB → uint8 array
    _, height, width, _ = input_details['shape']
    img = Image.open(path).convert('RGB').resize((width, height))
    arr = np.asarray(img, dtype=input_details['dtype'])
    # add batch dim
    return np.expand_dims(arr, axis=0)

def run_eval(model_path, data_root):
    # 1) load TFLite + delegate
    delegate = tflr.load_delegate('/usr/lib/libvx_delegate.so.2')
    interp   = tflr.Interpreter(
                  model_path=model_path,
                  experimental_delegates=[delegate],
                  num_threads=os.cpu_count()
               )
    interp.allocate_tensors()

    # 2) pull I/O details
    inp_det  = interp.get_input_details()[0]
    out_det  = interp.get_output_details()[0]
    in_idx   = inp_det['index']
    out_idx  = out_det['index']

    # 3) metrics
    labels  = ['broken','healthy']
    total   = correct = 0
    latencies = []

    # 4) loop over dataset
    for split in ['test']:
      for cls in labels:
        true_idx = labels.index(cls)
        print(f'true_idx: {true_idx}')
        folder   = os.path.join(data_root, split, cls)
        print(f'folder: {folder}')
        for fn in os.listdir(folder): #tqdm(os.listdir(folder)):
          img_path = os.path.join(folder, fn)
          inp      = preprocess_image(img_path, inp_det)

          interp.set_tensor(in_idx, inp)
          t0 = timer()
          interp.invoke()
          t1 = timer()
          latencies.append((t1 - t0) * 1000)  # ms

          out = interp.get_tensor(out_idx).squeeze()  # shape (1,) → float
          print(f'out: {out}')
          pred = int(out > 0.5)                      # threshold

          total += 1
          if pred == true_idx:
            correct += 1

    # 5) report
    acc = correct/total if total else 0
    print(f"Evaluated {total} images")
    print(f"Accuracy : {correct}/{total} = {acc*100:.2f}%")
    print(f"Avg latency: {np.mean(latencies):.1f} ms")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument('model_path', help='.tflite file')
    p.add_argument('data_root',
                   help='root folder with train/val/test subfolders')
    args = p.parse_args()

    run_eval(args.model_path, args.data_root)
