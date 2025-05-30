#!/usr/bin/env python3
import os
from pathlib import Path
from timeit import default_timer as timer
from argparse import ArgumentParser

from PIL import Image
import numpy as np
import cv2
from stai_mpu import stai_mpu_network

def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f]

def preprocess(path, width, height, dtype, mean, std):
    # load & resize via OpenCV
    img = cv2.imread(str(path))
    img = cv2.resize(img, (width, height))
    arr = img.astype(np.float32)
    # only float path supported in your model
    if dtype == np.float32:
        arr = (arr - mean) / std
    # add batch dim
    return np.expand_dims(arr, axis=0)

def run_eval(model_nb: str, data_root: str, labels_txt: str, mean: float, std: float):
    # 1) load & warm up
    stai = stai_mpu_network(model_path=model_nb, use_hw_acceleration=True)
    print("OVX accel plugin loaded, running on NPU")
    inp_info = stai.get_input_infos()[0]
    out_info = stai.get_output_infos()[0]

    # shapes + types
    _, H, W, _ = inp_info.get_shape()
    in_dtype = inp_info.get_dtype()
    # your example model is float16 output
    out_dtype = out_info.get_dtype()

    # dummy warm-up
    dummy = np.zeros((1,H,W,3), dtype=np.float32)
    stai.set_input(0, dummy)
    for _ in range(5):
        stai.run()

    # 2) load labels (class names)
    classes = load_labels(labels_txt)

    # 3) iterate splits
    for split in ("train","val","test"):
        total = correct = 0
        lbls = []
        confs = []
        root = Path(data_root)/split

        for cls_idx, cls_name in enumerate(classes):
            folder = root/cls_name
            if not folder.exists(): continue

            for img_path in folder.iterdir():
                # preprocess
                x = preprocess(img_path, W, H, in_dtype, mean, std)

                # inference + timing
                stai.set_input(0, x)
                t0 = timer()
                stai.run()
                t1 = timer()
                latency = (t1-t0)*1000.0

                # get raw output (float16→float32)
                out_raw = stai.get_output(0).astype(np.float32)
                scores = np.squeeze(out_raw)

                # for binary two-class, assume scores[1] is P(class1)
                if scores.ndim>0 and len(scores)==2:
                    prob = float(scores[1])
                    pred = 1 if prob>0.5 else 0
                    conf = prob
                else:
                    # general multiclass
                    ex = np.exp(scores - scores.max())
                    sm = ex/ex.sum()
                    pred = int(np.argmax(sm))
                    conf = float(sm[pred])

                # accumulate
                total += 1
                correct += (pred==cls_idx)
                lbls.append(1 if pred==cls_idx else 0)
                confs.append(conf)

                print(f"{img_path} → pred={pred}, conf={conf:.3f}, lat={latency:.1f}ms")

        # report + dump
        acc = 100*correct/total if total else 0
        print(f"Evaluated {total} images in {split}")
        print(f"SPECIAL_PRINTacc{split} {acc:.2f}")
        print(f"Accuracy : {correct}/{total} = {acc:.2f}%")

        with open(f"labels_{split}.txt","w") as f:
            for v in lbls: f.write(f"{v}\n")
        with open(f"confs_{split}.txt","w") as f:
            for v in confs: f.write(f"{v:.6f}\n")

if __name__=="__main__":

    run_eval('../../../models/temporal_optimization_model_st/temporal_optimization_model.nb', '../../../data/processed/IDID_cropped_224', 'labels_idid.txt',
             127.5, 127.5)
