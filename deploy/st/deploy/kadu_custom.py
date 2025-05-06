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

def preprocess_image(path: str,
                     width: int, height: int,
                     dtype, qtype: str,
                     scale=None, zp=None, dfp_pos=None,
                     mean: float=127.5, std: float=127.5):
    """
    Load, resize, normalize and quantize (if needed) into a batched array.
    """
    #img = Image.open(path).convert('RGB').resize((width, height))
      # shape (H, W, 3), dtype uint8
    img = cv2.imread(str(path))
    img = cv2.resize(img, (width, height))  # much faster than PIL
    arr = np.asarray(img)
    # float32 path
    if dtype == np.float32:
        arr = (arr.astype(np.float32) - mean) / std

    # staticAffine quantization:  val_q = round(val_f/scale) + zp
    elif qtype == "staticAffine":
        arr_f = (arr.astype(np.float32) - mean) / std
        arr = np.round(arr_f / scale + zp).astype(np.uint8)

    # dynamicFixedPoint: val_q = round(val_f * 2^dfp_pos)
    elif qtype == "dynamicFixedPoint":
        arr_f = (arr.astype(np.float32) - mean) / std
        arr = np.round(arr_f * (2 ** dfp_pos)).astype(dtype)

    else:
        # fallback cast
        arr = arr.astype(dtype)

    # add batch dim
    return np.expand_dims(arr, axis=0)

def run_eval(model_path: str, data_root: str, label_file: str,
             mean: float, std: float):
    # 1) Load model with HW acceleration
    stai_model = stai_mpu_network(model_path=model_path,
                                  use_hw_acceleration=True)

    # 2) Read input tensor info (we assume single-input)
    inp = stai_model.get_input_infos()[0]
    output_tensor_infos = stai_model.get_output_infos()
    shape = inp.get_shape()      # e.g. [1, 224, 224, 3]
    _, width, height, _ = shape
    dtype = inp.get_dtype()
    qtype = inp.get_qtype()

    # read quant params if needed
    scale = zp = dfp_pos = None
    if qtype == "staticAffine":
        scale = inp.get_scale()
        zp    = inp.get_zero_point()
    elif qtype == "dynamicFixedPoint":
        dfp_pos = inp.get_fixed_point_pos()

    # 3) Load labels & class names
    labels_list = load_labels(label_file)

    # 4) Loop over splits & classes
    for split in ['test']: #["train", "val", "test"]:
        total = correct = 0
        file_lbls = []
        file_confs = []

        for cls_idx, cls_name in enumerate(labels_list):
            folder = Path(data_root) / split / cls_name
            if not folder.is_dir():
                continue

            for img_path in folder.iterdir():
                # preprocess
                print(img_path)
                x = preprocess_image(
                    str(img_path),
                    width, height,
                    dtype, qtype,
                    scale, zp, dfp_pos,
                    mean, std
                )

                # inference + timing
                stai_model.set_input(0, x)
                t0 = timer()
                stai_model.run()
                t1 = timer()

                # record latency (ms)
                latency_ms = (t1 - t0) * 1000.0
                # get and interpret output
                out_qtype = output_tensor_infos[0].get_qtype()          # "staticAffine" or "dynamicFixedPoint" or ""
                if out_qtype == "staticAffine":
                    out_scale = output_tensor_infos[0].get_scale()
                    out_zp    = output_tensor_infos[0].get_zero_point()
                elif out_qtype == "dynamicFixedPoint":
                    out_dfp   = output_tensor_infos[0].get_fixed_point_pos()

                raw = stai_model.get_output(0)


                if out_qtype == "staticAffine":
                    # float_value = (quant_value - zero_point) * scale
                    arr_f = (raw.astype(np.float32) - out_zp) * out_scale

                elif out_qtype == "dynamicFixedPoint":
                    # float_value = quant_value / 2^dfp_pos
                    arr_f = raw.astype(np.float32) / (2 ** out_dfp)

                else:
                    arr_f = raw.astype(np.float32)

                scores = np.squeeze(arr_f)   # now in real [0,1] or logits
                if scores.ndim == 0:
                    pred = 1 if scores > 0.5 else 0
                    conf = float(scores) if pred == 1 else float(1.0 - scores)
                else:
                    pred = int(np.argmax(scores))
                    conf = float(scores[pred])
                '''
                arr = np.squeeze(out)

                if arr.ndim == 0:
                    # single score
                    score = float(arr)
                    pred  = 1 if score > 0.5 else 0
                    conf  = score if pred == 1 else 1.0 - score
                else:
                    # multi-class
                    pred = int(np.argmax(arr))
                    conf = float(arr[pred])
                '''
                print(f'{pred} {conf} {latency_ms}')
                total += 1
                if pred == cls_idx:
                    correct += 1
                    file_lbls.append(1)
                else:
                    file_lbls.append(0)
                file_confs.append(conf)

        # 5) report + dump
        acc = correct / total if total else 0.0
        print(f"Evaluated {total} images in {split}")
        print(f"SPECIAL_PRINTacc{split} {acc*100:.2f}")
        print(f"Accuracy : {correct}/{total} = {acc*100:.2f}%")

        with open(f"labels_{split}.txt", "w") as f:
            for v in file_lbls:
                f.write(f"{v}\n")

        with open(f"confs_{split}.txt", "w") as f:
            for v in file_confs:
                f.write(f"{v:.6f}\n")

if __name__ == "__main__":
    run_eval('../../../models/temporal_optimization_model_st/temporal_optimization_model.nb', '../../../data/processed/IDID_cropped_224', './labels_idid.txt', 127.5, 127.5)