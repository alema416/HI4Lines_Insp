from stai_mpu import stai_mpu_network
from numpy.typing import NDArray
from typing import Any, List
from pathlib import Path
from PIL import Image
from argparse import ArgumentParser
from timeit import default_timer as timer
import cv2 as cv
import numpy as np
import os
import time
from hydra import initialize, compose

def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

if __name__ == '__main__':
    with initialize(config_path="../../configs/"):
        cfg = compose(config_name="hw_eval_server_st")  # exp1.yaml with defaults key

    parser = ArgumentParser()
    parser.add_argument('-m','--model_file', help='model to be executed.')
    parser.add_argument('-i','--image', default='../../../data/processed/IDID_cropped_224/', help='image path to be classified.')
    parser.add_argument('-l','--label_file', default='./labels_idid_cropped.txt', help='name of labels file.')
    parser.add_argument('--input_mean', default=127.5, help='input_mean')
    parser.add_argument('--input_std', default=127.5,help='input stddev')
    args = parser.parse_args()
    
    times = []
    total = 0
    wrong = 0
    model_path = cfg.server.upload_dir
    stai_model = stai_mpu_network(model_path=os.path.join(model_path, args.model_file), use_hw_acceleration=True)
    # Read input tensor information
    num_inputs = stai_model.get_num_inputs()
    input_tensor_infos = stai_model.get_input_infos()
    
    for i in range(0, num_inputs):
        input_tensor_shape = input_tensor_infos[i].get_shape()
        input_tensor_name = input_tensor_infos[i].get_name()
        input_tensor_rank = input_tensor_infos[i].get_rank()
        input_tensor_dtype = input_tensor_infos[i].get_dtype()
        print("**Input node: {} -Input_name:{} -Input_dims:{} - input_type:{} -Input_shape:{}".format(i, input_tensor_name,
                                                                                                    input_tensor_rank,
                                                                                                    input_tensor_dtype,
                                                                                                    input_tensor_shape))
        if input_tensor_infos[i].get_qtype() == "staticAffine":
            # Reading the input scale and zero point variables
            input_tensor_scale = input_tensor_infos[i].get_scale()
            input_tensor_zp = input_tensor_infos[i].get_zero_point()
        if input_tensor_infos[i].get_qtype() == "dynamicFixedPoint":
            # Reading the dynamic fixed point position
            input_tensor_dfp_pos = input_tensor_infos[i].get_fixed_point_pos()


    # Read output tensor information
    num_outputs = stai_model.get_num_outputs()
    output_tensor_infos = stai_model.get_output_infos()
    for i in range(0, num_outputs):
        output_tensor_shape = output_tensor_infos[i].get_shape()
        output_tensor_name = output_tensor_infos[i].get_name()
        output_tensor_rank = output_tensor_infos[i].get_rank()
        output_tensor_dtype = output_tensor_infos[i].get_dtype()
        print("**Output node: {} -Output_name:{} -Output_dims:{} -  Output_type:{} -Output_shape:{}".format(i, output_tensor_name,
                                                                                                        output_tensor_rank,
                                                                                                        output_tensor_dtype,
                                                                                                        output_tensor_shape))
        if output_tensor_infos[i].get_qtype() == "staticAffine":
            # Reading the output scale and zero point variables
            output_tensor_scale = output_tensor_infos[i].get_scale()
            output_tensor_zp = output_tensor_infos[i].get_zero_point()
        if output_tensor_infos[i].get_qtype() == "dynamicFixedPoint":
            # Reading the dynamic fixed point position
            output_tensor_dfp_pos = output_tensor_infos[i].get_fixed_point_pos()

    # Reading input image
    input_width = input_tensor_shape[1]
    input_height = input_tensor_shape[2]
    for split in ['train', 'val', 'test']: 
        img_dir = os.path.join(args.image, split)
        labels = load_labels(args.label_file)

        for img in os.listdir(img_dir):

            img = os.path.join(img_dir, img)
            print(img)
            input_image = Image.open(img).resize((input_width,input_height))
            input_data = np.expand_dims(input_image, axis=0)
            if input_tensor_dtype == np.float32:
                input_data = (np.float32(input_data) - args.input_mean) /args.input_std

            stai_model.set_input(0, input_data)
            start = timer()
            stai_model.run()
            end = timer()
            times.append(end-start)
            
            #print("Inference time: ", (end - start) *1000, "ms")
            output_data = stai_model.get_output(index=0)
            results = np.squeeze(output_data)
            print(results)
            q_out = stai_model.get_output(index=0)         # shape (1,2) or (1,N), dtype=int8/int16
            q_logits = np.squeeze(q_out).astype(np.float32)  # e.g. [ -82., 127. ]
            logits = q_logits * output_tensor_scale                        # dequantized float  logits

            # — Softmax to get probabilities —
            probs = tf.nn.softmax(logits).numpy()            # e.g. [0.0000001, 0.9999999]
            pred_idx = np.argmax(probs)
            pred_label = labels[pred_idx]
            pred_conf = probs[pred_idx]

            print(f"Logits: {logits}")
            print(f"Probs:  {probs}")
            print(f"→ Prediction: {pred_label} ({pred_conf:.4f})\n")
            '''
            # single-output (binary) case
            score_pos = float(results)
            score_neg = 1.0 - score_pos
            pred_label = labels[1] if score_pos>=score_neg else labels[0]
            pred_conf  = max(score_pos, score_neg)
            total += 1
            if pred_label != args.cl:
                wrong += 1
                pass #print(f"{pred_conf:0.6f}: {pred_label}")
            '''
            
            print(f'SPECIAL_PRINTacc{split} {accuracy}')