from stai_mpu import stai_mpu_network
from numpy.typing import NDArray
from typing import Any, List
from pathlib import Path
from PIL import Image
from argparse import ArgumentParser
from timeit import default_timer as timer
import cv2 as cv
import numpy as np
import time
import os

def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

if __name__ == '__main__':
    parser = ArgumentParser() # remove
    parser.add_argument('--input_mean', default=127.5, help='input_mean')
    parser.add_argument('--input_std', default=127.5,help='input stddev')
    args = parser.parse_args()

    model_file = '../../../models/temporal_optimization_model_st/temporal_optimization_model.nb' 
    image = os.path.join('../../../data/processed/IDID_cropped_224/test/broken/', os.listdir('../../../data/processed/IDID_cropped_224/test/broken/')[0])
    label_file = 'labels_idid.txt'
    
    stai_model = stai_mpu_network(model_path=model_file, use_hw_acceleration=True)
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
    for i in range(1):
        start = timer()

        # Reading input image
        #input_width = input_tensor_shape[1]
        #input_height = input_tensor_shape[2]
        
        _, C, H, W = input_tensor_shape
        # PIL wants (width, height)
        print(f'{W}x{H}')

        input_image = Image.open(image).resize((W, H))
        input_np = np.array(input_image)                  # HxWxC
        input_chw = input_np.transpose(2,0,1)             # CxHxW
        
        input_data = np.expand_dims(input_image, axis=0)
        '''
        _, C, H, W = input_tensor_shape

        # Read and preprocess ONCE:
        # — read as BGR, convert to RGB if your model expects RGB
        bgr = cv.imread(image, cv.IMREAD_COLOR)
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        # — resize to W×H
        resized = cv.resize(rgb, (W, H), interpolation=cv.INTER_LINEAR)

        # — quantize:
        #    ((pixel/std + mean) / scale) + zp
        #    note: args.input_mean/std are scalars
        quant = ((resized.astype(np.float32) - args.input_mean) / args.input_std)
        quant = quant / input_tensor_scale + input_tensor_zp
        quant = np.clip(np.round(quant), -128, 127).astype(np.int8)

        # — transpose to C×H×W and add batch
        input_chw = quant.transpose(2, 0, 1)        # from H×W×C → C×H×W
        input_data = input_chw[np.newaxis, :, :, :]  # shape (1, C, H, W)
        '''
        if input_tensor_dtype == np.float32:
            input_data = (np.float32(input_data) - args.input_mean) /args.input_std

        stai_model.set_input(0, input_data)
        stai_model.run()
        

        output_data = stai_model.get_output(index=0)
        results = np.squeeze(output_data)
        labels = load_labels(label_file)

        print('dequantized: ')
        real_scores = (results.astype(np.float32) - output_tensor_zp) * output_tensor_scale
        top_k = real_scores.argsort()[-5:][::-1]
        for idx in top_k:
            print(f"{real_scores[idx]:.6f}: {labels[idx]}")
        print('not dequantized: ')
        top_k = results.argsort()[-5:][::-1]
        for i in top_k:
            if output_tensor_dtype == np.uint8:
                print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))
            else:
                print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
        end = timer()
        print("Inference time: ", (end - start) *1000, "ms")
