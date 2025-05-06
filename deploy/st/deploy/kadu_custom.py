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

def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_mean', default=127.5, help='input_mean')
    parser.add_argument('--input_std', default=127.5,help='input stddev')
    args = parser.parse_args()

    model_file = '../../../models/temporal_optimization_model_st/temporal_optimization_model.nb'
    image = '../../../data/processed/IDID_cropped_224/test/broken/'
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
    for i in range(10):
        # Reading input image
        input_width = input_tensor_shape[1]
        input_height = input_tensor_shape[2]
        input_image = Image.open(image).resize((input_width,input_height))
        input_data = np.expand_dims(input_image, axis=0)
        if input_tensor_dtype == np.float32:
            input_data = (np.float32(input_data) - args.input_mean) /args.input_std

        stai_model.set_input(0, input_data)
        start = timer()
        stai_model.run()
        end = timer()

        print("Inference time: ", (end - start) *1000, "ms")
        output_data = stai_model.get_output(index=0)
        results = np.squeeze(output_data)
        top_k = results.argsort()[-5:][::-1]
        labels = load_labels(label_file)
        for i in top_k:
            if output_tensor_dtype == np.uint8:
                print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))
            else:
                print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
