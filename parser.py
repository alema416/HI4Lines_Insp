# General imports used throughout the tutorial
import tensorflow as tf
from IPython.display import SVG
import subprocess
import argparse

# import the ClientRunner class from the hailo_sdk_client package
from hailo_sdk_client import ClientRunner

parser = argparse.ArgumentParser(description='Rethinking CC for FP')
parser.add_argument('--run_id', required=True, type=int, help='')
parser.add_argument('--pathh', required=True, type=str, help='')
args = parser.parse_args()

chosen_hw_arch = "hailo8"
model_name = f'model_{args.run_id}'

#onnx_model_name = input('name')
onnx_model_name = model_name #model_name.split('.')[0]
onnx_path = f'./{model_name}.onnx'

runner = ClientRunner(hw_arch=chosen_hw_arch)
hn, npz = runner.translate_onnx_model(
    onnx_path,
    onnx_model_name,
    net_input_shapes={"input.1": [1, 3, 224, 224]},
)

hailo_model_har_name = f"./{onnx_model_name}_hailo_model.har"
runner.save_har(hailo_model_har_name)

subprocess.run(['hailo', 'visualizer', f'{hailo_model_har_name}', '--no-browser'], check=True)
SVG(f"{onnx_model_name}.svg")