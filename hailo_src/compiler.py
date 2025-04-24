from hailo_sdk_client import ClientRunner
import argparse
import subprocess
import os
from hydra import initialize, compose

with initialize(config_path="../configs/"):
    cfg = compose(config_name="optimizer")  # exp1.yaml with defaults key

exp_name = cfg.optimizer.exp_name

parser = argparse.ArgumentParser(description='Rethinking CC for FP')
parser.add_argument('--run_id', required=True, type=int, help='')
args = parser.parse_args()

hef_dir = f'../models/{exp_name}/{args.run_id}'
quantized_har_dir = f'../models/{exp_name}/{args.run_id}'

model_name = f'model_{args.run_id}'
quantized_model_har_path = os.path.join(quantized_har_dir, f"{model_name}_quantized_model.har")

runner = ClientRunner(har=quantized_model_har_path)

hef = runner.compile()

file_name = f"{model_name}.hef"
file_name = os.path.join(hef_dir, file_name)
with open(file_name, "wb") as f:
    f.write(hef)
#subprocess.run(['hailo', 'profiler', f'{quantized_model_har_path}'], check=True)