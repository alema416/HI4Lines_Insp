from hailo_sdk_client import ClientRunner
import argparse
import subprocess

parser = argparse.ArgumentParser(description='Rethinking CC for FP')
parser.add_argument('--run_id', required=True, type=int, help='')
parser.add_argument('--pathh', required=True, type=str, help='')
args = parser.parse_args()

model_name = f'model_{args.run_id}'
quantized_model_har_path = f"{model_name}_quantized_model.har"

runner = ClientRunner(har=quantized_model_har_path)

hef = runner.compile()

file_name = f"{model_name}.hef"
with open(file_name, "wb") as f:
    f.write(hef)
subprocess.run(['hailo', 'profiler', f'{quantized_model_har_path}'], check=True)