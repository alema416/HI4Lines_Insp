import torch
from model import resnet18_custom
from model import resnet18
from collections import OrderedDict
import argparse
import os

parser = argparse.ArgumentParser(description='Rethinking CC for FP')
parser.add_argument('--run_id', required=True, type=int, help='')
parser.add_argument('--pathh', required=True, type=str, help='')
args = parser.parse_args()

device = torch.device("cpu")
num_class = 2
onnx_dir = './'
model_dict = {"num_classes": num_class}

ours = True

if ours:
  model = resnet18.ResNet18(num_classes=2).cuda()
else:
  model = resnet18_custom.resnet18().cuda()

model_name = f'model_{args.run_id}'
model_path = f'{args.pathh}/{args.run_id}/model.pth'

state_dict_fmfp = torch.load(model_path, map_location=device)

new_state_dict = OrderedDict()
for k, v in state_dict_fmfp.items():
    new_key = k.replace("module.", "") 
    if k == "n_averaged": 
        continue
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

inp = [torch.randn((1, 3, 224, 224), requires_grad=False)]

model(*inp)
onnx_path = os.path.join(onnx_dir, f'{model_name}.onnx')

torch.onnx.export(model, tuple(inp), onnx_path,
                  export_params=True,
training=torch.onnx.TrainingMode.PRESERVE,
do_constant_folding=False,
opset_version=13)
print('done')