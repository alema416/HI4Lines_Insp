import torch
from torchvision.models import resnet18
from model import resnet18_custom
#from model import resnet18
from collections import OrderedDict
import argparse
import os
from torchvision.models import mobilenet_v2
from hydra import initialize, compose

with initialize(config_path="../configs/"):
    cfg = compose(config_name="optimizer")  # exp1.yaml with defaults key

exp_name = cfg.optimizer.exp_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Rethinking CC for FP')
parser.add_argument('--run_id', required=True, type=int, help='')
args = parser.parse_args()

device = torch.device("cpu")
num_class = 2

onnx_dir = f'../models/{exp_name}/{args.run_id}'
model_dict = {"num_classes": num_class}

ours = True
model_name = f'model_{args.run_id}'

if ours:
  #model = resnet18.ResNet18(num_classes=2).to(device)
  model = resnet18(pretrained=False, num_classes=2)
  model_path = f'../models/{exp_name}/{args.run_id}/model_state_dict/model.pth'

else:
  model = mobilenet_v2(weights=None, num_classes=2)
  model_path = f'../models/{exp_name}/{args.run_id}/model_state_dict/model.pth'

#model_name = f'model_{args.run_id}'
#model_path = f'../models/{exp_name}/{args.run_id}/model_state_dict/model.pth'

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
