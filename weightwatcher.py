import weightwatcher as ww
from model import resnet18
import os
import logging
import torch
import argparse
import numpy as np
import torch.nn as nn
import csv
from model import vgg
import matplotlib.pyplot as plt
from utils import data as dataset
from model import resnet18
import custom_data as custom_data
from collections import OrderedDict
import torchvision.transforms as transforms
from torchvision import models, datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = '/home/apel/machairas/FMFP/parameters_10e02/idid_cropped_resnet18_Baseline_200/model.pth'
model_dict = {"num_classes": 2}
model = resnet18.ResNet18(**model_dict).to(device)
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(ww.__name__)
logger.setLevel(logging.INFO)
watcher = ww.WeightWatcher(model=model)
details = watcher.analyze(plot=False)
summary = watcher.get_summary(details)
print(summary)
print(details)