import torch
try:

    import pynvml
    pynvml.nvmlInit()
    gpu = True
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # if you only use GPU 0
    def get_gpu_temps(handle=gpu_handle):
        """Return (core_temp, mem_temp or None)."""
        core = pynvml.nvmlDeviceGetTemperature(handle,
                                            pynvml.NVML_TEMPERATURE_GPU)
        # Try the named constant, else fall back to sensor ID 1, else None
        mem = None
        try:
            mem = pynvml.nvmlDeviceGetTemperature(handle,
                                                pynvml.NVML_TEMPERATURE_MEMORY)
        except AttributeError:
            try:
                mem = pynvml.nvmlDeviceGetTemperature(handle, 1)
            except Exception:
                # sensor not available
                mem = None
        return core, mem

    def wait_for_cooldown(handle=gpu_handle, *, thresh=85, cool_to=75, interval=5):
        """
        Pause if either core or memory temps exceed 'thresh',
        and wait until they drop below 'cool_to'.
        If mem is None, only watch core.
        """
        core, mem = get_gpu_temps(handle)
        # decide whether to pause
        over_core = core >= thresh
        over_mem  = (mem is not None and mem >= thresh)
        if not (over_core or over_mem):
            return

        print(f"[GPU COOL] core {core}°C{' and mem '+str(mem)+'°C' if mem is not None else ''} ≥ {thresh}°C; pausing…")
        # wait loop
        while True:
            time.sleep(interval)
            core, mem = get_gpu_temps(handle)
            msg = f"[GPU COOL] core {core}°C"
            if mem is not None:
                msg += f", mem {mem}°C"
            print(msg)
            if core <= cool_to and (mem is None or mem <= cool_to):
                break

        print("[GPU COOL] temperatures back below threshold; resuming.")
except Exception:
    gpu = False
    print("NVML init failed; continuing without GPU metrics")

import time
import gc
from torchvision.models import mobilenet_v2
from torchvision.models import resnet18
import distutils
from hi4lines_insp.validate_on_device import validate_on_device

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
try:
    from distutils.version import LooseVersion
except ImportError:
    # Define a custom LooseVersion using packaging.version
    from packaging.version import parse as parse_version

    class LooseVersion:
        def __init__(self, v):
            self.v = parse_version(v)
        def __lt__(self, other):
            if isinstance(other, LooseVersion):
                return self.v < other.v
            return NotImplemented
        def __gt__(self, other):
            if isinstance(other, LooseVersion):
                return self.v > other.v
            return NotImplemented
        def __eq__(self, other):
            if isinstance(other, LooseVersion):
                return self.v == other.v
            return NotImplemented
        def __repr__(self):
            return str(self.v)
    # Inject our custom LooseVersion into distutils.version
    distutils.version.LooseVersion = LooseVersion
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from scipy.spatial import distance
from scipy.stats import chi2
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.preprocessing import normalize
import argparse
import os
import csv
#import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import resource

from collections import OrderedDict
import requests
#from model import resnet18
from hi4lines_insp.utils import data as dataset
from hi4lines_insp.utils import crl_utils
from hi4lines_insp.utils import metrics
from hi4lines_insp.utils import utils
import hi4lines_insp.train_base
import hi4lines_insp.custom_data as custom_data
from hydra import initialize, compose

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

from dotenv import load_dotenv
import boto3
from botocore.client import Config

load_dotenv()  # reads .env into os.environ

AWS_ACCESS_KEY_ID     = os.getenv("S3_ACCESS_KEY", None)
AWS_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_KEY", None)
REGION_NAME           = os.getenv("S3_REGION",         None)
ENDPOINT_URL          = os.getenv("S3_ENDPOINT_URL",       None)
BUCKET                = os.getenv("S3_BUCKET", None)

# If you leave ENDPOINT_URL as empty string, boto3 will default to AWS.
s3_client = boto3.client(
    "s3",
    endpoint_url=ENDPOINT_URL or None,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=REGION_NAME,
    config=Config(signature_version="s3v4"),  # ensures compatibility
)

def upload_directory(local_path: str, s3_prefix: str):
    """
    Recursively upload all files under local_path to s3://BUCKET/s3_prefix/...
    """
    for root, _, files in os.walk(local_path):
        for fname in files:
            full_path = os.path.join(root, fname)
            print(full_path)
            rel_path = os.path.relpath(full_path, local_path)
            print(rel_path)
            s3_key   = f"{s3_prefix.rstrip('/')}/{rel_path}"
            print(f"Uploading {full_path} → s3://{BUCKET}/{s3_key}")
            s3_client.upload_file(full_path, BUCKET, s3_key)


def validate(loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for input, target, idx, _ in loader:
            input, target = input.to(device), target.long().to(device)
            output = model(input)
            loss = criterion(output, target)
            total_loss += loss.item() * input.size(0)
            _, predicted = torch.max(output, 1)
            total_correct += predicted.eq(target).sum().item()
            total_samples += input.size(0)
    avg_loss = total_loss / total_samples
    acc = 100.0 * total_correct / total_samples
    return avg_loss, acc

def csv_writter(path, dic, start):
    if os.path.isdir(path) == False: os.makedirs(path)
    os.chdir(path)
    # Write dic
    if start == 1:
        mode = 'w'
    else:
        mode = 'a'
    with open('logs.csv', mode) as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        if start == 1:
            writer.writerow(dic.keys())
        writer.writerow([elem["string"] for elem in dic.values()])
class Counter(dict):
    def __missing__(self, key):
        return None

def one_trial_train(trial_number, epochs, base_lr, custom_weight_decay, custom_momentum, cfg):
    device = cfg.training.device
    print(f"Using device: {device}")

    server1 = True
    server2 = not server1

    val_freq = cfg.training.validate_freq
    
    plot = cfg.training.print_freq
    rank_weight = cfg.training.rank_weight
    port = 5002 if server2 else 5000
    
    lr_strat = [80, 130, 170]
    lr_factor =  0.1

    save_path = cfg.training.save_path
    batch_size = cfg.training.batch_size

    classnumber = cfg.training.classnumber #2
    input_size = cfg.training.input_size #224
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.training.gpu
    cudnn.benchmark = True
    print(f'{trial_number}')
    save_path = os.path.join(save_path, f'{trial_number}')
    method = 'Baseline'
    modelname = cfg.training.model_name

    if not os.path.exists(save_path):
        os.makedirs(save_path, mode=0o777)
        os.makedirs(os.path.join(save_path, 'logs'), mode=0o777)
        os.makedirs(os.path.join(save_path, 'model_state_dict'), mode=0o777)
    
    writer = SummaryWriter(log_dir=save_path) #run_name

    dataset_path = cfg.training.data_path

    train_loader, valid_loader, test_loader = custom_data.get_loader_local(dataset_path, batch_size=batch_size, input_size=input_size)
    
    print(100*'#')
    
    model = mobilenet_v2(pretrained=False, num_classes=classnumber).to(device) if modelname == 'mobilenet' else resnet18(pretrained=False, num_classes=classnumber).to(device)

    cls_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=custom_momentum,
                                weight_decay=custom_weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=lr_strat, gamma=lr_factor)

    correctness_history = crl_utils.History(len(train_loader.dataset))
    ranking_criterion = nn.MarginRankingLoss(margin=0.0).to(device)
    args = None
    # start Train
    for epoch in range(1, epochs + 1):
        
        train_loss, train_acc = hi4lines_insp.train_base.train(train_loader,
                    model,
                    cls_criterion,
                    ranking_criterion,
                    optimizer,
                    epoch,
                    correctness_history,
                    plot, method, rank_weight, classnumber)
        scheduler.step()
        
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        if gpu:
            wait_for_cooldown(thresh=75, cool_to=65, interval=5)

        # save model
        if epoch == epochs:
            torch.save(model.state_dict(),
                        os.path.join(save_path, 'model_state_dict', 'model.pth'))
        # finish train

        # calc measure
        if epoch % val_freq == 0:
            val_loss, val_acc = validate(valid_loader, model, cls_criterion, device)
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('val_acc', val_acc, epoch)

            acc, auroc, aupr_success, aupr, fpr, tnr, aurc, eaurc, augrc = metrics.calc_metrics(args, valid_loader,
                                                                                model,
                                                                                cls_criterion, save_path, 'DELETE')
    epoch = epochs
    writer.add_scalar('params/lr', base_lr)
    writer.add_scalar('params/weight_decay', custom_weight_decay)
    writer.add_scalar('params/momentum', custom_momentum)
    writer.add_scalar('params/epochs', epochs)
    writer.add_scalar('params/batch_size', batch_size)

    torch.save(model.state_dict(), os.path.join(save_path, 'model_state_dict', 'model.pth'))
    
    acc, auroc, aupr_success, aupr, fpr, tnr, aurc, eaurc, augrc = metrics.calc_metrics(args, train_loader, model, cls_criterion, save_path, 'train') 
    writer.add_scalar('train_acc_final', auroc)
    writer.add_scalar('train_augrc_final', augrc)
    
    print(f'ckpt train acc: {acc}')
    print(f'ckpt train augrc: {augrc}')

    acc, auroc, aupr_success, aupr, fpr, tnr, aurc, val_eaurc, val_augrc = metrics.calc_metrics(args, valid_loader, model, cls_criterion, save_path, 'val')
    writer.add_scalar('val_acc_final', auroc)
    writer.add_scalar('val_augrc_final', val_augrc)
    
    print(f'ckpt val acc: {acc}')
    print(f'ckpt val augrc: {augrc}')


    acc, auroc, aupr_success, aupr, fpr, tnr, aurc, eaurc, augrc = metrics.calc_metrics(args, test_loader, model, cls_criterion, save_path, 'test')
    writer.add_scalar('test_acc_final', auroc)
    writer.add_scalar('test_augrc_final', augrc)

    print(f'ckpt test acc: {acc}')
    print(f'ckpt test augrc: {augrc}')
    
    result = validate_on_device('coral', cfg, trial_number)

    for key, val in result.items():
        if isinstance(val, (int, float)):
            writer.add_scalar(f'hw_metrics_{key}', val)
     
    gc.collect()
    if gpu:
        torch.cuda.synchronize()                     # finish all kernels
        torch.cuda.empty_cache()
        wait_for_cooldown(thresh=75, cool_to=65, interval=5)
    
    writer.close()
    upload_directory(save_path, f'trial_{str(trial_number)}')

def main():
    pass
if __name__ == "__main__":
    main()
