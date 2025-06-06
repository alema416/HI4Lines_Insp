import torch
# Patch for distutils.version.LooseVersion if missing
import pynvml
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # if you only use GPU 0
import time
import multiprocessing as mp
import distutils
import requests
import base64
from torchvision.models import resnet18
import gc
import threading
from torchvision.models import mobilenet_v2
from torchvision.models import efficientnet_b0
from model.custom_mob import build_model

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
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
from scipy.spatial import distance
from scipy.stats import chi2
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.preprocessing import normalize
import argparse
import os
import optuna
import csv
import math
import pandas as pd
import numpy as np
import resource
from collections import OrderedDict
from model import resnet
from model import mobilenet
from model import resnet18_custom
#from model import resnet18
from utils import data as dataset
from utils import crl_utils
from utils import metrics
from utils import utils
import train_fmfp
import custom_data as custom_data
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.sam import SAM
from hydra import initialize, compose
'''
def get_gpu_temps(handle=gpu_handle):
    core = pynvml.nvmlDeviceGetTemperature(
              handle, pynvml.NVML_TEMPERATURE_GPU)
    return core, None

def get_gpu_temps(handle=gpu_handle):
    # always works:
    core = pynvml.nvmlDeviceGetTemperature(handle,
                                           pynvml.NVML_TEMPERATURE_GPU)
    # memory sensor is ID 1 if the binding constant is missing
    try:
        mem = pynvml.nvmlDeviceGetTemperature(handle,
                                              pynvml.NVML_TEMPERATURE_MEMORY)
    except AttributeError:
        mem = pynvml.nvmlDeviceGetTemperature(handle, 1)
    return core, mem

def get_gpu_temps(handle=gpu_handle):
    """Returns (core_temp, memory_temp) in °C"""
    core = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    mem  = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_MEMORY)
    return core, mem

def wait_for_cooldown(handle=gpu_handle, thresh=85, cool_to=75, interval=10):
    """
    If memory temp ≥ thresh, sleep in a loop until it drops below cool_to.
    """
    _, mem = get_gpu_temps(handle)
    if int(mem) < int(thresh):
        return
    print(f"[GPU COOLER] memory temp {mem}°C ≥ {thresh}°C; pausing training…")
    while mem > cool_to:
        time.sleep(interval)
        _, mem = get_gpu_temps(handle)
        print(f"[GPU COOLER] memory temp now {mem}°C; waiting until ≤ {cool_to}°C")
    print("[GPU COOLER] OK, resuming training.")
'''

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


def validate(loader, model, criterion):
    device = 'cuda'
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
def chmod_recursive_777(path):
    for root, dirs, files in os.walk(path):
        os.chmod(root, 0o777)
        for fname in files:
            os.chmod(os.path.join(root, fname), 0o777)

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

cfg = None

def objective(trial_number, epochs, base_lr, custom_weight_decay, custom_momentum, swa_start, swa_lr):
    global cfg
    if cfg is None:
        from hydra import initialize, compose
        with initialize(config_path="../configs/", version_base="1.1"):
            cfg = compose(config_name="fmfp")
        device = cfg.training.device
    #with initialize(config_path="../configs/"):
    #    cfg = compose(config_name="fmfp")  # exp1.yaml with defaults key

    device = cfg.training.device #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    server1 = True
    server2 = not server1
    
    val_freq = cfg.training.validate_freq
    plot = cfg.training.print_freq
    batch_size = cfg.training.batch_size 
    port = 5000 #5001 if server2 else 5000
    save_path = cfg.training.save_path
    classnumber = cfg.training.classnumber #2

    method = 'fmfp' 

    input_size = cfg.training.input_size
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.training.gpu
    cudnn.benchmark = True

    save_path = os.path.join(save_path, f'{trial_number}')
    RUN_ID = trial_number
    modelname = cfg.training.model_name
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, mode=0o777)
        os.makedirs(os.path.join(save_path, 'model_state_dict'), mode=0o777)
        os.makedirs(os.path.join(save_path, 'logs'), mode=0o777)
    
    chmod_recursive_777(save_path)
    writer = SummaryWriter(log_dir=save_path) #run_name

    dataset_path = cfg.training.data_path
    train_loader, valid_loader, test_loader = custom_data.get_loader_local(dataset_path, batch_size=batch_size, input_size=input_size)

    print(100 * '#')
    modelname = cfg.training.model_name
    model = mobilenet_v2(pretrained=False, num_classes=classnumber).to(device) if modelname == 'mobilenet' else resnet18(pretrained=False, num_classes=classnumber).to(device)

    drop = False
    if drop:
        old_fc = model.fc
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),    # drop 40% of activations
            old_fc
        ).to(device)

    cls_criterion = nn.CrossEntropyLoss().to(device)

    correctness_history = crl_utils.History(len(train_loader.dataset))
    ranking_criterion = nn.MarginRankingLoss(margin=0.0).to(device)

    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=base_lr, momentum=custom_momentum, weight_decay=custom_weight_decay)
    
    swa_model = AveragedModel(model).to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=swa_start, eta_min=1e-5)
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
    
    args = None
    
    best_val_loss = float('inf')
    patience = 10
    num_bad_epochs = 0
    last_ep = 0
    ac_ep = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc  = train_fmfp.train(RUN_ID, train_loader, \
                                                model, cls_criterion, ranking_criterion, optimizer, epoch, correctness_history, plot, method)
        
        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
                
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        wait_for_cooldown(thresh=80, cool_to=75, interval=5)
        # save model
        '''
        if epoch == epochs:
            torch.save(model.state_dict(), os.path.join(save_path, 'model_state_dict', 'model.pth'))
        '''
        # calc measure
        if epoch % val_freq == 0:

            if epoch > swa_start:      
                val_loss, val_acc = validate(valid_loader, model, cls_criterion)
                acc, auroc, aupr_success, aupr, fpr, tnr, aurc, eaurc, augrc = metrics.calc_metrics(args, valid_loader,
                                                                                model,cls_criterion, save_path, 'DELETE')

                if val_loss < best_val_loss - 1e-4:    # a tiny delta to avoid float noise
                    best_val_loss = val_loss
                    num_bad_epochs = 0
                    torch.save(swa_model.state_dict(), os.path.join(save_path, 'model_state_dict', 'best_model_runner.pth'))
                else:
                    num_bad_epochs += 1
                    print(f"No improvement in val_loss for {num_bad_epochs}/{patience} checks.")
                    if num_bad_epochs >= patience:
                        print(f"Stopping early at epoch {epoch} (best_val_loss={best_val_loss:.4f}).")
                        last_ep = epoch
                        break
            else:
                val_loss, val_acc = validate(valid_loader, model, cls_criterion)
                acc, auroc, aupr_success, aupr, fpr, tnr, aurc, eaurc, augrc = metrics.calc_metrics(args, valid_loader,
                                                                                model,cls_criterion, save_path, 'DELETE')
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('val_acc', val_acc, epoch)
                                                                
            print(f'val loss: {val_loss}, val acc: {val_acc}, val augrc: {augrc}')
            print('Validation Loss: {0}\t'
                'Validation Acc: {1})\t'
                'Validation AUGRC: {2})\t'.format(val_loss, val_acc, augrc))
            ac_ep = epoch
            torch.cuda.empty_cache()
    if last_ep > ac_ep:
        epoch = last_ep
    else:
        epoch = ac_ep
    writer.add_scalar('params/lr', base_lr)
    writer.add_scalar('params/swa_start', swa_start)
    writer.add_scalar('params/weight_decay', custom_weight_decay)
    writer.add_scalar('params/momentum', custom_momentum)
    writer.add_scalar('params/swa_lr', swa_lr)
    writer.add_scalar('params/epochs', epochs)
    writer.add_scalar('params/batch_size', batch_size)

    swa_model.load_state_dict(torch.load(os.path.join(save_path, 'model_state_dict', 'best_model_runner.pth'), map_location=device))
    torch.optim.swa_utils.update_bn(train_loader, swa_model.cpu())
    model = swa_model.to(device)
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
    '''
    #RPI CODE
    ccc = 0
    hailo_ip = cfg.training.ds_device_ip
    while ccc < 10:
        try:
            response = requests.post(f"http://{hailo_ip}:{port}/validate", json={"run_id": RUN_ID})
            response.raise_for_status()
            break
        except requests.RequestException as e:
            print(e)
            print(f"================ERROR #{ccc}================")
            ccc += 1
            continue
    
    result = response.json()
    
    for key, val in result.items():
        if isinstance(val, (int, float)):
            writer.add_scalar(key, val)
    
    augrc_hw_val = result.get("augrc_hw_val")
    
    
    # ST CODE
    ccc = 0
    stmz = cfg.training.ds_device_ip_st
    while ccc < 10:
        try:
            with open(os.path.join(cfg.training.save_path, str(RUN_ID), 'model_state_dict', 'model.pth'), "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            payload = {
                "file": encoded,
                "run_id": str(RUN_ID)
            }
            headers = {"Content-Type": "application/json"}

# Send the request
            response = requests.post(f"http://{stmz}:{5002}/validate", json=payload, headers=headers)

            response.raise_for_status()
            break
        except requests.RequestException as e:
            print(e)
            print(f"================ERROR #{ccc}================")
            ccc += 1
            continue
    
    result = response.json()
    
    for key, val in result.items():
        if isinstance(val, (int, float)):
            writer.add_scalar(f'hw_metrics_{key}', val)
    
    augrc_hw_val = result.get("augrc_hw_val")
    '''
    gc.collect()
    del model, optimizer, scheduler, ranking_criterion   # etc.
    torch.cuda.synchronize()                     # finish all kernels
    torch.cuda.empty_cache()                     # drop cached allocations
    #torch.cuda.empty_cache()
    wait_for_cooldown(thresh=75, cool_to=65, interval=5)
    return float(1)
    

def main():
    global cfg
    with initialize(config_path="../configs/"):
        cfg = compose(config_name="fmfp")  # exp1.yaml with defaults key

    device = cfg.training.device #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    specs = [[0, 20, 0.01, 1e-6, 0.99], [1, 20, 0.2, 1e-2, 0.92], [2, 20, 0.01, 1e-2, 0.8], [3, 20, 0.001, 1e-4, 0.9]]
    
    for spec in specs:
        print(spec)
        objective(spec[0], spec[1], spec[2], spec[3], spec[4], 1, 0.1)

if __name__ == "__main__":
    main()
