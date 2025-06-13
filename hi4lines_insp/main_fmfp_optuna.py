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
from torch.utils.tensorboard import SummaryWriter
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
'''
with initialize(config_path="../configs/"):
    cfg = compose(config_name="fmfp")  # exp1.yaml with defaults key

device = cfg.training.device #torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
'''

project_root = os.getcwd()            # now points to ~/HI4Lines_Insp/fresh_study
mlruns_dir   = os.path.join(project_root, 'fresh', "mlruns")
os.makedirs(mlruns_dir, exist_ok=True)

cfg = None

def objective(trial):
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
    epochs = trial.suggest_int("epochs", cfg.training.epochs_low, cfg.training.epochs_high)
    val_freq = cfg.training.validate_freq
    plot = cfg.training.print_freq
    batch_size = cfg.training.batch_size 
    port = 5000 #5001 if server2 else 5000
    save_path = cfg.training.save_path

    base_lr = trial.suggest_loguniform('lr', cfg.training.base_lr_low, cfg.training.base_lr_high)
    print(f'base_lr: {base_lr}')
    swa_start = trial.suggest_int("swa_start", int(epochs/2), int((3/4)*epochs))
    custom_weight_decay = trial.suggest_loguniform('weight_decay', cfg.training.weight_decay_low, cfg.training.weight_decay_high) 
    custom_momentum = trial.suggest_uniform('momentum', cfg.training.momentum_low, cfg.training.momentum_high) 
    swa_lr = trial.suggest_loguniform('swa_lr', cfg.fmfp.swa_lr_low, cfg.fmfp.swa_lr_high) 
    print(f'swa_start: {swa_start}')
    method = 'fmfp' 

    input_size = cfg.training.input_size
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.training.gpu
    cudnn.benchmark = True

    save_path = os.path.join(save_path, f'{trial.number}')
    RUN_ID = trial.number
    modelname = cfg.training.model_name
    run_name = f'trial_{trial.number}'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, mode=0o777)
        os.makedirs(os.path.join(save_path, 'model_state_dict'), mode=0o777)
        os.makedirs(os.path.join(save_path, 'logs'), mode=0o777)
    writer = SummaryWriter(log_dir=save_path) #run_name

    chmod_recursive_777(save_path)
        
    dataset_path = cfg.training.data_path
    train_loader, valid_loader, test_loader = custom_data.get_loader_local(dataset_path, batch_size=batch_size, input_size=cfg.training.input_size)
    
    num_class = cfg.training.classnumber  
    model_dict = { "num_classes": num_class, 'weights': 'MobileNet_V2_Weights'}
    model_dict1 = { "num_classes": num_class}

    print(100 * '#')
    print(f'{modelname}!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    resss = True
    if resss:
        model = resnet18(pretrained=False, num_classes=2).to(device)
    else:
        model = mobilenet_v2(pretrained=False, num_classes=2).to(device)

    drop = False
    if drop:
        drop_p_stage = 0.2
        drop_p_fc    = 0.4

        # 1) Wrap each stage (layer1–4) with a Dropout2d
        for name, module in list(model.named_children()):
            if name.startswith('layer'):
                wrapped = nn.Sequential(
                    module,
                    nn.Dropout2d(p=drop_p_stage)
                )
                setattr(model, name, wrapped)

            # 2) Wrap the avgpool output
        model.avgpool = nn.Sequential(
            model.avgpool,
            nn.Dropout(p=drop_p_stage)
        )
        old_fc = model.fc
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),    # drop 40% of activations
            old_fc
        ).to(device)
    #model = resnet18.ResNet18(**model_dict1).to(device)
    #model = mobilenet_v2(pretrained=False, num_classes=2).to(device)
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
                                                                
            writer.add_scalar('val_augrc', augrc, epoch)
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
    writer.add_scalar('params/model_name', modelname)
    swa_model.load_state_dict(torch.load(os.path.join(save_path, 'model_state_dict', 'best_model_runner.pth'), map_location=device))
    torch.optim.swa_utils.update_bn(train_loader, swa_model.cpu())
    model = swa_model.to(device)
    torch.save(model.state_dict(), os.path.join(save_path, 'model_state_dict', 'model.pth'))
    '''
    swa_model = swa_model.to(device)
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
    swa_model.eval()
    torch.save(swa_model.state_dict(), os.path.join(save_path, 'model_state_dict', 'swa_model.pth'))
    '''
    acc, auroc, aupr_success, aupr, fpr, tnr, aurc, eaurc, augrc = metrics.calc_metrics(args, train_loader, model, cls_criterion, save_path, 'train') 
    
    writer.add_scalar('train_acc', auroc, epoch)
    writer.add_scalar('train_auroc', auroc, epoch)
    writer.add_scalar('train_aupr_success', auroc, epoch)
    writer.add_scalar('train_aupr', aupr, epoch)
    writer.add_scalar('train_aurc', aurc, epoch)
    writer.add_scalar('train_eaurc', eaurc, epoch)
    writer.add_scalar('train_augrc', augrc, epoch)
    writer.add_scalar('params/run_id', RUN_ID)
    print(f'ckpt train acc: {acc}')
    print(f'ckpt train augrc: {augrc}')
    acc, auroc, aupr_success, aupr, fpr, tnr, aurc, val_eaurc, val_augrc = metrics.calc_metrics(args, valid_loader, model, cls_criterion, save_path, 'val')
    
    writer.add_scalar('val_acc', auroc, epoch)
    writer.add_scalar('val_auroc', auroc, epoch)
    writer.add_scalar('val_aupr_success', auroc, epoch)
    writer.add_scalar('val_aupr', aupr, epoch)
    writer.add_scalar('val_aurc', aurc, epoch)
    writer.add_scalar('val_eaurc', val_eaurc, epoch)
    writer.add_scalar('val_augrc', val_augrc, epoch)
    print(f'ckpt val acc: {acc}')
    print(f'ckpt val augrc: {augrc}')
    acc, auroc, aupr_success, aupr, fpr, tnr, aurc, eaurc, augrc = metrics.calc_metrics(args, test_loader, model, cls_criterion, save_path, 'test')
    
    writer.add_scalar('test_acc', auroc, epoch)
    writer.add_scalar('test_auroc', auroc, epoch)
    writer.add_scalar('test_aupr_success', auroc, epoch)
    writer.add_scalar('test_aupr', aupr, epoch)
    writer.add_scalar('test_aurc', aurc, epoch)
    writer.add_scalar('test_eaurc', eaurc, epoch)
    writer.add_scalar('test_augrc', augrc, epoch)
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
    
    augrc_hw_val = result.get("augrc_hw_val")
    '''
    '''
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
    
    augrc_hw_val = result.get("augrc_hw_val")
    '''
    # ORCA CODE
    ccc = 0
    stmz = cfg.training.ds_device_ip_orca
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
            writer.add_scalar(key, val)
    
    augrc_hw_val = result.get("augrc_hw_val")

    gc.collect()
    del model, optimizer, scheduler, ranking_criterion   # etc.
    torch.cuda.synchronize()                     # finish all kernels
    torch.cuda.empty_cache()                     # drop cached allocations
    #torch.cuda.empty_cache()
    wait_for_cooldown(thresh=75, cool_to=65, interval=5)
    writer.close()
    return float(augrc_hw_val)
    

def main():
    global cfg
    with initialize(config_path="../configs/"):
        cfg = compose(config_name="fmfp")  # exp1.yaml with defaults key

    device = cfg.training.device #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #mp.set_start_method("forkserver", force=True)
    study_name = cfg.training.study_name #input('study_name: ')
    storage = f"sqlite:///{os.path.join(os.getcwd(), 'fresh_study_db.sqlite')}"
    study = optuna.create_study(direction='minimize', load_if_exists=True, study_name = study_name, storage=storage)
    print(f"Sampler is {study.sampler.__class__.__name__}")
    study.optimize(objective, n_trials=1, n_jobs=1)

    print("Best hyperparameters:", study.best_params)
    print("Best accuracy:", study.best_value)

if __name__ == "__main__":
    main()
