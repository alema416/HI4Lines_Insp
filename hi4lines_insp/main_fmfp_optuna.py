import torch
# Patch for distutils.version.LooseVersion if missing
import distutils
import requests
import base64

import gc
import threading
from torchvision.models import mobilenet_v2



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
import mlflow
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
from model import resnet18
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

def validate(loader, model, criterion):
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

with initialize(config_path="../configs/"):
    cfg = compose(config_name="fmfp")  # exp1.yaml with defaults key

device = cfg.training.device #torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

mlflow.set_experiment(cfg.training.experiment_name if hasattr(cfg.training, 'experiment_name') else 'fmfp_experiment')

def objective(trial):
    print(cfg)
    server1 = True
    server2 = not server1
    epochs = cfg.training.epochs
    plot = cfg.training.validate_freq
    batch_size = cfg.training.batch_size 
    port = 5002 #5001 if server2 else 5000
    save_path = cfg.training.save_path

    base_lr = trial.suggest_loguniform('lr', cfg.training.base_lr_low, cfg.training.base_lr_high)
    swa_start = trial.suggest_int("swa_start", cfg.fmfp.swa_start_low, cfg.fmfp.swa_start_high)
    custom_weight_decay = trial.suggest_loguniform('weight_decay', cfg.training.weight_decay_low, cfg.training.weight_decay_high) 
    custom_momentum = trial.suggest_uniform('momentum', cfg.training.momentum_low, cfg.training.momentum_high) 
    swa_lr = trial.suggest_loguniform('swa_lr', cfg.fmfp.swa_lr_low, cfg.fmfp.swa_lr_high) 
    
    method = 'fmfp' 

    input_size = cfg.training.input_size
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.training.gpu
    cudnn.benchmark = True

    save_path = os.path.join(save_path, f'{trial.number}')
    RUN_ID = trial.number
    modelname = 'kadu' #cfg.training.model_name
    run_name = f'trial_{trial.number}'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(os.path.join(save_path, 'model_state_dict'))
        os.makedirs(os.path.join(save_path, 'logs'))

    with mlflow.start_run(run_name=run_name, nested=True):
        
        dataset_path = cfg.training.data_path
        train_loader, valid_loader, test_loader = custom_data.get_loader_local(dataset_path, batch_size=batch_size, input_size=cfg.training.input_size)
        
        num_class = cfg.training.classnumber  
        model_dict = { "num_classes": num_class }
        
        print(100 * '#')
        if modelname == 'resnet':
            model = resnet18.ResNet18(**model_dict).to(device)
        elif modelname == 'mobilenet':
            model = mobilenet.mobilenet(**model_dict).to(device)
        else:
            model = mobilenet_v2(pretrained=False, num_classes=2).to(device)
        cls_criterion = nn.CrossEntropyLoss().to(device)

            
        correctness_history = crl_utils.History(len(train_loader.dataset))
        ranking_criterion = nn.MarginRankingLoss(margin=0.0).to(device)

        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, lr=base_lr, momentum=custom_momentum, weight_decay=custom_weight_decay)
        
        swa_model = AveragedModel(model)
        scheduler = CosineAnnealingLR(optimizer, T_max=100)
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
        
        args = None
        
        for epoch in range(1, epochs + 1):
            train_loss, train_acc  = train_fmfp.train(train_loader, \
                                                    model, cls_criterion, ranking_criterion, optimizer, epoch, correctness_history, plot, method)
            
            if epoch > swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()
                    
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('train_acc', train_acc, step=epoch)

            # save model
            if epoch == epochs:
                torch.save(model.state_dict(), os.path.join(save_path, 'model_state_dict', 'model.pth'))
            
            # calc measure
            if epoch % plot == 0:
                print(f"{'#'*50} validating... {50*'#'}")
                val_loss, val_acc = validate(valid_loader, model, cls_criterion)
                mlflow.log_metric('val_loss', val_loss, step=epoch)
                mlflow.log_metric('val_acc', val_acc, step=epoch)

                acc, auroc, aupr_success, aupr, fpr, tnr, aurc, eaurc, augrc = metrics.calc_metrics(args, valid_loader,
                                                                                    model,
                                                                                    cls_criterion, save_path, 'DELETE')
                mlflow.log_metric('val_augrc', augrc, step=epoch)
    
        epoch = epochs
        mlflow.log_param('lr', base_lr)
        mlflow.log_param('swa_start', swa_start)
        mlflow.log_param('weight_decay', custom_weight_decay)
        mlflow.log_param('momentum', custom_momentum)
        mlflow.log_param('swa_lr', swa_lr)
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('model_name', modelname)
    
        torch.optim.swa_utils.update_bn(train_loader, swa_model.cpu())
        model = swa_model.to(device)
        torch.save(model.state_dict(), os.path.join(save_path, 'model_state_dict', 'model.pth'))

        acc, auroc, aupr_success, aupr, fpr, tnr, aurc, eaurc, augrc = metrics.calc_metrics(args, train_loader, model, cls_criterion, save_path, 'train') 
        
        mlflow.log_metric('train_acc', auroc, step=epoch)
        mlflow.log_metric('train_auroc', auroc, step=epoch)
        mlflow.log_metric('train_aupr_success', auroc, step=epoch)
        mlflow.log_metric('train_aupr', aupr, step=epoch)
        mlflow.log_metric('train_aurc', aurc, step=epoch)
        mlflow.log_metric('train_eaurc', eaurc, step=epoch)
        mlflow.log_metric('train_augrc', augrc, step=epoch)
        print(f'ckpt train acc: {acc}')
        print(f'ckpt train acc: {augrc}')
        acc, auroc, aupr_success, aupr, fpr, tnr, aurc, val_eaurc, val_augrc = metrics.calc_metrics(args, valid_loader, model, cls_criterion, save_path, 'val')
        
        mlflow.log_metric('val_acc', auroc, step=epoch)
        mlflow.log_metric('val_auroc', auroc, step=epoch)
        mlflow.log_metric('val_aupr_success', auroc, step=epoch)
        mlflow.log_metric('val_aupr', aupr, step=epoch)
        mlflow.log_metric('val_aurc', aurc, step=epoch)
        mlflow.log_metric('val_eaurc', val_eaurc, step=epoch)
        mlflow.log_metric('val_augrc', val_augrc, step=epoch)
        print(f'ckpt val acc: {acc}')
        print(f'ckpt val acc: {augrc}')
        acc, auroc, aupr_success, aupr, fpr, tnr, aurc, eaurc, augrc = metrics.calc_metrics(args, test_loader, model, cls_criterion, save_path, 'test')
        
        mlflow.log_metric('test_acc', auroc, step=epoch)
        mlflow.log_metric('test_auroc', auroc, step=epoch)
        mlflow.log_metric('test_aupr_success', auroc, step=epoch)
        mlflow.log_metric('test_aupr', aupr, step=epoch)
        mlflow.log_metric('test_aurc', aurc, step=epoch)
        mlflow.log_metric('test_eaurc', eaurc, step=epoch)
        mlflow.log_metric('test_augrc', augrc, step=epoch)
        print(f'ckpt test acc: {acc}')
        print(f'ckpt test acc: {augrc}')
        mlflow.pytorch.log_model(model, artifact_path="model")
        '''
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
                mlflow.log_metric(key, val)
        
        augrc_hw_val = result.get("augrc_hw_val")
        '''
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
                mlflow.log_metric(key, val)
        
        augrc_hw_val = result.get("augrc_hw_val")

        gc.collect()
        torch.cuda.empty_cache()

        return float(augrc_hw_val)
    

def main():
    study_name = cfg.training.study_name #input('study_name: ')
    storage = f'sqlite:///{study_name}_storage.db'
    study = optuna.create_study(direction='minimize', load_if_exists=True, study_name = study_name, storage=storage)
    print(f"Sampler is {study.sampler.__class__.__name__}")
    mlflow.log_param('sampler', study.sampler.__class__.__name__)
    study.optimize(objective, n_trials=3, n_jobs=1)

    print("Best hyperparameters:", study.best_params)
    print("Best accuracy:", study.best_value)

if __name__ == "__main__":
    main()
