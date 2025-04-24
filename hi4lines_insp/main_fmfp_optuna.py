import torch
# Patch for distutils.version.LooseVersion if missing
import distutils
import requests
import gc
import threading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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
from model import resnet18_custom
from model import resnet18
from utils import data as dataset
from utils import crl_utils
from utils import metrics
from utils import utils
from torch.utils.tensorboard import SummaryWriter
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

def objective(trial):
    #cfg = OmegaConf.load("../configs/fmfp.yaml")
    with initialize(config_path="../configs/"):
        cfg = compose(config_name="fmfp")  # exp1.yaml with defaults key
    print(cfg)
    server1 = True
    server2 = not server1
    epochs = cfg.training.epochs
    plot = cfg.training.validate_freq
    batch_size = cfg.training.batch_size 
    port = 5001
    save_path = cfg.training.save_path
    base_lr = trial.suggest_loguniform('lr', cfg.training.base_lr_low, cfg.training.base_lr_high)
    
    method = 'fmfp' 
    swa_start = trial.suggest_int("swa_start", cfg.fmfp.swa_start_low, cfg.fmfp.swa_start_high)
    
    custom_weight_decay = trial.suggest_loguniform('weight_decay', cfg.training.weight_decay_low, cfg.training.weight_decay_high) 
    custom_momentum = trial.suggest_uniform('momentum', cfg.training.momentum_low, cfg.training.momentum_high) 
    swa_lr = trial.suggest_loguniform('swa_lr', cfg.fmfp.swa_lr_low, cfg.fmfp.swa_lr_high) 

    data = 'idid_cropped'
    classnumber = cfg.training.classnumber
    input_size = cfg.training.input_size
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.training.gpu
    cudnn.benchmark = True

    save_path = os.path.join(save_path, f'{trial.number}')
    RUN_ID = trial.number
    modelname = cfg.training.model_name
    run_name = f'{modelname}_{method}_{input_size}_{swa_start}_{epochs}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(os.path.join(save_path, 'logs'))
        os.makedirs(os.path.join(save_path, 'model_state_dict'))

    writer = SummaryWriter(log_dir=os.path.join(save_path, 'logs'))

    dataset_path = cfg.training.data_path
    train_loader, valid_loader, test_loader = custom_data.get_loader_local(dataset_path, batch_size=batch_size, input_size=cfg.training.input_size)
    
    num_class = cfg.training.classnumber  
    model_dict = { "num_classes": num_class }
    
    print(100 * '#')
    
    model = resnet18.ResNet18(**model_dict).to(device)
    cls_criterion = nn.CrossEntropyLoss().to(device)

        
    correctness_history = crl_utils.History(len(train_loader.dataset))
    ranking_criterion = nn.MarginRankingLoss(margin=0.0).to(device)

    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=base_lr, momentum=custom_momentum, weight_decay=custom_weight_decay)
    
    swa_model = AveragedModel(model)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
    # start Train
    
    early_stopping_patience = 100
    min_delta = 0.01
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    best_model_epoch = None
    validation_lock = threading.Lock()
    args = None
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc  = train_fmfp.train(train_loader, \
                                                  model, cls_criterion, ranking_criterion, optimizer, epoch, correctness_history, plot, method)
        
        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
                
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)

        # save model
        if epoch == epochs:
            torch.save(model.state_dict(), os.path.join(save_path, 'model_state_dict', 'model.pth'))

        # finish train

        # calc measure
        if epoch % plot == 0:
            print(f"{'#'*50} validating... {50*'#'}")
            val_loss, val_acc = validate(valid_loader, model, cls_criterion)
            trial.report(val_loss, step=epoch)
            #if trial.should_prune():
            #    raise optuna.TrialPruned()
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Accuracy/Validation', val_acc, epoch)

            acc, auroc, aupr_success, aupr, fpr, tnr, aurc, eaurc, augrc = metrics.calc_metrics(args, valid_loader,
                                                                                model,
                                                                                cls_criterion, save_path, 'DELETE')
            writer.add_scalar('Metrics/accuracy', acc, epoch)
            writer.add_scalar('Metrics/auroc', auroc, epoch)
            writer.add_scalar('Metrics/aupr_success', aupr_success, epoch)
            writer.add_scalar('Metrics/aupr', aupr, epoch)
            writer.add_scalar('Metrics/fpr', fpr, epoch)
            writer.add_scalar('Metrics/tnr', fpr, epoch)
            writer.add_scalar('Metrics/aurc', aurc, epoch)                
            writer.add_scalar('Metrics/eaurc', eaurc, epoch)
            writer.add_scalar('Metrics/augrc', augrc, epoch)
        
        
        
            #best_model_state = swa_model.state_dict()  # Save best state
               
    epoch = epochs
         
    writer.add_scalar('Params/initial_lr', base_lr, epoch)
    writer.add_scalar('Params/weight_decay', custom_weight_decay, epoch)
    writer.add_scalar('Params/momentum', custom_momentum, epoch)
    writer.add_scalar('Params/trial_number', trial.number, epoch)
    
    writer.add_text('Params/method', method, epoch)
    writer.add_scalar('Params/swa_start', swa_start, epoch)
    writer.add_scalar('Params/swa_lr', swa_lr, epoch)
    print('before update')
    torch.optim.swa_utils.update_bn(train_loader, swa_model.cpu())
    print('after update')
    model = swa_model.to(device)
    print('model loaded!!')
    torch.save(model.state_dict(), os.path.join(save_path, 'model_state_dict', 'model.pth'))

    print('model saved!!')
    acc, auroc, aupr_success, aupr, fpr, tnr, aurc, eaurc, augrc = metrics.calc_metrics(args, train_loader, model, cls_criterion, save_path, 'train') 
    
    writer.add_scalar('final_Metrics/train_accuracy', acc, epoch)
    writer.add_scalar('final_Metrics/train_auroc', auroc, epoch)
    writer.add_scalar('final_Metrics/train_aupr_success', aupr_success, epoch)
    writer.add_scalar('final_Metrics/train_aupr', aupr, epoch)
    writer.add_scalar('final_Metrics/train_fpr', fpr, epoch)
    writer.add_scalar('final_Metrics/train_tnr', tnr, epoch)
    writer.add_scalar('final_Metrics/train_aurc', aurc, epoch)                
    writer.add_scalar('final_Metrics/train_eaurc', eaurc, epoch)
    writer.add_scalar('final_Metrics/train_augrc', augrc, epoch)
    
    acc, auroc, aupr_success, aupr, fpr, tnr, aurc, val_eaurc, val_augrc = metrics.calc_metrics(args, valid_loader, model, cls_criterion, save_path, 'val')
    
    writer.add_scalar('final_Metrics/val_accuracy', acc, epoch)
    writer.add_scalar('final_Metrics/val_auroc', auroc, epoch)
    writer.add_scalar('final_Metrics/val_aupr_success', aupr_success, epoch)
    writer.add_scalar('final_Metrics/val_aupr', aupr, epoch)
    writer.add_scalar('final_Metrics/val_fpr', fpr, epoch)
    writer.add_scalar('final_Metrics/val_tnr', tnr, epoch)
    writer.add_scalar('final_Metrics/val_aurc', aurc, epoch)                
    writer.add_scalar('final_Metrics/val_eaurc', val_eaurc, epoch)
    
    
    acc, auroc, aupr_success, aupr, fpr, tnr, aurc, eaurc, augrc = metrics.calc_metrics(args, test_loader, model, cls_criterion, save_path, 'test')
    
    writer.add_scalar('final_Metrics/test_accuracy', acc, epoch)
    writer.add_scalar('final_Metrics/test_auroc', auroc, epoch)
    writer.add_scalar('final_Metrics/test_aupr_success', aupr_success, epoch)
    writer.add_scalar('final_Metrics/test_aupr', aupr, epoch)
    writer.add_scalar('final_Metrics/test_fpr', fpr, epoch)
    writer.add_scalar('final_Metrics/test_tnr', tnr, epoch)
    writer.add_scalar('final_Metrics/test_aurc', aurc, epoch)                
    writer.add_scalar('final_Metrics/test_eaurc', eaurc, epoch)
    writer.add_scalar('final_Metrics/test_augrc', augrc, epoch)
    
    ccc = 0
    hailo_ip = input('hailo ip address: ')
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
    augrc_hw_train = result.get("augrc_hw_train")
    acc_hw_train = result.get("acc_hw_train")
    
    augrc_hw_val = result.get("augrc_hw_val")
    acc_hw_val = result.get("acc_hw_val")
    
    augrc_hw_test = result.get("augrc_hw_test")
    acc_hw_test = result.get("acc_hw_test")
    
    augrc_emu = result.get("augrc_emu")
    acc_emu = result.get("acc_emu")
    
    print(augrc_hw_val)
    print(augrc_emu)
    print(acc_hw_val)
    print(acc_emu)
    
    
    writer.add_scalar('final_Metrics/train_augrc_hw', augrc_hw_train, epoch)
    writer.add_scalar('final_Metrics/train_acc_hw', acc_hw_train, epoch)
    
    writer.add_scalar('final_Metrics/val_augrc_hw', augrc_hw_val, epoch)
    writer.add_scalar('final_Metrics/val_acc_hw', acc_hw_val, epoch)
    
    writer.add_scalar('final_Metrics/test_augrc_hw', augrc_hw_test, epoch)
    writer.add_scalar('final_Metrics/test_acc_hw', acc_hw_test, epoch)
    
    
    writer.add_scalar('final_Metrics/val_acc_emu', acc_emu, epoch)    
    writer.add_scalar('final_Metrics/val_augrc_emu', augrc_emu, epoch)
    
    writer.close()
    gc.collect()
    torch.cuda.empty_cache()
    return float(augrc_hw_val)
    

def main():
    study_name = input('study_name: ')
    storage = "sqlite:///haht_augrc_quant_no_ES.db" #'' # (sensitive)
    study = optuna.create_study(direction='minimize', load_if_exists=True, study_name = study_name, storage=storage)
    print(f"Sampler is {study.sampler.__class__.__name__}")
    study.optimize(objective, n_trials=400, n_jobs=1)

    print("Best hyperparameters:", study.best_params)
    print("Best accuracy:", study.best_value)

if __name__ == "__main__":
    main()
