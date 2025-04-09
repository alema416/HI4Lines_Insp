import torch
import gc
import distutils
import time
import optuna
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
from torch.utils.tensorboard import SummaryWriter

from collections import OrderedDict
from model import LeNet
from model import AlexNet
from model import resnet
import requests
from model import resnet18
from model import densenet_BC
from model import vgg
from model import mobilenet
from model import efficientnet
from model import wrn
from model import convmixer
from utils import data as dataset
from utils import crl_utils
from utils import metrics
from utils import utils
import train_base
import custom_data as custom_data

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

def validate(loader, model, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for input, target, idx, _ in loader:
            input, target = input.cuda(), target.long().cuda()
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


parser = argparse.ArgumentParser(description='Rethinking CC for FP')
parser.add_argument('--epochs', default=200, type=int, help='Total number of epochs to run')
parser.add_argument('--plot', default=5, type=int, help='')
parser.add_argument('--model', default='resnet18', type=str, help='Models name to use [res110, dense, wrn, cmixer, efficientnet, mobilenet, vgg]')
parser.add_argument('--method', default='Baseline', type=str, help='[Baseline, Mixup, LS, L1, focal, CRL]')
parser.add_argument('--rank_weight', default=1.0, type=float, help='Rank loss weight')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
parser.add_argument('--print-freq', '-p', default=72, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--cwd_weight', default=0.1, type=float, help='Training time tempscaling')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
args = parser.parse_args()

def objective(trial):
    server1 = True
    server2 = not server1
    
    base_lr = trial.suggest_loguniform('lr', 1e-3, 1e-1) 
    lr_strat = [80, 130, 170]
    lr_factor = 0.1  # Learning rate decrease factor
    custom_weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)  
    custom_momentum = trial.suggest_uniform('momentum', 0.85, 0.99)

    save_path = '/home/amax/machairas/FMFP-edge-idid/hailo_src/haht_augrc_resnet_18_baseline_augrc_2/' if server1 else '/home/apel/machairas/HAILO/shared_with_docker/haht_augrc_resnet_18_baseline/'
    batch_size = 16 if 'server2' else 16

    data = 'idid_cropped'
    classnumber = 2
    input_size = 224
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cudnn.benchmark = True

    save_path = os.path.join(save_path, f'{trial.number}')
    RUN_ID = trial.number
    method = 'baseline'
    run_name = f'{args.model}_{method}_{input_size}_{args.epochs}'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(log_dir=save_path)
    dataset_path = '/home/amax/machairas/FMFP-edge-idid/yolo_m2_class_square_JOIN_224/' if server1 else '/home/apel/machairas/pipeline/tutorials/docker/classification/out/yolo_m2_class_square_JOIN_224/'

    train_loader, valid_loader, test_loader = custom_data.get_loader_local(dataset_path, batch_size=batch_size, input_size=224)
    num_class = 2
    
    model_dict = {"num_classes": num_class}
    print(100*'#')
    model = resnet18.ResNet18(**model_dict).cuda()

    cls_criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=custom_momentum,
                                weight_decay=custom_weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=lr_strat, gamma=lr_factor)

    correctness_history = crl_utils.History(len(train_loader.dataset))
    ranking_criterion = nn.MarginRankingLoss(margin=0.0).cuda()

    # start Train
    for epoch in range(1, args.epochs + 1):
        
        train_loss, train_acc = train_base.train(train_loader,
                    model,
                    cls_criterion,
                    ranking_criterion,
                    optimizer,
                    epoch,
                    correctness_history,
                    args, classnumber)
        scheduler.step()
        
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)

        # save model
        if epoch == args.epochs:
            torch.save(model.state_dict(),
                        os.path.join(save_path, 'model.pth'))
        # finish train

        # calc measure
        if epoch % args.plot == 0:
            print(f"{'#'*50} validating... {50*'#'}")
            val_loss, val_acc = validate(valid_loader, model, cls_criterion)
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

    writer.add_scalar('Params/initial_lr', base_lr, args.epochs)
    writer.add_scalar('Params/lr_factor', lr_factor, args.epochs)
    writer.add_scalar('Params/weight_decay', custom_weight_decay, args.epochs)
    writer.add_scalar('Params/momentum', custom_momentum, args.epochs)
    writer.add_scalar('Params/trial_number', trial.number, args.epochs)
    epoch = args.epochs
    torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))
    
    acc, auroc, aupr_success, aupr, fpr, tnr, aurc, eaurc, augrc = metrics.calc_metrics(args, train_loader, model, cls_criterion, save_path, 'train') 
    writer.add_scalar('final_Metrics/train_accuracy', acc, epoch)
    writer.add_scalar('final_Metrics/train_auroc', auroc, epoch)
    writer.add_scalar('final_Metrics/train_aupr_success', aupr_success, epoch)
    writer.add_scalar('final_Metrics/train_aupr', aupr, epoch)
    writer.add_scalar('final_Metrics/train_fpr', fpr, epoch)
    writer.add_scalar('final_Metrics/train_tnr', fpr, epoch)
    writer.add_scalar('final_Metrics/train_aurc', aurc, epoch)                
    writer.add_scalar('final_Metrics/train_eaurc', eaurc, epoch)
    writer.add_scalar('final_Metrics/train_augrc', augrc, epoch)

    val_acc, auroc, aupr_success, aupr, fpr, tnr, aurc, eaurc, val_augrc = metrics.calc_metrics(args, valid_loader, model, cls_criterion, save_path, 'val')
    
    writer.add_scalar('final_Metrics/val_accuracy', val_acc, epoch)
    writer.add_scalar('final_Metrics/val_auroc', auroc, epoch)
    writer.add_scalar('final_Metrics/val_aupr_success', aupr_success, epoch)
    writer.add_scalar('final_Metrics/val_aupr', aupr, epoch)
    writer.add_scalar('final_Metrics/val_fpr', fpr, epoch)
    writer.add_scalar('final_Metrics/val_tnr', fpr, epoch)
    writer.add_scalar('final_Metrics/val_aurc', aurc, epoch)                
    writer.add_scalar('final_Metrics/val_eaurc', eaurc, epoch)
    writer.add_scalar('final_Metrics/val_augrc', val_augrc, epoch)

    acc, auroc, aupr_success, aupr, fpr, tnr, aurc, eaurc, augrc = metrics.calc_metrics(args, test_loader, model, cls_criterion, save_path, 'test')
    
    writer.add_scalar('final_Metrics/test_accuracy', acc, epoch)
    writer.add_scalar('final_Metrics/test_auroc', auroc, epoch)
    writer.add_scalar('final_Metrics/test_aupr_success', aupr_success, epoch)
    writer.add_scalar('final_Metrics/test_aupr', aupr, epoch)
    writer.add_scalar('final_Metrics/test_fpr', fpr, epoch)
    writer.add_scalar('final_Metrics/test_tnr', fpr, epoch)
    writer.add_scalar('final_Metrics/test_aurc', aurc, epoch)                
    writer.add_scalar('final_Metrics/test_eaurc', eaurc, epoch)
    writer.add_scalar('final_Metrics/test_augrc', augrc, epoch)

    ccc = 0
    while ccc < 10:
        try:
            port = 5000 if server1 else 5001
            response = requests.post(f"http://localhost:{port}/validate", json={"run_id": RUN_ID})
            response.raise_for_status()
            break
        except requests.RequestException as e:
            print(e)
            print(f"================ERROR #{ccc}================")
            #time.sleep(60)
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
    
    #end_time = time.time()
    #print(f"model hardware check took: {((end_time - start_time)/60):.1f} min")
    
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
    storage = 'sqlite:///baseline_resnet18_optimize_augrc_hw2232d.db'
    study = optuna.create_study(direction='minimize', study_name = "baseline1", storage=storage)
    print(f"Sampler is {study.sampler.__class__.__name__}")
    study.optimize(objective, n_trials=50, n_jobs=1)

    print("Best hyperparameters:", study.best_params)
    print("Best accuracy:", study.best_value)
if __name__ == "__main__":
    main()
