import utils.crl_utils
from utils import utils
import torch.nn as nn
import time
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import random
from hydra import initialize, compose

with initialize(config_path="../configs/"):
    cfg = compose(config_name="fmfp")  # exp1.yaml with defaults key

device = cfg.training.device #torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train(loader, model, criterion, criterion_ranking, optimizer, epoch, history, plot, method):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    total_losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    cls_losses = utils.AverageMeter()
    ranking_losses = utils.AverageMeter()
    end = time.time()
    model.train()
    
    for i, (input, target, idx, _) in enumerate(loader):
        data_time.update(time.time() - end)
        input, target = input.to(device), target.long().to(device)

        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        if method == 'sam' or method == 'fmfp':
            optimizer.first_step(zero_grad=True)
            criterion(model(input), target).backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()

        # record loss and accuracy
        prec, correct = utils.accuracy(output, target)

        total_losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % plot == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
           'Prec {top1.val:.2f}% ({top1.avg:.2f}%)'.format(epoch, i, len(loader), batch_time=batch_time,data_time=data_time, loss=total_losses, top1=top1))
    return total_losses.avg, top1.avg