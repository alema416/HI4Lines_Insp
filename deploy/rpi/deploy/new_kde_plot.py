import pandas as pd
from nmetr import AUGRC
import argparse
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='Rethinking CC for FP')
parser.add_argument('--run_id', required=True, type=int, help='')
parser.add_argument('--idel', required=True, type=str, help='')
args = parser.parse_args()

def custom_seaborn(df_pos, df_neg, model_id, split, prefix, AUGRC, ACC, mean, std, cc):
    if len(df_pos) < 2 or len(df_neg) < 2:
        return

    # extract
    pos = df_pos['confidence']
    neg = df_neg['confidence']

    plt.figure(figsize=(6,4))
    # `cut=3` extends each KDE 3 bandwidths beyond the data min/max
    sns.kdeplot(pos, 
                fill=True, 
                color='green', 
                alpha=0.4, 
                bw_method='scott', 
                cut=3,
                label='correct predictions')
    sns.kdeplot(neg, 
                fill=True, 
                color='red', 
                alpha=0.4, 
                bw_method='scott', 
                cut=3,
                label='misclassifications')
    modelname = 'resnet' #'mobilenet' #input('Model Name: ') #'ResNet18'
    plt.xlabel('confidence')
    plt.ylabel('density')
    plt.xlim(0.0, 1.2)
    info = f"Model: {modelname}\nID: {id}\n"
    if split == 'test':
        info += f"AUGRC: {AUGRC['test']:.2f}\n"
        info += f"success mean: {mean['s_test']:.2f}, success st: {std['s_test']:.2f}\n"
        info += f"error mean: {mean['e_test']:.2f}, error std: {std['e_test']:.2f}\n"

    else:
        info += f"AUGRC: {AUGRC['train']:.2f}, {AUGRC['val']:.2f}\n"
        # given mu1, sigma1, n1, mu2, sigma2, n2:
        n1 = cc['succ_tr']
        n2 = cc['succ_val']

        mu1 = mean['s_train']
        mu2 = mean['s_val']
        sigma1 = std['s_train']
        sigma2 = std['s_val']
    
        mu_tot = (n1*mu1 + n2*mu2) / (n1 + n2)

        num = n1*(sigma1**2 + mu1**2) + n2*(sigma2**2 + mu2**2)
        sigma_tot_sq = num/(n1+n2) - mu_tot**2
        sigma_tot = np.sqrt(sigma_tot_sq)
        info += f"success mean: {mu_tot:.2f}, success std: {sigma_tot:.2f}\n"

        n1 = cc['err_tr']
        n2 = cc['err_val']

        mu1 = mean['e_train']
        mu2 = mean['e_val']
        sigma1 = std['e_train']
        sigma2 = std['e_val']
    
        mu_tot = (n1*mu1 + n2*mu2) / (n1 + n2)

        num = n1*(sigma1**2 + mu1**2) + n2*(sigma2**2 + mu2**2)
        sigma_tot_sq = num/(n1+n2) - mu_tot**2
        sigma_tot = np.sqrt(sigma_tot_sq)
        info += f"error mean: {mu_tot:.2f}, error std: {sigma_tot:.2f}\n"

    print(prefix)	
    name = 'Baseline' if prefix=='b_' else 'FMFP'
    print(name)
    plt.gca().text(
        0.02, 0.68, info,
        transform=plt.gca().transAxes,
        va='top', ha='left',
        fontsize=7,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6)
    )
    if split == 'test':
        plt.title(f'{name} - {modelname} - id={model_id} - {split} set')
    else:
        plt.title(f'{name} - {modelname} - id={model_id} - {split} sets')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f'custom_{prefix}{model_id}_{split}.png', dpi=300)
    plt.close()

id = args.run_id
idel = args.idel

import degirum as dg
import degirum_tools
from pprint import pprint
import class_eval_src
import argparse
from hydra import initialize, compose

with initialize(config_path="../../../configs/"):
    cfg = compose(config_name="hw_classifier")  # exp1.yaml with defaults key

addon = '' if args.idel == '' else '_bsln' 
model_name = f'{cfg.classifier.modelname}_{args.run_id}{addon}'
#model_name = cfg.classifier.modelname
print(model_name)
model = dg.load_model(
    model_name=model_name,
    inference_host_address='@local',
    zoo_url= cfg.classifier.model_zoo_dir
)
evaluator_tr = class_eval_src.ImageClassificationModelEvaluator(
    model,
    show_progress=False, split='train', id=id, ide=idel,
    top_k=[1],  # Evaluate top-1 and top-5 accuracy
    foldermap={0: "broken", 1: "healthy"}  # Mapping class IDs to folders
)
evaluator_v = class_eval_src.ImageClassificationModelEvaluator(
    model,
    show_progress=False, split='val', id=id, ide=idel,
    top_k=[1],  # Evaluate top-1 and top-5 accuracy
    foldermap={0: "broken", 1: "healthy"}  # Mapping class IDs to folders
)
evaluator_te = class_eval_src.ImageClassificationModelEvaluator(
    model,
    show_progress=False, split='test',id=id, ide=idel,
    top_k=[1],  # Evaluate top-1 and top-5 accuracy
    foldermap={0: "broken", 1: "healthy"}  # Mapping class IDs to folders
)

print(f'WORKING ON: train set')
results_train, tr_mean_tr, tr_std_tr, tr_mean_err, tr_std_err, tr_tr, tr_fa = evaluator_tr.evaluate(cfg.classifier.train_set_dir, None, -1)
print(f'WORKING ON: val set')
results_eval, val_mean_tr, val_std_tr, val_mean_err, val_std_err, val_tr, val_fa = evaluator_v.evaluate(cfg.classifier.val_set_dir, None, -1)
print(f'WORKING ON: test set')
start = time.time()
results_test, test_mean_tr, test_std_tr, test_mean_err, test_std_err, test_tr, test_fa = evaluator_te.evaluate(cfg.classifier.test_set_dir, None, -1)
end = time.time()
length = end - start

# Show the results : this can be altered however you like
print("It took", length, "seconds!")

ide = idel
print(f'train_set top1 acc: {results_train[0][0]:.5f}%')
print(f'validation_set top1 acc: {results_eval[0][0]:.5f}%')
print(f'test_set top1 acc: {results_test[0][0]:.5f}%')

print(f'train_set per_class accuracies: {results_train[1][0][0]:.5f}%, {results_train[1][1][0]:.5f}%')
print(f'validation_set per_class accuracies: {results_eval[1][0][0]:.5f}%, {results_eval[1][1][0]:.5f}%')
print(f'test_set per_class accuracies: {results_test[1][0][0]:.5f}%, {results_test[1][1][0]:.5f}%')

print(f'SPECIAL_PRINTacctrain {results_train[0][0]:.3f}')
print(f'SPECIAL_PRINTaccval {results_eval[0][0]:.3f}')
print(f'SPECIAL_PRINTacctest {results_test[0][0]:.3f}')

cc = {}
mean = {}
std = {}

cc['succ_tr'] = tr_tr
cc['succ_val'] = val_tr
cc['succ_test'] = test_tr
mean['s_train'] = tr_mean_tr
mean['s_val'] = val_mean_tr
mean['s_test'] = test_mean_tr
std['s_train'] = tr_std_tr
std['s_val'] = val_std_tr
std['s_test'] = test_std_tr
cc['err_tr'] = tr_fa
cc['err_val'] = val_fa
cc['err_test'] = test_fa
mean['e_train'] = tr_mean_err
mean['e_val'] = val_mean_err
mean['e_test'] = test_mean_err
std['e_train'] = tr_std_err
std['e_val'] = val_std_err
std['e_test'] = test_std_err

ACC_dict = {'train': results_train[0][0], 'val': results_eval[0][0], 'test': results_test[0][0]}

AUGRC_dict = {}
for split in ['train', 'val', 'test']:
    with open(f'{ide}labels_{id}_{split}.txt', "r") as file:
      labels = [int(line.strip()) for line in file]
    with open(f'{ide}confs_{id}_{split}.txt', "r") as file:
      confs = [float(line.strip()) for line in file]  
    
    probs = torch.tensor(confs, dtype=torch.float32)  # Now shape (N, C)
    numeric_labels_tensor = torch.tensor(labels, dtype=torch.long)
    augrc_metric = AUGRC()
  
    augrc_metric.update(probs, numeric_labels_tensor)
    
    augrc_value = augrc_metric.compute()
    print(f'SPECIAL_PRINTaugrc{split} {1000*augrc_value.item()}')
    AUGRC_dict[f'{split}'] = 1000*augrc_value.item()
print(AUGRC_dict)
print(ACC_dict)





if True:
    if True:
        for ide in [idel]:
            # Replace 'confidences.txt' and 'labels.txt' with your actual file paths
            confidences = pd.read_csv(f'{ide}confs_{id}_train.txt', header=None, names=['confidence'])
            labels = pd.read_csv(f'{ide}labels_{id}_train.txt', header=None, names=['correct'])

            # Combine into a single DataFrame
            df1 = pd.concat([confidences, labels], axis=1)

            confidences = pd.read_csv(f'{ide}confs_{id}_val.txt', header=None, names=['confidence'])
            labels = pd.read_csv(f'{ide}labels_{id}_val.txt', header=None, names=['correct'])

            # Combine into a single DataFrame
            df2 = pd.concat([confidences, labels], axis=1)


            #print(len(df1))
            #print(len(df2))
            df = pd.concat([df1, df2], axis=0, ignore_index=True)
            print(len(df))
            df_zero = df[df['correct'] == 0]
            df_one = df[df['correct'] == 1]
            #plt.xlim(x_min, x_max)

            #plt.tight_layout()
            custom_seaborn(df_one, df_zero, id, 'train-val', ide, AUGRC_dict, ACC_dict, mean, std, cc)
            df_zero.to_csv(f'output_{ide}{id}_valtrain_error.csv', index=False)
            df_one.to_csv(f'output_{ide}{id}_valtrain_success.csv', index=False)

if True:
    if True:
        for ide in [idel]:
            # Replace 'confidences.txt' and 'labels.txt' with your actual file paths
            confidences = pd.read_csv(f'{ide}confs_{id}_test.txt', header=None, names=['confidence'])
            labels = pd.read_csv(f'{ide}labels_{id}_test.txt', header=None, names=['correct'])

            # Combine into a single DataFrame
            df = pd.concat([confidences, labels], axis=1)
            print(len(df))
            df_zero = df[df['correct'] == 0]
            df_one = df[df['correct'] == 1]

            custom_seaborn(df_one, df_zero, id, 'test', ide, AUGRC_dict, ACC_dict, mean, std, cc)

            df_zero.to_csv(f'output_{ide}{id}_test_error.csv', index=False)
            df_one.to_csv(f'output_{ide}{id}_test_success.csv', index=False)

