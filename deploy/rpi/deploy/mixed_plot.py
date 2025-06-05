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
import degirum as dg
import degirum_tools
from pprint import pprint
import class_eval_src
import argparse
from hydra import initialize, compose
import argparse

with initialize(config_path="../../../configs/"):
    cfg = compose(config_name="hw_classifier")  # exp1.yaml with defaults key

def custom_seaborn(df_pos, df_neg, model_id1, model_id2, split, prefix):
    if len(df_pos) < 2 or len(df_neg) < 2:
        return
        
    # extract
    #pos = df_pos1['confidence']
    #neg = df_neg1['confidence']
    pos = pd.concat([df_pos['confidence'], df_pos['confidence']], ignore_index=True)

    # similarly for neg if needed:
    neg = pd.concat([df_neg['confidence'], df_neg['confidence']], ignore_index=True)
    
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
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.title('2 best models train&val sets confidence separation')
    plt.savefig(f'MIXED_custom_{prefix}{model_id1}{model_id2}_{split}.png')
    plt.close()
idel = ''
if True:
    if True:
        for ide in [idel]:
            # Replace 'confidences.txt' and 'labels.txt' with your actual file paths
            id = 14
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
            df_1 = pd.concat([df1, df2], axis=0, ignore_index=True)
            
            
            id = 18
            confidences = pd.read_csv(f'{ide}confs_{id}_train.txt', header=None, names=['confidence'])
            labels = pd.read_csv(f'{ide}labels_{id}_train.txt', header=None, names=['correct'])

            # Combine into a single DataFrame
            df1 = pd.concat([confidences, labels], axis=1)

            confidences = pd.read_csv(f'{ide}confs_{id}_val.txt', header=None, names=['confidence'])
            labels = pd.read_csv(f'{ide}labels_{id}_val.txt', header=None, names=['correct'])

            # Combine into a single DataFrame
            df2 = pd.concat([confidences, labels], axis=1)


            df_2 = pd.concat([df1, df2], axis=0, ignore_index=True)

            df = pd.concat([df_1, df_2], axis=0, ignore_index=True)

            
            df_zero = df[df['correct'] == 0]
            df_one = df[df['correct'] == 1]
            #plt.xlim(x_min, x_max)

            #plt.tight_layout()
            custom_seaborn(df_one, df_zero, 14, 18, 'train-val', ide)
            #df_zero.to_csv(f'output_{ide}{id}_valtrain_error.csv', index=False)
            #df_one.to_csv(f'output_{ide}{id}_valtrain_success.csv', index=False)
'''
if True:
    if True:
        for ide in [idel]:
            # Replace 'confidences.txt' and 'labels.txt' with your actual file paths
            confidences = pd.read_csv(f'{ide}confs_{id}_test.txt', header=None, names=['confidence'])
            labels = pd.read_csv(f'{ide}labels_{id}_test.txt', header=None, names=['correct'])

            # Combine into a single DataFrame
            df = pd.concat([confidences, labels], axis=1)
            df_zero = df[df['correct'] == 0]
            df_one = df[df['correct'] == 1]

            custom_seaborn(df_one1, df_zero1, df_one2, df_zero2, id, 'test', ide)

            #df_zero.to_csv(f'output_{ide}{id}_test_error.csv', index=False)
            #df_one.to_csv(f'output_{ide}{id}_test_success.csv', index=False)
'''
