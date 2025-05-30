import pandas as pd
from nmetr import AUGRC
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def custom_seaborn(df_pos, df_neg, model_id, split, prefix):
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

    plt.xlabel('confidence')
    plt.ylabel('density')
    plt.xlim(0.0, 1.2)
    name = 'baseline' if prefix=='b_' else 'fmfp'
    plt.title(f'{name} - model {model_id} - {split} set(s)')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f'custom_{prefix}{model_id}_{split}.png', dpi=300)
    plt.close()

def custom(df, dfn, id, split, ide):
    if len(df) < 2 or len(dfn) < 2:
        return 0
    values = df['confidence'].values
    values_n = dfn['confidence'].values
    # 2. Compute the kernel density estimate
    kde = gaussian_kde(values, bw_method='scott')   # you can change bw_method to adjust smoothing
    kden = gaussian_kde(values_n, bw_method='scott')
    # 3. Prepare a grid for plotting
    x_min, x_max = values.min(), values.max()
    x_minn, x_maxn = values_n.min(), values_n.max()
    xs = np.linspace(x_min, x_max, 200)
    xsn = np.linspace(x_minn, x_maxn, 200)
    x_min = min(values.min(), values_n.min())
    x_max = max(values.max(), values_n.max())
    xs = np.linspace(x_min, x_max, 200)
    # 4. Plot the KDE curve
    plt.figure()
    plt.fill_between(xs, kde(xs), alpha=0.4, color='green')    # fill under the curve
    plt.plot(xs, kde(xs), linewidth=2, color='green')
    plt.fill_between(xs, kden(xs), alpha=0.4, color='red')
    plt.plot(xs, kden(xs), linewidth=2, color='red')
    #plt.xlim(0, 1.2)
    plt.xlim(0.0, x_max)

    plt.tight_layout()
    plt.xlabel('confidence')
    plt.ylabel('density')
    modelname = 'baseline' if ide=='b_' else 'fmfp'
    plt.title(f'{modelname} - model {id} - {split} set(s)')
    plt.savefig(f'custom_{ide}{id}_{split}.png')
id = input('id: ')

idel = ''





import degirum as dg
import degirum_tools
from pprint import pprint
import class_eval_src
import argparse
from hydra import initialize, compose

with initialize(config_path="../../../configs/"):
    cfg = compose(config_name="hw_classifier")  # exp1.yaml with defaults key

model_name = cfg.classifier.modelname

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
results_train = evaluator_tr.evaluate(cfg.classifier.train_set_dir, None, -1)
print(f'WORKING ON: val set')
results_eval = evaluator_v.evaluate(cfg.classifier.val_set_dir, None, -1)
print(f'WORKING ON: test set')
results_test = evaluator_te.evaluate(cfg.classifier.test_set_dir, None, -1)
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
        for ide in ['']:
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
            custom_seaborn(df_one, df_zero, id, 'train-val', ide)
            df_zero.to_csv(f'output_{ide}{id}_valtrain_error.csv', index=False)
            df_one.to_csv(f'output_{ide}{id}_valtrain_success.csv', index=False)

if True:
    if True:
        for ide in ['']:
            # Replace 'confidences.txt' and 'labels.txt' with your actual file paths
            confidences = pd.read_csv(f'{ide}confs_{id}_test.txt', header=None, names=['confidence'])
            labels = pd.read_csv(f'{ide}labels_{id}_test.txt', header=None, names=['correct'])

            # Combine into a single DataFrame
            df = pd.concat([confidences, labels], axis=1)
            print(len(df))
            df_zero = df[df['correct'] == 0]
            df_one = df[df['correct'] == 1]

            custom_seaborn(df_one, df_zero, id, 'test', ide)
            df_zero.to_csv(f'output_{ide}{id}_test_error.csv', index=False)
            df_one.to_csv(f'output_{ide}{id}_test_success.csv', index=False)
