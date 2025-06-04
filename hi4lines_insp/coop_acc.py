import numpy as np
import pandas as pd
import os
def coop_accuracy(y_true, pi_pred, pi_conf, srv_pred):

    #   y_true      — ground-truth labels (ints)
    #   pi_pred     — Pi’s predicted labels
    #   pi_conf     — Pi’s confidences (floats in [0,1])
    #   srv_pred    — server’s predicted labels
    
    y_true, pi_pred, pi_conf, srv_pred = np.asarray(y_true), np.asarray(pi_pred), np.asarray(pi_conf), np.asarray(srv_pred)
    N = y_true.shape[0]

    thresholds = np.linspace(0.5, 1.001, 10)   # e.g. 0%, 1%, …, 100%
    mask = pi_conf[None, :] >= thresholds[:, None]
    coop_preds = np.where(mask, pi_pred[None, :], srv_pred[None, :])
    accuracies = (coop_preds == y_true[None, :]).mean(axis=1)
    
    offloaded_count = (~mask).sum(axis=1)
    offloaded_ratio = offloaded_count / N
    
    acc_df = pd.DataFrame({
        'thresholds': thresholds,        # list/array of Pi predictions
        'accuracies': accuracies,         # list of file paths or names
        'offloaded_count': offloaded_count,
        'offloaded_ratio': offloaded_ratio  
    })

    return acc_df

def read_csv(pi_csv_path, srv_csv_path):

    pi_df = pd.read_csv(pi_csv_path)
    srv_df = pd.read_csv(srv_csv_path)

    pi_df  = pi_df.rename(columns={'prediction':'pi_label', 'confidence':'pi_conf', 'ground_truth': 'true_label_rpi', 'correct': 'correct_rpi'})
    srv_df = srv_df.rename(columns={'true_label': 'true_label_serv', 'predicted_label':'srv_label','confidence':'srv_conf', 'correct': 'correct_serv'})

    df = pi_df.merge(srv_df, on='filename', how='inner')
    #print(df)

    expected = ['filename','pi_label','pi_conf','srv_label','srv_conf','true_label_serv', 'true_label_rpi']
    
    missing = [c for c in expected if c not in df.columns]
    
    if missing:
        raise ValueError(f"Missing columns after merge: {missing}")
    #print(df)
    return df
def read_csv_st(pi_csv_path, srv_csv_path):

    pi_df = pd.read_csv(pi_csv_path)
    srv_df = pd.read_csv(srv_csv_path)

    pi_df  = pi_df.rename(columns={'prediction':'pi_label', 'confidence':'pi_conf', 'ground_truth': 'true_label_rpi', 'correct': 'correct_rpi'})
    srv_df = srv_df.rename(columns={'true_label': 'true_label_serv', 'predicted_label':'srv_label','confidence':'srv_conf', 'correct': 'correct_serv'})

    df = pi_df.merge(srv_df, on='filename', how='inner')
    # 1) Convert Pi’s labels to int (they are already int, but do it explicitly)
    df['pi_label']       = df['pi_label'].astype(int)
    df['true_label_rpi'] = df['true_label_rpi'].astype(int)

    # 2) Convert Server’s string labels → int using your mapping
    label_map = {
        'broken':  0,
        'healthy': 1
    }
    df['srv_label']       = df['srv_label'].map(label_map).astype(int)
    df['true_label_serv'] = df['true_label_serv'].map(label_map).astype(int)

    # 3) (Optional) Check that Pi’s and Server’s ground truths agree
    mismatch = (df['true_label_rpi'] != df['true_label_serv']).any()
    if mismatch:
        raise ValueError("Ground‐truth encoding differs between Pi CSV and Server CSV!")
    #print(df)

    expected = ['filename','pi_label','pi_conf','srv_label','srv_conf','true_label_serv', 'true_label_rpi']
    
    missing = [c for c in expected if c not in df.columns]
    
    if missing:
        raise ValueError(f"Missing columns after merge: {missing}")
    #print(df)
    return df

def main(id):
    DIRNAME = './' #'/home/alema416/dev/work/HI4Lines_Insp/reports/figures/HAILO/new_resnet/fmfp/data'
    results = {}
    device = 'rpi'
    model = 'idk'
    
    for set in ['train', 'val', 'test']:
        df = read_csv_st(os.path.join(DIRNAME, f'per_sample_{set}_{id}.csv'), f'./L-ML_per_sample_{set}.csv')
        accs = coop_accuracy(df['true_label_rpi'], df['pi_label'], df['pi_conf'], df['srv_label'])
        results[f'{device}_{model}_{set}'] = accs
    
    for i in results:
        #print(i)
        #print(i.split('_')[-1])
        spl = i.split('_')[-1] #input('split: ')
        print(id)
        print(spl)
        print(results[i])
        results[i].to_csv(f'coop_{spl}_{id}.csv', sep='\t')
    '''
    results = {}
    for device in ['rpi', 'st']:
        for model in [0, 1, 2]:
            for set in ['train', 'val', 'test']:
                df = read_csv(f'./{device}_results/{model}/per_sample_{set}.csv', f'./L-ML_per_sample_{set}.csv')
                accs = coop_accuracy(df['true_label_rpi'], df['pi_label'], df['pi_conf'], df['srv_label'])
                results[f'{device}_{model}_{set}'] = accs
    print(results)
    '''

if __name__ == '__main__':
    for id in [22, 46]:
         main(id)
