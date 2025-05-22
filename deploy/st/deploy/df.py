import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd

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
    # 4. Plot the KDE curve
    plt.figure()
    plt.fill_between(xs, kde(xs), alpha=0.4, color='green')    # fill under the curve
    plt.plot(xs, kde(xs), linewidth=2, color='green')
    plt.fill_between(xsn, kden(xsn), alpha=0.4, color='red')
    plt.plot(xsn, kden(xsn), linewidth=2, color='red')

    plt.xlabel('confidence')
    plt.ylabel('density')
    modelname = 'baseline' if ide=='b_' else 'fmfp'
    plt.title(f'{modelname} - model {id} - {split} set')
    plt.savefig(f'custom_{ide}{id}_{split}.png')
for id in [32, 49, 56]:
    for split in ['train', 'val', 'test']:
        for ide in ['', 'b_']:
            # Replace 'confidences.txt' and 'labels.txt' with your actual file paths
            confidences = pd.read_csv(f'{ide}confs_{id}_{split}.txt', header=None, names=['confidence'])
            labels = pd.read_csv(f'{ide}labels_{id}_{split}.txt', header=None, names=['correct'])

            # Combine into a single DataFrame
            df = pd.concat([confidences, labels], axis=1)

            df_zero = df[df['correct'] == 0]
            df_one = df[df['correct'] == 1]
            custom(df_one, df_zero, id, split, ide)
            df_zero.to_csv(f'output_{ide}{id}_{split}_error.csv', index=False)
            df_one.to_csv(f'output_{ide}{id}_{split}_success.csv', index=False)

