#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def load_data(labels_path, confs_path):
    """Load labels (ints) and confidences (floats) from text files."""
    try:
        labels = np.loadtxt(labels_path, dtype=int)
        confs  = np.loadtxt(confs_path, dtype=float)
    except Exception as e:
        sys.exit(f"Error loading files: {e}")
    if labels.shape != confs.shape:
        sys.exit("Error: labels and confidences must have the same number of entries.")
    return labels, confs
ide = 'b_'
def plot_density(id, split, labels, confs, bins=30):
    """Plot density histograms of confidences for correct vs wrong."""
    correct = confs[labels == 1]
    wrong   = confs[labels == 0]

    # shared bins from 0 to 1
    bins = np.linspace(0, 1, bins + 1)

    plt.figure(figsize=(8, 5))
    plt.hist(correct, bins=bins, density=True,
             color='green', alpha=0.6, label='Correct')
    plt.hist(wrong, bins=bins, density=True,
             color='red',   alpha=0.6, label='Wrong')
    plt.xlim([0.5, 1])
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.title('Confidence Density Plot')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plot_{ide}{id}_{split}.png')

def main():
    '''
    parser = argparse.ArgumentParser(
        description="Plot confidence density for correct vs wrong classifications.")
    parser.add_argument('labels', help="Path to labels.txt (lines of 1 or 0)")
    parser.add_argument('confs',  help="Path to confs.txt (lines of floats 0–1)")
    parser.add_argument('-b','--bins', type=int, default=30,
                        help="Number of histogram bins (default: 30)")
    args = parser.parse_args()
    '''
    for id in [32, 49, 56]:
        for split in ['train', 'val', 'test']:    
            labels, confs = load_data(f'{ide}labels_{id}_{split}.txt', f'{ide}confs_{id}_{split}.txt')
            plot_density(id, split, labels, confs, bins=100)

if __name__ == '__main__':
    main()
