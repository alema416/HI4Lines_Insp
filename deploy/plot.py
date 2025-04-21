import pandas as pd
import matplotlib.pyplot as plt

for split in ['train', 'val', 'test']:
    # Load the confidence and correctness data from files
    confidence_file = f'confs_{split}.txt'  # Update with your confidence file path
    correctness_file = f'labels_{split}.txt'  # Update with your correctness file path
    
    # Read the data
    confidence_data = pd.read_csv(confidence_file, header=None, names=['confidence'])
    correctness_data = pd.read_csv(correctness_file, header=None, names=['correct'])
    
    # Combine the two datasets
    data = pd.concat([confidence_data, correctness_data], axis=1)
    
    # Separate data into correct and error categories
    correct_confidence = data[data['correct'] == 1]['confidence']
    error_confidence = data[data['correct'] == 0]['confidence']
    
    # Create the density plot
    plt.figure(figsize=(10, 6))
    plt.hist(correct_confidence, bins=300, density=False, color='green', alpha=0.5, label='Correct', edgecolor='black')
    plt.hist(error_confidence, bins=300, density=False, color='red', alpha=0.5, label='Error', edgecolor='black')
    
    # Add labels and title
    plt.title(f'Confidence Plot for Correct vs Error for {split} set', fontsize=16)
    plt.xlabel('Confidence', fontsize=14)
    plt.ylabel('Samples', fontsize=14)
    plt.legend()
    
    # Show the plot
    plt.savefig(f'conf_plot_{split}.png')
