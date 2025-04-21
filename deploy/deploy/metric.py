from nmetr import AUGRC
import argparse
import torch

for split in ['train', 'val', 'test']:
    with open(f'labels_{split}.txt', "r") as file:
      labels = [int(line.strip()) for line in file]  # Using strip() to remove newline characters
    with open(f'confs_{split}.txt', "r") as file:
      confs = [float(line.strip()) for line in file]  # Using strip() to remove newline characters
    
    probs = torch.tensor(confs, dtype=torch.float32)  # Now shape (N, C)
    numeric_labels_tensor = torch.tensor(labels, dtype=torch.long)
    augrc_metric = AUGRC()
    augrc_metric.update(probs, numeric_labels_tensor)
    augrc_value = augrc_metric.compute()
    print(f'SPECIAL_PRINTaugrc{split} {1000*augrc_value.item()}')