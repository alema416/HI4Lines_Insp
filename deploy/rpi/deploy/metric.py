from nmetr import AUGRC
import argparse
import torch
ide = ''
id = input('id: ')
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
