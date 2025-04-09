import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
class ImageFolderWithIndex(ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        #return sample, target, index
        path, _ = self.samples[index]
        # Get just the file name
        file_name = os.path.basename(path)
        return sample, target, index, file_name
def get_loader_local(root, batch_size, input_size):

    mean = [0.50463295, 0.46120012, 0.4291694 ]
    stdv = [0.18364702, 0.1885083,  0.19882548]

    train_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomCrop(input_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    
    train_set = ImageFolderWithIndex(root=os.path.join(root, 'train'), transform=train_transforms)

    #half_length = int(len(train_set) / 4)
    #create a subset containing only the first half of the indices
    #train_set = Subset(train_set, list(range(half_length)))

    valid_set = ImageFolderWithIndex(root=os.path.join(root, 'val'), transform=val_test_transforms)
    test_set  = ImageFolderWithIndex(root=os.path.join(root, 'test'), transform=val_test_transforms)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    return train_loader, valid_loader, test_loader#, test_onehot, test_labels       
    

def one_hot_encoding(labels, num_classes):
    import numpy as np
    one_hot = np.eye(num_classes)[labels]
    return one_hot
