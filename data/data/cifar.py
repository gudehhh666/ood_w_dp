import torch
import torchvision
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10
import torchvision.transforms as trn


class CIFAR100Noisy(CIFAR100):
    def __init__(self, root = r'D:\CASIA\Undergraduate thesis\codes\code_cc_misd\data', **kwargs):
        super().__init__(root=root, **kwargs)
        label_path = os.path.join(root, "noisy/cifar.npy")
        self.targets = np.load(label_path)
        print("Load noisy label from", label_path)
        
        

def get_cifar10(root, test_only = False, return_set = True, batch_size = 64,  **kwargs):
    
    # transforms
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    transform_train = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                                trn.ToTensor(), trn.Normalize(mean, std)])
    transform_test = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
    
    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(trainset, shuffle=True, batch_size=batch_size,  **kwargs)
    test_loader = DataLoader(testset, shuffle=False, batch_size=batch_size, **kwargs)
    if test_only:
        if return_set:
            return test_loader, testset
        else:
            return test_loader
    else:
        if return_set:
            return train_loader, test_loader, trainset, testset
        else:
            return train_loader, test_loader


def get_cifar100(root, test_only = False, return_set = True, batch_size = 64,  **kwargs):
    # transforms
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    transform_train = trn.Compose([trn.RandomHorizontalFlip(), 
                                    trn.RandomCrop(32, padding=4),
                                    trn.ToTensor(), 
                                    trn.Normalize(mean, std)])
    transform_test = trn.Compose([trn.ToTensor(), 
                                    trn.Normalize(mean, std)])
    
    trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(trainset, shuffle=True, batch_size=batch_size,  **kwargs)
    test_loader = DataLoader(testset, shuffle=False, batch_size=batch_size, **kwargs)
    if test_only:
        if return_set:
            return test_loader, testset
        else:
            return test_loader
    else:
        if return_set:
            return train_loader, test_loader, trainset, testset
        else:
            return train_loader, test_loader

def get_cifar100_mislabel(root, return_set = True, batch_size = 64, **kwargs):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    transform_train = trn.Compose([trn.RandomHorizontalFlip(), 
                                    trn.RandomCrop(32, padding=4),
                                    trn.ToTensor(), 
                                    trn.Normalize(mean, std)])

    train_set = CIFAR100Noisy(root, train=True, transform = transform_train, download= True)
    train_loader = DataLoader(train_set, shuffle= True, batch_size=batch_size, **kwargs)
    
    if return_set:
        return train_loader, train_set
    else:
        return train_loader