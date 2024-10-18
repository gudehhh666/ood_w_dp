from typing import Any, Callable, Optional, Tuple
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import pickle
import torchvision
import torchvision.transforms as transforms
from models import resnet
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset, ImageFolder

from torchvision.datasets.cifar import CIFAR10, CIFAR100

class CIFAR10_loss(CIFAR10):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        loss_dir : str = None,
        cut_rate: float = 0.0,
        shift : int = 0
    ) -> None:

        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        
        self.losses = None
        if loss_dir is not None:
            self.losses = []
            with open(loss_dir, 'rb') as f:
                self.losses = pickle.load(f)
        
            if cut_rate > 0 and self.losses is not None:
                # cut the loss by the value of loss, drop the least loss
                print('use cut_rate: ', cut_rate)
                loss_sort = np.sort(self.losses)
                print('score_sort', loss_sort[0], loss_sort[-1])
                cut_index = int(len(loss_sort) * cut_rate)
                if shift == 0:
                    self.save_index = [self.losses.index(value) for value in loss_sort[cut_index:]]
                else:
                    self.save_index = [self.losses.index(value) for value in loss_sort[cut_index - shift: - shift]]
                self._refresh_data()
                
    def _refresh_data(self):
        self.data = [self.data[i] for i in self.save_index]
        self.targets = [self.targets[i] for i in self.save_index]
        # if self.losses is not None:
        self.losses = [self.losses[i] for i in self.save_index]
        
        print('refresh data', len(self.data), len(self.targets), len(self.losses))
                
            
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, loss) where target is index of the target class.
        """
        
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)
        # img.save('./temp.jpg')
        # print('save img', img.size, img.mode, target)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        # print('target', target)
        if self.target_transform is not None:
            target = self.target_transform(target)
            # print('target', target)
        if self.losses is not None:
            loss = self.losses[index]

            return img, target
        else:
            return img, target
        
        
class CIFAR100_loss(CIFAR100):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        loss_dir : str = None,
        cut_rate: float = 0.0,
        shift: int = 0
    ) -> None:

        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        
        self.losses = None
        if loss_dir is not None:
            self.losses = []
            with open(loss_dir, 'rb') as f:
                self.losses = pickle.load(f)
        
            if cut_rate > 0 and self.losses is not None:
                # cut the loss by the value of loss, drop the least loss
                print('use cut_rate: ', cut_rate)
                loss_sort_idx = np.argsort(self.losses)
                print('score_sort', self.losses[loss_sort_idx[0]], self.losses[loss_sort_idx[-1]])
                save_n = int(len(loss_sort_idx) * (1- cut_rate))
                self.save_index = loss_sort_idx[-save_n:]
                print('save_index with highest', len(self.save_index), self.losses[self.save_index[0]], self.losses[self.save_index[-1]])
                self._refresh_data()
                
    def _refresh_data(self):
        self.data = [self.data[i] for i in self.save_index]
        self.targets = [self.targets[i] for i in self.save_index]
        # if self.losses is not None:
        self.losses = [self.losses[i] for i in self.save_index]
        
        print('refresh data', len(self.data), len(self.targets), len(self.losses))
                
            
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, loss) where target is index of the target class.
        """
        
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)
        # img.save('./temp.jpg')
        # print('save img', img.size, img.mode, target)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        # print('target', target)
        if self.target_transform is not None:
            target = self.target_transform(target)
            # print('target', target)
        if self.losses is not None:
            loss = self.losses[index]

            return img, target
        else:
            return img, target




class CIFAR10_loss_ablation(CIFAR10):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        loss_dir : str = None,
        cut_rate: float = 0.0,
        mod: str = 'highest'
    ) -> None:

        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        
        self.losses = None
        if loss_dir is not None:
            self.losses = []
            with open(loss_dir, 'rb') as f:
                self.losses = pickle.load(f)
        
            if cut_rate > 0 and self.losses is not None:
                # cut the loss by the value of loss, drop the least loss
                print('use cut_rate: ', cut_rate)
                loss_sort_idx = np.argsort(self.losses)
                print('score_sort', self.losses[loss_sort_idx[0]], self.losses[loss_sort_idx[-1]])
                save_rate = 1 - cut_rate
                save_n = int(len(loss_sort_idx) * (save_rate))
                if mod == 'highest':
                    self.save_index = loss_sort_idx[-save_n:]
                    print('save_index with highest', len(self.save_index))
                if mod == 'highest_class':
                    self.save_index = []
                    label_list = np.array(self.targets)
                    unique_labels = np.unique(self.targets)
                    print(unique_labels.size)
                    for label in unique_labels:
                        label_index = np.where(label_list == label)[0]
                        # print(label_index.size)
                        losses = np.array(self.losses)[label_index]
                        lower_quantile = np.percentile(losses, cut_rate*100)
                        middle_indices = label_index[(losses >= lower_quantile)]
                        self.save_index.extend(middle_indices)
                    print(len(self.save_index))
                if mod == 'lowest':
                    self.save_index = loss_sort_idx[:save_n]
                    print('save_index with lowest', len(self.save_index))
                if mod == 'middle':
                    self.save_index = loss_sort_idx[len(loss_sort_idx)//2 - save_n//2: len(loss_sort_idx)//2 + save_n//2]
                    print('save_index with middle', len(self.save_index))
                if mod == 'middle_class':
                    self.save_index = []
                    label_list = np.array(self.targets)
                    unique_labels = np.unique(self.targets)
                    print(unique_labels.size)
                    for label in unique_labels:
                        label_index = np.where(label_list == label)[0]
                        # print(label_index.size)
                        losses = np.array(self.losses)[label_index]
                        lower_quantile = np.percentile(losses, (0.5 - save_rate/2)*100)
                        # print(lower_quantile)
                        upper_quantile = np.percentile(losses, (0.5 + save_rate/2)*100)
                        # print(upper_quantile)
                        middle_indices = label_index[(losses >= lower_quantile) & (losses <= upper_quantile)]
                        self.save_index.extend(middle_indices)
                    print(len(self.save_index))
                        # exit()
                        

                if mod == 'random':
                    self.save_index = np.random.choice(loss_sort_idx, save_n, replace=False)
                    print('save_index with random', len(self.save_index))
                if mod == 'two_side':
                    index1 = loss_sort_idx[:save_n//2]
                    index2 = loss_sort_idx[-save_n//2:]
                    self.save_index = np.concatenate([index1, index2])
                    print('save_index with two_side', len(self.save_index))
                
                if mod == 'uniform':
                    idx_list = np.round(np.linspace(0, len(loss_sort_idx)-1, save_n)).astype(int)
                    self.save_index = loss_sort_idx[idx_list]
                    # print('save_index with uniform', idx_list)
                    print('save_index with uniform', len(self.save_index), save_n)
                                
                self._refresh_data()
                
    def _refresh_data(self):
        self.data = [self.data[i] for i in self.save_index]
        self.targets = [self.targets[i] for i in self.save_index]
        # if self.losses is not None:
        self.losses = [self.losses[i] for i in self.save_index]
        
        print('refresh data', len(self.data), len(self.targets), len(self.losses))
                
            
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, loss) where target is index of the target class.
        """
        
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)
        # img.save('./temp.jpg')
        # print('save img', img.size, img.mode, target)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        # print('target', target)
        if self.target_transform is not None:
            target = self.target_transform(target)
            # print('target', target)
        if self.losses is not None:
            loss = self.losses[index]

            return img, target
        else:
            return img, target
    



class CIFAR100_loss_ablation(CIFAR100):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        loss_dir : str = None,
        cut_rate: float = 0.0,
        mod: str = 'highest'
    ) -> None:

        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        
        self.losses = None
        if loss_dir is not None:
            self.losses = []
            with open(loss_dir, 'rb') as f:
                self.losses = pickle.load(f)
        
            if cut_rate > 0 and self.losses is not None:
                # cut the loss by the value of loss, drop the least loss
                print('use cut_rate: ', cut_rate)
                loss_sort_idx = np.argsort(self.losses)
                print('score_sort', self.losses[loss_sort_idx[0]], self.losses[loss_sort_idx[-1]])
                save_rate = 1 - cut_rate
                save_n = int(len(loss_sort_idx) * (save_rate))
                if mod == 'highest':
                    self.save_index = loss_sort_idx[-save_n:]
                    print('save_index with highest', len(self.save_index))
                if mod == 'highest_class':
                    self.save_index = []
                    label_list = np.array(self.targets)
                    unique_labels = np.unique(self.targets)
                    print(unique_labels.size)
                    for label in unique_labels:
                        label_index = np.where(label_list == label)[0]
                        # print(label_index.size)
                        losses = np.array(self.losses)[label_index]
                        lower_quantile = np.percentile(losses, cut_rate*100)
                        middle_indices = label_index[(losses >= lower_quantile)]
                        self.save_index.extend(middle_indices)
                    print(len(self.save_index))
                if mod == 'lowest':
                    self.save_index = loss_sort_idx[:save_n]
                    print('save_index with lowest', len(self.save_index))
                if mod == 'middle':
                    self.save_index = loss_sort_idx[len(loss_sort_idx)//2 - save_n//2: len(loss_sort_idx)//2 + save_n//2]
                    print('save_index with middle', len(self.save_index))
                if mod == 'middle_class':
                    self.save_index = []
                    label_list = np.array(self.targets)
                    unique_labels = np.unique(self.targets)
                    print(unique_labels.size)
                    for label in unique_labels:
                        label_index = np.where(label_list == label)[0]
                        # print(label_index.size)
                        losses = np.array(self.losses)[label_index]
                        lower_quantile = np.percentile(losses, (0.5 - save_rate/2)*100)
                        # print(lower_quantile)
                        upper_quantile = np.percentile(losses, (0.5 + save_rate/2)*100)
                        # print(upper_quantile)
                        middle_indices = label_index[(losses >= lower_quantile) & (losses <= upper_quantile)]
                        self.save_index.extend(middle_indices)
                    print(len(self.save_index))
                        # exit()
                        

                if mod == 'random':
                    self.save_index = np.random.choice(loss_sort_idx, save_n, replace=False)
                    print('save_index with random', len(self.save_index))
                if mod == 'two_side':
                    index1 = loss_sort_idx[:save_n//2]
                    index2 = loss_sort_idx[-save_n//2:]
                    self.save_index = np.concatenate([index1, index2])
                    print('save_index with two_side', len(self.save_index))
                
                if mod == 'uniform':
                    idx_list = np.round(np.linspace(0, len(loss_sort_idx)-1, save_n)).astype(int)
                    self.save_index = loss_sort_idx[idx_list]
                    # print('save_index with uniform', idx_list)
                    print('save_index with uniform', len(self.save_index), save_n)
                                
                self._refresh_data()
                
    def _refresh_data(self):
        self.data = [self.data[i] for i in self.save_index]
        self.targets = [self.targets[i] for i in self.save_index]
        # if self.losses is not None:
        self.losses = [self.losses[i] for i in self.save_index]
        
        print('refresh data', len(self.data), len(self.targets), len(self.losses))
                
            
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, loss) where target is index of the target class.
        """
        
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)
        # img.save('./temp.jpg')
        # print('save img', img.size, img.mode, target)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        # print('target', target)
        if self.target_transform is not None:
            target = self.target_transform(target)
            # print('target', target)
        if self.losses is not None:
            loss = self.losses[index]

            return img, target
        else:
            return img, target
    
     
        
class Dataset_ood(Dataset):
    def __init__(self, root, transforms) -> None:
        super().__init__()
        self.root = root
        self.tarnsforms = transforms
        
        self.img_list = os.listdir(self.root)
    
    def __getitem__(self, index) -> Any:
        
        
        return 
