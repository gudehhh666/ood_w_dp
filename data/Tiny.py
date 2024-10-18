
import numpy as np
import torch
from bisect import bisect_left
import torchvision.transforms as transforms


class RandomImages(torch.utils.data.Dataset):

    def __init__(self, transform=None, load=None):
        
        if load:
            self.load_dir = load
        else:
            self.load_dir = 'C:\Users\\10520\\Downloads\\300K_random_images.npy'
        
        self.data = np.load(self.load_dir)

        self.transform = transform
        self.basic_transform = transforms.Compose([transforms.ToPILImage(), 
                                                transforms.Resize((32, 32))])


        

    def __getitem__(self, index):
        
        img = self.data[index]
        img = self.basic_transform(img)
        if self.transform is not None:
            img = self.transform(img)


        return img, 0  # 0 is the class

    def __len__(self):
        return self.data.shape[0]

