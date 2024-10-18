import sys
sys.path.append("..")

import torchvision
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_from_disk

class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.image = self.dataset['image']
        self.targets = self.dataset['label']
        # print(len(self.image))
        # print(len(self.targets))
        # 将图像转换为RGB格式
        
        # image = image.convert('RGB') if image.mode != 'RGB' else image for image in self.image

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.image[idx]
        image = image.convert('RGB') if image.mode != 'RGB' else image
        label = self.targets[idx]
        
        if self.transform:
            image = self.transform(image)
        return image, label


# mislabel data ratio
mislabel_ratio = 0.2

root_dir = {'train': r"E:\dataset\data\tiny-imagenet\train"}
trainset = load_from_disk(root_dir['train'])

trainset = CustomDataset(trainset, transform=torchvision.transforms.ToTensor())


# trainset = ImageFolder("../data/tiny-imagenet-200/train", transform=torchvision.transforms.ToTensor())
noisy_targets = np.zeros_like(trainset.targets)
trainloader = DataLoader(trainset, batch_size=32, shuffle=False, pin_memory=True, num_workers=0)

imgs, targets = [], []

for img, label in tqdm(trainloader):
    # print(img.shape)
    # print(label.shape)
    for i in range(label.shape[0]):
        p = np.random.rand()
        if p > 1 - mislabel_ratio:
            probs = np.ones((200,), dtype=float) / 199
            probs[label[i]] = 0
            label[i] = np.random.choice(200, p=probs)
    imgs.append((img.permute((0, 2, 3, 1)) * 255.0).type(torch.ByteTensor).numpy())
    targets.append(label.numpy())
imgs = np.concatenate(imgs)
targets = np.concatenate(targets)

print("Saving...")
np.save("./tiny_img_noisy_2", imgs)
np.save("./tiny_target_noisy_2", targets)