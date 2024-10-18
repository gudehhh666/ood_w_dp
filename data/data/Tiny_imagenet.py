import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_from_disk
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.datasets import ImageFolder
# Train Dataset Mean: tensor([0.4802, 0.4481, 0.3975])
# Train Dataset Std: tensor([0.2302, 0.2265, 0.2262])
# Test Dataset Mean: tensor([0.4824, 0.4495, 0.3981])
# Test Dataset Std: tensor([0.2301, 0.2264, 0.2261])


class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.data = dataset['image']
        self.label = dataset['label']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.label[idx]
        
        # 将图像转换为RGB格式
        image = image.convert('RGB') if image.mode != 'RGB' else image
        
        if self.transform:
            image = self.transform(image)
        return image, label

def pil_loader(path: str):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
    
class TinyNoisy(Dataset):
    def __init__(self, root_dir, transform, **kwargs):
        # super().__init__(loader=pil_loader, **kwargs)
        self.data = np.load(root_dir['data'])
        self.target = np.load(root_dir['target'])
        self.transform = transform

    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        sample = self.data[index]
        target = self.target[index]
        # try:
        #     sample = sample.convert('RGB') if sample.mode != 'RGB' else sample
        # except Exception as e:
        #     print(e)

        sample = Image.fromarray(sample)
        
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target




# 从磁盘加载Tiny ImageNet数据集

def get_tiny(root_dir, test_only: bool = False, return_set = True, batch_size= 64, **kwargs):
    
    dataset = load_from_disk(root_dir['train'])
    test_dataset = load_from_disk(root_dir['test'])

    # mean = [0.4802, 0.4481, 0.3975]
    # std = [0.2302, 0.2265, 0.2262]
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        # transforms.Resize(64, padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std),
    ])
    transform_test = transforms.Compose([
        # transforms.Resize(int(64/0.875)),
        # transforms.CenterCrop(64),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std),
    ])

    # 创建自定义数据集
    # print(len(train_dataset), len(test_dataset))

    # 定义DataLoader
    if test_only:
        test_dataset = CustomDataset(test_dataset, transform=transform_test)
        test_loader = DataLoader(test_dataset, shuffle= False, batch_size=batch_size, **kwargs)

        if return_set:
            return test_loader, test_dataset
        else:
            return test_loader
    else:
        train_dataset = CustomDataset(dataset, transform=transform_train)
        test_dataset = CustomDataset(test_dataset, transform=transform_test)
        train_loader = DataLoader(train_dataset, shuffle= True, batch_size=batch_size, **kwargs)
        test_loader = DataLoader(test_dataset, shuffle= False, batch_size=batch_size, **kwargs)
        if return_set:
            return train_loader, test_loader, train_dataset, test_dataset
        else:
            return train_loader, test_loader


# root_dir = {'data': r"D:\CASIA\Undergraduate thesis\codes\code_cc_misd\tiny_img_noisy_2.npy", 
#             'target': r"D:\CASIA\Undergraduate thesis\codes\code_cc_misd\tiny_target_noisy_2.npy"}

def get_tiny_mislabel(root_dir, return_set: True, batch_size = 64,  **kwargs):
    mean = [0.4802, 0.4481, 0.3975]
    std = [0.2302, 0.2265, 0.2262]
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std),
    ])

    train_set = TinyNoisy(root_dir, transform = transform_train)
    train_loader = DataLoader(train_set, shuffle= True, batch_size=batch_size, **kwargs)
    if return_set:
        return train_loader, train_set
    else:
        return train_loader

# if __name__ == "__main__":
#     root_dir = {'data': r"D:\CASIA\Undergraduate thesis\codes\code_cc_misd\tiny_img_noisy_2.npy", 
#             'target': r"D:\CASIA\Undergraduate thesis\codes\code_cc_misd\tiny_target_noisy_2.npy"}
#     tiny_dir = {'train': r"E:\dataset\data\tiny-imagenet\train",
#                 'test': r"E:\dataset\data\tiny-imagenet\valid"}
#     # train_loader, test_loader = get_tiny(root_dir, batch_size=32)
#     kwargs = {'batch_size': 64, 'num_workers': 0, 'pin_memory': True}
#     train_loader = get_tiny_mislabel(root_dir, **kwargs)
#     test_loader = get_tiny(tiny_dir, test_only=True, **kwargs)
#     print(len(train_loader.dataset))
#     print(len(test_loader.dataset))
#     for i, (img, target) in enumerate(train_loader):
#         print(img.shape)
#         print(target)
#         if i == 0:
#             break
#     for i, (img, target) in enumerate(test_loader):
#         print(img.shape)
#         print(target)
#         if i == 0:
#             break
    # print(len(test_loader.dataset))
    # print(train_loader.dataset[0])














# # 计算均值和标准差
# def calculate_mean_std(loader):
#     mean = 0.0
#     std = 0.0
#     total_images_count = 0
#     for images, _ in loader:
#         images = images.view(images.size(0), images.size(1), -1)
#         mean += images.mean(2).sum(0)
#         std += images.std(2).sum(0)
#         total_images_count += images.size(0)
#     mean /= total_images_count
#     std /= total_images_count
#     return mean, std

# train_mean, train_std = calculate_mean_std(train_loader)
# test_mean, test_std = calculate_mean_std(test_loader)

# print(f"Train Dataset Mean: {train_mean}")
# print(f"Train Dataset Std: {train_std}")
# print(f"Test Dataset Mean: {test_mean}")
# print(f"Test Dataset Std: {test_std}")
