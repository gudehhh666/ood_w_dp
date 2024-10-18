import sys
sys.path.append("..")

import torchvision
import os
import numpy as np
from torchvision.datasets import CIFAR100

# mislabeled data rate
mislabel_rate = 0.2

trainset = CIFAR100('/root/dataset/cifar100', train=True, download=True, transform=torchvision.transforms.ToTensor())
noisy_targets = np.zeros_like(trainset.targets)

for i in range(50000):
    p = np.random.random()
    if p > 1 - mislabel_rate:
        # add random noise to label
        probs = np.ones((100,), dtype=float) / 99
        probs[trainset.targets[i]] = 0
        noisy_targets[i] = np.random.choice(100, p=probs)
    else:
        noisy_targets[i] = trainset.targets[i]
    
print("Saving...")
os.makedirs("./noisy/", exist_ok=True)
np.save("./noisy/cifar", noisy_targets)

class CIFAR100Noisy(CIFAR100):
    def __init__(self, root = r'D:\CASIA\Undergraduate thesis\codes\code_cc_misd\data', **kwargs):
        super().__init__(root=root, **kwargs)
        label_path = os.path.join(root, "noisy/cifar.npy")
        self.targets = np.load(label_path)
        print("Load noisy label from", label_path)