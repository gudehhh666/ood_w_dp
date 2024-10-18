import torch
import numpy as np
import torch.utils
from data.dataset import CIFAR100_loss_ablation
import argparse
import torchvision.transforms as trn
import os
from models.wrn import WideResNet
from collections import defaultdict
import torchvision.datasets as datasets

parser = argparse.ArgumentParser(description='PyTorch CIFAR Adversarial Training')
model_options = ['res18', 'res50', 'wrn', 'allconv']
dataset_options = ['cifar10', 'cifar100', 'tiny-imagenet']
parser.add_argument('--dataset', '-d', default='cifar100', choices=dataset_options)
parser.add_argument('--model', '-a', default='wrn', choices=model_options)
parser.add_argument('--mod', default='middle_class', choices=['highest', 'lowest', 'middle', 'random', 'two_side', 'uniform', 'middle_class'],help='model directory')
parser.add_argument('--rate', default=0.5, type=float, help='cut rate')
# parser.add_argument('')
parser.add_argument('--load', default='/lustre/home/xmwang/codes/code_cc_misd/checkpoint/wrn-cifar100/EL2N_abl_r0.5/middle/model_200_EL2N_abl_train_r0.5.pth', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

# wrn
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')



args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# get dataset

# 用在dataloader的时候加快进程 only!
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# transforms
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
transform_train = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])
transform_test = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

loss_dir = os.path.join('/lustre/home/xmwang/share/data/cifar100', args.model, 'EL2N' +'.pkl')
print('loss_dir', loss_dir)

trainset = CIFAR100_loss_ablation(root='/lustre/home/xmwang/share/data/cifar100', train=True, download=True, transform=transform_train, loss_dir=loss_dir, cut_rate=args.rate, mod=args.mod)
num_classes = 100
train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, **kwargs)
testset = datasets.CIFAR100(root='/lustre/home/xmwang/share/data/cifar100', train=False, download=True, transform=transform_test)

test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, **kwargs)
# get model
if args.model =='wrn':
    model = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate).to(device)

if args.load:
    load_dict = torch.load(args.load, map_location=device)
    model.load_state_dict(load_dict)
    print('load the model successfully')
    

print('---------count the num and acc:')

class_correct = [0] * num_classes  # 初始化每个类别的正确预测次数
class_total_train = [0] * num_classes    # 初始化每个类别的总样本数
class_total = [0] * num_classes

model.eval()
with torch.no_grad():
    for idx, (input, target) in enumerate(train_loader):
        # input = input.to(device)
        # target = target.to(device)
        # output = model(input)
        # pred = output.data.max(1, keepdim=True)[1]
        
        for i in range(len(target)):
                    label = target[i]
                    # prediction = pred[i]
                    # if label == prediction:
                    #     class_correct[label] += 1
                    class_total_train[label] += 1    

model.eval()
with torch.no_grad():
    for idx, (input, target) in enumerate(test_loader):
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        pred = output.data.max(1, keepdim=True)[1]
        
        for i in range(len(target)):
                    label = target[i]
                    prediction = pred[i]
                    if label == prediction:
                        class_correct[label] += 1
                    class_total[label] += 1

class_accuracy = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)]

print('class_accuracy:', class_accuracy)
# print('class_total:', class_total)
print('class_total_train:', class_total_train)

import statistics

mean_value = statistics.mean(class_accuracy)
variance_value = statistics.variance(class_accuracy)
stdev = statistics.stdev(class_accuracy)
print("Mean and Variance of class_accuracy", mean_value, variance_value, stdev)
mean_value = statistics.mean(class_total_train)
stdev = statistics.stdev(class_total_train)
variance_value = statistics.variance(class_total_train)

print("Mean and Variance of class_total_train", mean_value, variance_value, stdev)
# import pickle
# with open(f'./EL2N_{args.mod}_{args.rate}_class_accuracy.pkl', 'wb') as f:
#     pickle.dump((class_accuracy, class_total_train), f)
    