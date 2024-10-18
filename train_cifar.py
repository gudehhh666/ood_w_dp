from __future__ import print_function
from cmath import inf
import os
import argparse
import logging
import time
import numpy as np
# import xlwt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import csv

from models.resnet import *
from models.wideresnet import *
from data.infobatch import InfoBatch
import metrics

# define the parser for the input arguments
parser = argparse.ArgumentParser(description='PyTorch CIFAR Adversarial Training')
model_options = ['resnet18', 'wrn-28-10', 'wrn-40-2']
dataset_options = ['cifar10', 'cifar100', 'tiny-imagenet']
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N', help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W')
parser.add_argument('--lr_max', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--model-dir', default='', help='directory of model for saving checkpoint')
parser.add_argument('--data-dir', default='', help='directory of dataset')
parser.add_argument('--save-freq', '-s', default=100, type=int, metavar='N', help='save frequency')

parser.add_argument('--dataset', '-d', default='cifar10', choices=dataset_options)
parser.add_argument('--model', '-a', default='resnet18', choices=model_options)

args = parser.parse_args()
# print(args)


def append_to_csv_file(dict_data, file_path, fieldnames):
            write_header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
            with open(file_path, 'a', newline='') as file:
                
                
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader() if write_header else None
                # No need to write the header each time; just append rows
                writer.writerow(dict_data)
                print("Metrics data saved to", file_path)

# settings
# model_dir = args.model_dir
# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)

use_cuda = not args.no_cuda and torch.cuda.is_available()
# torch.manual_seed(args.seed)
device = torch.device("cuda:0" if use_cuda else "cpu")

# 用在dataloader的时候加快进程 only!
kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

# setup data loader
# transforms.RandomCrop(32, padding=4)：随机裁剪为32*32，填充为4
# transforms.RandomHorizontalFlip()：随机水平翻转
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
transform_train = transforms.Compose([transforms.RandomHorizontalFlip(), 
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.ToTensor(), 
                                        transforms.Normalize(mean, std)])
transform_test = transforms.Compose([transforms.ToTensor(), 
                                         transforms.Normalize(mean, std)])

if args.dataset == 'cifar10':
    args.data_dir = '/lustre/home/xmwang/share/data/cifar10'
    model_dir = '/lustre/home/xmwang/codes/code_cc_misd/checkpoint/res18-cifar10'
    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    num_classes = 10
    
elif args.dataset == 'cifar100':
    args.data_dir = '/lustre/home/xmwang/share/data/cifar100'
    model_dir = '/lustre/home/xmwang/codes/code_cc_misd/checkpoint/res18-cifar100'
    trainset = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_train)
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    num_classes = 100

trainset = InfoBatch(trainset, args.epochs, 0.5, 0.875)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, sampler=trainset.sampler, **kwargs)
test_labels = testset.targets
criterion = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none')


# 测试集验证


if args.model == 'wrn-28-10':
        model = WideResNet(depth=28, num_classes=num_classes, widen_factor=args.width, dropRate=0.0).to(device)
elif args.model == 'wrn-40-2':
    model = WideResNet(depth=40, num_classes=num_classes, widen_factor=2, dropRate=0.0).to(device)
elif args.model == 'resnet18':
        model = ResNet18(num_classes=num_classes).to(device)


optimizer = optim.SGD(model.parameters(), 
                      lr=0.05, 
                      momentum=args.momentum, 
                      weight_decay=args.weight_decay)

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.05, steps_per_epoch=len(train_loader),
                                                    epochs=args.epochs, div_factor=25.0,
                                                    final_div_factor=10000.0, pct_start=0.3)

def eval_test(model, device, test_loader):
    model.eval()
    test_acc = 0
    test_n = 0
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_n += target.size(0)
    test_time = time.time()
    test_accuracy = correct/test_n
    return test_loss/test_n, test_accuracy, test_n, test_time
def main():
    # define the logger
    logger = logging.getLogger(__name__)
    
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        # Sets the logging level to DEBUG. 
        # This means that all messages at this level 
        # and above (DEBUG, INFO, WARNING, ERROR, CRITICAL) will be captured.
        
        # a list of the handlers to attach to the root logger.
        handlers=[
            logging.FileHandler(os.path.join(model_dir, 'output.log')),
            logging.StreamHandler()
        ])
    # logger记录args
    logger.info(args)
    
    # define the model and optimizer
    
    # create a excel
    # f = xlwt.Workbook()  # init 
    # worksheet1 = f.add_sheet('metrics')
    # worksheet1.write(1, 1,'acc')
    # worksheet1.write(1, 2, 'auroc')
    # worksheet1.write(1, 3, 'aupr-s')
    # worksheet1.write(1, 4, 'aupr-e')
    # worksheet1.write(1, 5, 'fpr')
    # worksheet1.write(1, 6, 'aurc')
    # worksheet1.write(1, 7, 'e-aurc')
    # worksheet1.write(1, 8, 'ece')
    # worksheet1.write(1, 9, 'nll')
    # worksheet1.write(1, 10, 'new_fpr')
    
    

    logger.info('Epoch \t Train Time \t Test Time \t LR \t Train Loss \t Train Reg \t Train Acc \t  Test Loss \t Test Acc')
    
        
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.05, steps_per_epoch=len(train_loader),
                                                    epochs=args.epochs, div_factor=25.0,
                                                    final_div_factor=10000.0, pct_start=0.3,
                                                    last_epoch=epoch*len(train_loader)-1)

        print('Epoch:', epoch)
        print('iterations', len(train_loader))
        model.train()
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_reg_loss = 0
        train_n = 0  
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            with torch.cuda.amp.autocast():
                
                output_clean = model(data)
                loss = criterion(output_clean, target)
            
                loss = trainset.update(loss)
                # print('loss', loss)
            
            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # lr_scheduler.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            
            train_loss += loss.item() * target.size(0)
            train_reg_loss = 0
            train_acc += (output_clean.max(1)[1] == target).sum().item()
            train_n += target.size(0)

        train_time = time.time()
        
        # just eval on the test set
        test_loss, test_accuracy, test_n, test_time = eval_test(model, device, test_loader)

        print('train_loss', train_loss/train_n)
        print('train_acc', train_acc/train_n)
        print('test_loss', test_loss)
        print('test_acc', test_accuracy)
        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, f'model_{epoch}.pth'))
            # torch.save(optimizer.state_dict(), os.path.join(model_dir, f'opt_{epoch}.pth'))


if __name__ == '__main__':
    main()
