from __future__ import print_function
from cmath import inf
import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import csv
from data.infobatch import InfoBatch
from torch.utils.data.distributed import DistributedSampler


from models.resnet import *
from models.wideresnet import *

import metrics
from data.dataset import *
from utils.save_items import append_to_csv_file

parser = argparse.ArgumentParser(description='PyTorch CIFAR Adversarial Training')
model_options = ['resnet18', 'wrn-28-10', 'wrn-40-2']
dataset_options = ['cifar10', 'cifar100', 'tiny-imagenet']
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N', help='input batch size for testing (default: 128)')

parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W')
parser.add_argument('--lr_max', type=float, default=0.1, metavar='LRM', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
# parser.add_argument('--model-dir', default='checkpoint/res18-cifar10', help='directory of model for saving checkpoint')
# parser.add_argument('--data-dir', default='', help='directory of dataset')
parser.add_argument('--save-freq', '-s', default=10, type=int, metavar='N', help='save frequency')

parser.add_argument('--dataset', '-d', default='cifar10', choices=dataset_options)
parser.add_argument('--model', '-a', default='resnet18', choices=model_options)

# methods = ['EL2N', 'loss', 'Grad']
parser.add_argument('--rate', type=float, default=0.5, metavar='R', help='rate of the dataset')
# parser.add_argument('--trials', type=int, default=3, metavar='T', help='number of trials')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='The initial learning rate.')

parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train')
parser.add_argument('--method', type=str, default='oe', metavar='M', help='method')
parser.add_argument('--model-dir', default='/lustre/home/xmwang/codes/code_cc_misd/checkpoint/res18-cifar10/oe_tune', help='directory of model for saving checkpoint')
# parser.add_argument()
args = parser.parse_args()

model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda:0" if use_cuda else "cpu")

# 用在dataloader的时候加快进程 only!
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# setup data loader
# transforms.RandomCrop(32, padding=4)：随机裁剪为32*32，填充为4
# transforms.RandomHorizontalFlip()：随机水平翻转
transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),])
transform_test = transforms.Compose([transforms.ToTensor(),])

if args.dataset == 'cifar10':
    args.data_dir = '/lustre/home/xmwang/share/data/cifar10'
    loss_dir = os.path.join(args.data_dir, args.method+'.pkl')
    print('loss_dir', loss_dir)
    trainset = CIFAR10_loss(root=args.data_dir, train=True, download=True, transform=transform_train)
    # trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testset = CIFAR10_loss(root=args.data_dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    num_classes = 10
elif args.dataset == 'cifar100':
    
    args.data_dir = '/lustre/home/xmwang/share/data/cifar100'
    loss_dir = os.path.join(args.data_dir, args.method+'.pkl')
    print('loss_dir', loss_dir)
    # trainset = CIFAR100_loss(root=args.data_dir, train=True, download=True, transform=transform_test)
    trainset = CIFAR100_loss(root=args.data_dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testset = CIFAR100_loss(root=args.data_dir, train=False, download=True, transform=transform_test)
    # testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    num_classes = 100
    

test_labels = testset.targets
# =========================ood dataset and ood loader
print(len(trainset))
dummy_targets = torch.ones(len(trainset))
ood_data = torch.from_numpy(np.float32(np.clip(
    np.random.normal(size=(len(trainset), 3, 32, 32), scale=0.5), -1, 1)))
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)

ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=False, **kwargs)



# ==============================
# define the model
if args.model == 'allconv':
    model = WideResNet(depth=28, num_classes=num_classes, widen_factor=args.width, dropRate=0.0).to(device)
elif args.model == 'wrn':
    model = WideResNet(depth=40, num_classes=num_classes, widen_factor=2, dropRate=0.3).to(device)
elif args.model == 'res18':
    model = ResNet18(num_classes=num_classes).to(device)
    if args.dataset == 'cifar10':
        load = '/lustre/home/xmwang/codes/code_cc_misd/checkpoint/res18-cifar10/model_200_baseline.pth'
    if args.dataset =='cifar100':
        load = '/lustre/home/xmwang/codes/code_cc_misd/checkpoint/res18-cifar100/model_200_baseline.pth'
    model.load_state_dict(torch.load(load, map_location=device))
    print('load model successfully', load)

# ===============================
def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))
    
    
optimizer = optim.SGD(model.parameters(), 
                      lr=args.learning_rate, 
                      momentum=args.momentum, 
                      weight_decay=args.weight_decay)



# 设置 lr_scheduler
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))




criterion = nn.CrossEntropyLoss()
# 测试集验证
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
    test_accuracy = correct
    return test_loss, test_accuracy, test_n, test_time

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
            logging.FileHandler(os.path.join(model_dir, 'output-fine-tuning.log')),
            logging.StreamHandler()
        ])
    # logger记录args
    logger.info(args)
    



    

    logger.info('Epoch \t Train Time \t Test Time \t LR \t Train Loss \t Train Reg \t Train Acc \t  Test Loss \t Test Acc')
    
    
    for epoch in range(1, args.epochs + 1):
        # temp_lr = adjust_learning_rate(optimizer, epoch)
        model.train()
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_reg_loss = 0
        train_n = 0  
        for in_set, out_set in zip(train_loader, ood_loader):
            data = torch.cat((in_set[0], out_set[0]), 0)
            target  = in_set[1]
            data, target = data.to(device), target.to(device)
            # batch_size = len(data)
            output_clean = model(data)
            loss = criterion(output_clean[:len(in_set[0])], target)
            
            loss += 0.5 * -(output_clean[len(in_set[0]):].mean(1) - torch.logsumexp(output_clean[len(in_set[0]):], dim=1)).mean()
            # scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * target.size(0)
            train_reg_loss = 0
            train_acc += (output_clean[:len(in_set[0])].max(1)[1] == target).sum().item()
            train_n += target.size(0)

        train_time = time.time()
        
        # just eval on the test set
        test_loss, test_accuracy, test_n, test_time = eval_test(model, device, test_loader)

        acc, auroc, aupr_success, aupr, fpr, aurc, eaurc, ece, nll, new_fpr = metrics.calc_metrics(test_loader,
                                                                                                 model)
    
        metrics_data = {
                "epoch": epoch,
                # "rate" : args.rate,
                'lr': optimizer.state_dict()['param_groups'][0]['lr'],
                "model": args.model,
                "loss": loss.item(),  # Placeholder for actual loss value
                "test-acc": acc,  # Placeholder for actual accuracy value
                "auroc": auroc,  # Placeholder for actual AUROC value
                "aupr_success": aupr_success,  # Placeholder for actual AUPR for successful predictions
                "aupr": aupr,  # Placeholder for actual AUPR value
                "fpr": fpr,  # Placeholder for actual FPR value
                "aurc": aurc,  # Placeholder for actual AURC value
                "eaurc": eaurc,  # Placeholder for actual EAURC value
                "ece": ece,  # Placeholder for actual ECE value
                "nll": nll,  # Placeholder for actual NLL value
                "new_fpr": new_fpr  # Placeholder for actual new FPR value
                }  
        file_path = model_dir + '/{}-{}-{}-tune.csv'.format(args.model, args.dataset, args.method)
        append_to_csv_file(metrics_data, file_path) 
        # logging the metrics
        logger.info('%d \t \t \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f',
                epoch, train_time - start_time, test_time - train_time, optimizer.state_dict()['param_groups'][0]['lr'],
                train_loss/train_n, train_reg_loss/train_n, train_acc/train_n,
                test_loss/test_n, test_accuracy/test_n)

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, f'model_{epoch}_{args.method}_tuning_r{args.rate}.pth'))
            print('save model successfully', os.path.join(model_dir, f'model_{epoch}_{args.method}_tuning_r{args.rate}.pth'))

if __name__ == '__main__':
    main()




