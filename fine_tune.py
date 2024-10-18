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
import torchvision.transforms as trn
from models.resnet import *
from models.wideresnet import *

from data.infobatch import InfoBatch
from models.allconv import AllConvNet
from models.wrn import WideResNet

import metrics
from data.dataset import *
from utils.save_items import append_to_csv_file

# define the parser for the input arguments
parser = argparse.ArgumentParser(description='PyTorch CIFAR Adversarial Training')
model_options = ['res18', 'res50','wrn', 'allconv']
dataset_options = ['cifar10', 'cifar100', 'tiny-imagenet']
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N', help='input batch size for testing (default: 128)')

parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--save-freq', '-s', default=20, type=int, metavar='N', help='save frequency')

parser.add_argument('--dataset', '-d', default='cifar100', choices=dataset_options)
parser.add_argument('--model', '-a', default='res50', choices=model_options)

# parser.add_argument('--rate', type=float, default=0.5, metavar='R', help='rate of the dataset')
# parser.add_argument('--trials', type=int, default=3, metavar='T', help='number of trials')
parser.add_argument('--learning-rate', type=float, default=0.05, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W')

parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
methods = ['EL2N', 'loss', 'baseline']
parser.add_argument('--method', type=str, default='baseline', help='method')
parser.add_argument('--model-dir', default='', help='directory of model for saving checkpoint')
# parser.add_argument()

# wrn
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')


# infobatch
parser.add_argument('--rate',default=0.5,type=float)
parser.add_argument('--delta',default=0.875,type=float)
parser.add_argument('--shuffle', default=False, action='store_true')


# retrain：
parser.add_argument('--train-scratch', default=False, action='store_true')
parser.add_argument('--load', default='', help='load model')
args = parser.parse_args()

if args.train_scratch:
    args.learning_rate = 0.05
    print('train from scratch and modify thr learning rate', args.learning_rate)

if args.dataset == 'cifar10':
    model_dir = f'/lustre/home/xmwang/codes/code_cc_misd/checkpoint/{args.model}-cifar10'
elif args.dataset == 'cifar100':
    model_dir = f'/lustre/home/xmwang/codes/code_cc_misd/checkpoint/{args.model}-cifar100'

# settings
if args.model_dir =='':
    model_dir = os.path.join(model_dir, args.method)
else:
    model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
print('=============>model dir:', model_dir)



use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

# 用在dataloader的时候加快进程 only!
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# setup data loader
# transforms.RandomCrop(32, padding=4)：随机裁剪为32*32，填充为4
# transforms.RandomHorizontalFlip()：随机水平翻转
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

transform_train = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])
transform_test = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if args.dataset == 'cifar10':
    args.data_dir = '/lustre/home/xmwang/share/data/cifar10'
    if args.method in  methods:
        
        if args.method != 'baseline':
            loss_dir = os.path.join(args.data_dir, args.model, args.method+'.pkl')
            print('loss_dir', loss_dir)
        else:
            loss_dir=None
            args.rate = 0
        trainset = CIFAR10_loss(root=args.data_dir, train=True, download=True, transform=transform_train, loss_dir=loss_dir, cut_rate=args.rate)
    # trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_test)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    elif args.method == 'infobatch':
        trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
        trainset = InfoBatch(trainset, args.rate if args.rate else None, args.epochs, args.delta)

        print('==========>dataloader')
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=args.shuffle, sampler = trainset.sampler)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False)
    
    num_classes = 10
elif args.dataset == 'cifar100':
    
    args.data_dir = '/lustre/home/xmwang/share/data/cifar100'
    if args.method in  methods:

        if args.method != 'baseline':
            loss_dir = os.path.join(args.data_dir, args.model, args.method+'.pkl')
            print('loss_dir', loss_dir)
        else:
            loss_dir=None
            args.rate = 0
    # trainset = CIFAR100_loss(root=args.data_dir, train=True, download=True, transform=transform_test)
        trainset = CIFAR100_loss(root=args.data_dir, train=True, download=True, transform=transform_train, loss_dir=loss_dir, cut_rate=args.rate)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
        # testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    elif args.method == 'infobatch':
        trainset = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
        trainset = InfoBatch(trainset, args.rate if args.rate else None, args.epochs, args.delta)

        print('==========>dataloader')
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=args.shuffle, sampler = trainset.pruning_sampler())
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False)
    num_classes = 100
    

# test_labels = testset.targets
if args.method == 'infobatch':
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1,reduction='none')
else:
    criterion = nn.CrossEntropyLoss(reduction='mean')


# Create model
if args.model == 'allconv' :
    model = AllConvNet(num_classes)
elif args.model == 'res18' :
    model = ResNet18(num_classes=num_classes).to(device)
    if args.dataset == 'cifar10':
        load = '/lustre/home/xmwang/codes/outlier_exposure/CIFAR/checkpoints/baseline/cifar10_resnet18_baseline_epoch_199.pt'
    if args.dataset =='cifar100':
        load = '/lustre/home/xmwang/codes/outlier_exposure/CIFAR/checkpoints/baseline/cifar100_resnet18_baseline_epoch_199.pt'

else:
    model = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate).to(device)
    if args.dataset == 'cifar10':
        load = '/lustre/home/xmwang/codes/outlier_exposure/CIFAR/snapshots/baseline/cifar10_wrn_baseline_epoch_99.pt'
    if args.dataset =='cifar100':
        load = '/lustre/home/xmwang/codes/outlier_exposure/CIFAR/snapshots/baseline/cifar100_wrn_baseline_epoch_99.pt'

if args.train_scratch:
    print('train from scratch')
else:
    if args.load == '':
        model.load_state_dict(torch.load(load, map_location=device))
        print('load model successfully', load)
    else:
        print('train from the pre-trained model', args.load)
        model.load_state_dict(torch.load(args.load, map_location=device))
# optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=args.momentum, weight_decay=args.weight_decay)



import torch.backends.cudnn as cudnn
cudnn.benchmark = True  # fire on all cylinders

optimizer = torch.optim.SGD(
    model.parameters(), args.learning_rate, momentum=args.momentum,
    weight_decay=args.weight_decay, nesterov=True)



# def cosine_annealing(step, total_steps, lr_max, lr_min):
#     return lr_min + (lr_max - lr_min) * 0.5 * (
#             1 + np.cos(step / total_steps * np.pi))

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.learning_rate,
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=args.epochs+1, div_factor=25,
                                                    final_div_factor=10000, pct_start=0.3)

# 设置 lr_scheduler
# scheduler = torch.optim.lr_scheduler.LambdaLR(
#     optimizer,
#     lr_lambda=lambda step: cosine_annealing(
#         step,
#         args.epochs * len(train_loader),
#         1,  # since lr_lambda computes multiplicative factor
#         1e-6 / args.learning_rate))






# 测试集验证
def eval_test(model, device, test_loader):
    model.eval()
    test_acc = 0
    test_n = 0
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target,  in test_loader:
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
    
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        # temp_lr = adjust_learning_rate(optimizer, epoch)
        model.train()
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_reg_loss = 0
        train_n = 0
        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.learning_rate,
                                                           steps_per_epoch=len(train_loader),
                                                           epochs=args.epochs+1, div_factor=25,
                                                           final_div_factor=10000, pct_start=0.3,
                                                           last_epoch=epoch * len(train_loader) - 1)
        
        
        if args.method != 'infobatch':  
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                batch_size = len(data)
                output_clean = model(data)
                loss = criterion(output_clean, target)
                

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                train_loss += loss.item() * target.size(0)
                train_reg_loss = 0
                train_acc += (output_clean.max(1)[1] == target).sum().item()
                train_n += target.size(0)

        if args.method == 'infobatch':
            print('using infobatch')
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                
                output_clean = model(inputs)
                loss = criterion(output_clean, targets)
                loss = trainset.update(loss)
                # trainset.__setscore__(indices.detach().cpu().numpy(),loss.detach().cpu().numpy())

                # loss = loss*rescale_weight
                # loss = torch.mean(loss)
                loss.backward()
                
                optimizer.step()
                lr_scheduler.step()


                train_loss += loss.item() * targets.size(0)
                train_reg_loss = 0
                train_acc += (output_clean.max(1)[1] == targets).sum().item()
                train_n += targets.size(0)

        train_time = time.time()
        
        # just eval on the test set
        test_loss, test_accuracy, test_n, test_time = eval_test(model, device, test_loader)

        acc, auroc, aupr_success, aupr, fpr, aurc, eaurc, ece, nll, new_fpr = metrics.calc_metrics(test_loader,
                                                                                                 model)
    
        metrics_data = {
                "epoch": epoch,
                "rate/ratio" : args.rate ,
                'train_time': train_time - start_time,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "model": args.model,
                "loss": loss.item(),  # Placeholder for actual loss value
                "train-acc": train_acc/train_n,  # Placeholder for actual accuracy value
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
        if args.train_scratch:
            file_path = model_dir + '/{}-{}-{}-train.csv'.format(args.model, args.dataset, args.method)
        else: 
            file_path = model_dir + '/{}-{}-{}-tune.csv'.format(args.model, args.dataset, args.method)
        append_to_csv_file(metrics_data, file_path) 
        # logging the metrics
        logger.info('%d \t \t \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f',
                epoch, train_time - start_time, test_time - train_time, 1e-4,
                train_loss/train_n, train_reg_loss/train_n, train_acc/train_n,
                test_loss/test_n, test_accuracy/test_n)
        if acc > best_acc:
            torch.save(model.state_dict(), os.path.join(model_dir, f'model_best.pth'))
            best_acc = acc
            print(f'save the best model successfully with {acc}', os.path.join(model_dir, f'model_best.pth'))
        
        
        if not args.train_scratch:
        # save checkpoint
            if epoch % args.save_freq == 0:
                if args.method == 'infobatch':
                    torch.save(model.state_dict(), os.path.join(model_dir, f'model_{epoch}_{args.method}_tuning_r{args.rate}.pth'))
                    print('save model successfully', os.path.join(model_dir, f'model_{epoch}_{args.method}_tuning_r{args.rate}.pth'))
                else:
                    
                    torch.save(model.state_dict(), os.path.join(model_dir, f'model_{epoch}_{args.method}_tuning_r{args.rate}.pth'))
                    print('save model successfully', os.path.join(model_dir, f'model_{epoch}_{args.method}_tuning_r{args.rate}.pth'))
        else:
            if epoch % args.save_freq == 0:
                if args.method in ['infobatch']:
                    torch.save(model.state_dict(), os.path.join(model_dir, f'model_{epoch}_{args.method}_train_r{args.rate}.pth'))
                    print('save model successfully', os.path.join(model_dir, f'model_{epoch}_{args.method}_train_r{args.rate}.pth'))
                else:
  
                    torch.save(model.state_dict(), os.path.join(model_dir, f'model_{epoch}_{args.method}_train_r{args.rate}.pth'))
                    print('save model successfully', os.path.join(model_dir, f'model_{epoch}_{args.method}_train_r{args.rate}.pth'))

            
            

if __name__ == '__main__':
    main()
