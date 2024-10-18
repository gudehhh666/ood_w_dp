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

from data.infobatch_abl import InfoBatch
from models.allconv import AllConvNet
from models.wrn import WideResNet

import metrics
from data.dataset import *
from data.logitnorm import LogitNormLoss
from utils.save_items import append_to_csv_file

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"


# define the parser for the input arguments
parser = argparse.ArgumentParser(description='PyTorch CIFAR Adversarial Training')
model_options = ['res18', 'res50', 'wrn', 'allconv']
dataset_options = ['cifar10', 'cifar100', 'tiny-imagenet']
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=512, metavar='N', help='input batch size for testing (default: 128)')

parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=88, metavar='S', help='random seed (default: 1)')

parser.add_argument('--save-freq', '-s', default=100, type=int, metavar='N', help='save frequency')

parser.add_argument('--dataset', '-d', default='cifar100', choices=dataset_options)
parser.add_argument('--model', '-a', default='wrn', choices=model_options)


parser.add_argument('--learning-rate', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W')

parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
methods = ['EL2N', 'loss', 'baseline', 'EL2N_abl'] # 'infobatch'
parser.add_argument('--method', type=str, default='infobatch_pru', help='method')
parser.add_argument('--model-dir', default='', help='directory of model for saving checkpoint')
# parser.add_argument()

# wrn
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')

parser.add_argument('--rate_e', default=0.1,type=float)

# infobatch
parser.add_argument('--rate', default=0.5,type=float)
parser.add_argument('--delta', default=0.875,type=float)
parser.add_argument('--shuffle', default=False, action='store_true')
parser.add_argument('--high', default=0.9, type=float)


# retrain：
parser.add_argument('--state', default='train', choices=['train', 'tune', 'tune_oe'],type=str, help='state of the model')
parser.add_argument('--load', default='', help='load model')
parser.add_argument('--metrics', default=True, help='metrics')
parser.add_argument('--ls', default=0.1, type=float, help='label smoothing')
parser.add_argument('--use_ln', default=False, action='store_true', help='use logit norm')
parser.add_argument('--mod', default='highest', choices=['highest', 'highest_class', 'lowest', 'middle', 'random', 'two_side', 'uniform', 'middle_class'],help='model directory')
args = parser.parse_args()


# show the state of the model
if args.state == 'train':
    args.learning_rate = 0.1
    print('train and modify thr learning rate', args.learning_rate)
else:
    args.learning_rate = 0.001
    print('tune and modify thr learning rate', args.learning_rate)


model_dir = f'/lustre/home/xmwang/codes/code_cc_misd/checkpoint/{args.model}-{args.dataset}'

# set the model directory to save the model
if args.model_dir =='':
    model_dir = os.path.join(model_dir, f'{args.method}_r{args.rate}_r{args.rate_e}_h{args.high}', f'{args.mod}_d{args.delta}')
else:
    model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
print('=============>model dir:', model_dir)

# use cuda
use_cuda = not args.no_cuda and torch.cuda.is_available()
# torch.manual_seed(args.seed)
device = torch.device("cuda:4" if use_cuda else "cpu")
print('using device :', device)

# 用在dataloader的时候加快进程 only!
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# transforms
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
transform_train = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])
transform_test = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])


if args.dataset == 'cifar10':
    
    data_dir = '/lustre/home/xmwang/share/data/cifar10'
    if args.method == 'infobatch_pru':
        loss_dir = f'/lustre/home/xmwang/share/data/cifar10/{args.model}/EL2N_abl.pkl'
        trainset = CIFAR10_loss_ablation(root=data_dir, train=True, download=True, transform=transform_train, loss_dir=loss_dir, cut_rate=args.rate_e, mod=args.mod)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        trainset = InfoBatch(trainset, args.epochs, args.rate, args.delta, args.high)
        # trainset = InfoBatch(trainset, 0.5, 200, 0.875)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=args.shuffle, sampler = trainset.sampler)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False)
    num_classes = 10
        
if args.dataset == 'cifar100':
    
    data_dir = '/lustre/home/xmwang/share/data/cifar100'
    if args.method == 'infobatch_pru':
        loss_dir = f'/lustre/home/xmwang/share/data/cifar100/{args.model}/EL2N_abl.pkl'
        trainset = CIFAR100_loss_ablation(root=data_dir, train=True, download=True, transform=transform_train, loss_dir=loss_dir, cut_rate=args.rate_e, mod=args.mod)
        testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
        trainset = InfoBatch(trainset, args.epochs, args.rate, args.delta, args.high)
        # trainset = InfoBatch(trainset, 0.5, 200, 0.875)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=args.shuffle, sampler = trainset.sampler)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False)
    num_classes = 100
    
print(f'==========>load the dataset {args.dataset} with method {args.method}')


# test_labels = testset.targets
if args.method == 'infobatch' or args.method == 'infobatch_pru':
    if args.use_ln:
        criterion = LogitNormLoss(device, t=0.01, reduction='none')
        print('=======> use logit norm')
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')
    # criterion = nn.CrossEntropyLoss(reduction='none')
else:
    if args.use_ln:
        criterion = LogitNormLoss(device, t=0.01, reduction='mean')
        print('=======> use logit norm')

    else:
        criterion = nn.CrossEntropyLoss(reduction='mean')
    
print(f'==========>load the criterion with method {args.method} and label smoothing {args.ls}:')
print(f'label smoothing is of no use') if args.method != 'infobatch' else print(f'label smoothing is of use')


# Create model
if args.model == 'allconv' :
    model = AllConvNet(num_classes).to(device)
elif args.model == 'res50' :
    model = ResNet50(num_classes=num_classes).to(device)
    
elif args.model == 'res18' :
    model = ResNet18(num_classes=num_classes).to(device)
    load = f'/lustre/home/xmwang/codes/outlier_exposure/CIFAR/checkpoints/baseline/{args.dataset}_resnet18_baseline_epoch_199.pt'
else:
    model = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate).to(device)
    load = f'/lustre/home/xmwang/codes/outlier_exposure/CIFAR/snapshots/baseline/{args.dataset}_wrn_baseline_epoch_99.pt'

if args.state == 'tune_oe':
    model.load_state_dict(torch.load(load, map_location=device))
    print('fine-tune with oe baseline model load the model successfully', load)

else:
    if args.load != '': 
        print('=======>load the model', args.load)
        model.load_state_dict(torch.load(args.load, map_location=device))
    else:
        print('=======>train the model from scratch')
# optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=args.momentum, weight_decay=args.weight_decay)



# import torch.backends.cudnn as cudnn
# cudnn.benchmark = True  # fire on all cylinders


# set the optimizer and lr_scheduler
optimizer = torch.optim.SGD(
    model.parameters(), args.learning_rate, momentum=args.momentum,
    weight_decay=args.weight_decay)

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.learning_rate,
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=args.epochs, div_factor=25,
                                                    final_div_factor=10000, pct_start=0.3)





# 测试集验证
def eval_test(model, device, test_loader):
    model.eval()
    test_acc = 0
    test_n = 0
    test_loss = 0
    correct = 0
    start_time = time.time()
    with torch.no_grad():
        for data, target,  in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_n += target.size(0)
    test_time = time.time() - start_time
    test_accuracy = correct
    return test_loss/test_n, test_accuracy/test_n, test_time



scaler = torch.cuda.amp.GradScaler()

def train(model, device, train_loader, lr_scheduler):
    model.train()
    start_time = time.time()
    train_loss = 0
    correct = 0
    train_n = 0
    print(len(train_loader))
    # print('lr', optimizer.param_groups[0]['lr'])

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # batch_size = len(data)
        # print(len(train_loader))
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output_clean = model(data)
            loss = criterion(output_clean, target)
            # 3. use <InfoBatch>.update(loss), all scoring/rescaling/getting mean is now conducted at the backend, see previous (research version) code for details.
            if args.method == 'infobatch' or args.method == 'infobatch_pru':

                loss = trainset.update(loss)
                # print('loss', loss)
                # print('ground truth loss:', F.cross_entropy(output_clean, target, reduction='mean', label_smoothing=0.1))
        scaler.scale(loss).backward()
        # loss.backward()
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()
        # print('notice!!!!')
        lr_scheduler.step()
           
        # output_clean = model(data)
        # loss = criterion(output_clean, target)
        # if args.method == 'infobatch':
        #     # print('trainset.update')
        #     loss = trainset.update(loss)

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        train_loss += loss.item()
        correct += output_clean.max(1)[1].eq(target).sum().item()
        train_n += target.size(0)
    print('acc', correct/train_n)
    train_time = time.time() - start_time
    
    return train_loss/len(train_loader), correct/train_n, train_time


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
    
    acc_best = 0
    
    for epoch in range(1, args.epochs + 1):
        # temp_lr = adjust_learning_rate(optimizer, epoch)

        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.learning_rate,
                                                           steps_per_epoch=len(train_loader),
                                                           epochs=args.epochs, div_factor=25,
                                                           final_div_factor=10000, pct_start=0.3,
                                                           last_epoch=(epoch-1) * len(train_loader) - 1)
        
        train_loss, train_acc, train_time = train(model, device, train_loader, lr_scheduler)
        
        # just eval on the test set
        test_loss, test_accuracy, test_time = eval_test(model, device, test_loader)

        if args.metrics:
            acc, auroc, aupr_success, aupr, fpr, aurc, eaurc, ece, nll, new_fpr = metrics.calc_metrics(test_loader,
                                                                                                 model)
        else:
            acc, auroc, aupr_success, aupr, fpr, aurc, eaurc, ece, nll, new_fpr = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        metrics_data = {
                "epoch": epoch,
                "cutting rate" : args.rate ,
                'train_time': train_time,
                "test_time": test_time,
                "model": args.model,
                "train-loss": train_loss,  # Placeholder for actual loss value
                "train-acc": train_acc*100,  # Placeholder for actual accuracy value
                "test-loss": test_loss,  # Placeholder for actual loss value
                "test-acc": test_accuracy*100,  # Placeholder for actual accuracy value
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
        if args.state =='train':
            file_path = model_dir + '/{}-{}-{}-train.csv'.format(args.model, args.dataset, args.method)
        else: 
            file_path = model_dir + '/{}-{}-{}-tune.csv'.format(args.model, args.dataset, args.method)
        append_to_csv_file(metrics_data, file_path) 
        # logging the metrics
        logger.info('%d \t \t \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f',
                epoch, train_time, test_time, optimizer.param_groups[0]['lr'],
                train_loss, 0, train_acc,
                test_loss, test_accuracy)
        print('epoch:', epoch, '  Training Accuracy:', round(100.*train_acc, 3), '  Train loss:', round(train_loss, 4))
        print('epoch:', epoch, '  Testing Accuracy:', round(100.*test_accuracy, 3), '  Test loss:', round(test_loss, 4))

        if test_accuracy > acc_best and epoch > 100:
            torch.save(model.state_dict(), os.path.join(model_dir, f'model_best.pth'))
            print(f'save the best model successfully {test_accuracy} at {epoch}', os.path.join(model_dir, f'model_best.pth'))
            acc_best = test_accuracy
            
        if epoch % args.save_freq == 0 or epoch == 175:
            if args.state == 'train':
                torch.save(model.state_dict(), os.path.join(model_dir, f'model_{epoch}_{args.method}_train_r{args.rate}.pth'))
                print('save model successfully', os.path.join(model_dir, f'model_{epoch}_{args.method}_train_r{args.rate}.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(model_dir, f'model_{epoch}_{args.method}_tune_r{args.rate}.pth'))
                print('save model successfully', os.path.join(model_dir, f'model_{epoch}_{args.method}_tune_r{args.rate}.pth'))
    
    print(f' the best model successfully {test_accuracy} at {epoch}')

            
            

if __name__ == '__main__':
    main()
