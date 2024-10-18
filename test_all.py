
'''
Evaluate the performance of basic param in a model on CIFAR-10 or CIFAR-100 dataset.
'''


import numpy as np
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
import sys
from skimage.filters import gaussian as gblur
import metrics

# sys.path.append("/lustre/home/xmwang/codes/code_cc_misd/utils")
from utils.save_items import append_to_csv_file

from utils.display_results import print_measures, print_measures_with_std, get_measures, show_performance

recall_level_default = 0.95

parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=200, help='test batch size')
# parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
# parser.add_argument('--method_name', '-m', type=str, default='cifar100_resnet_baseline', help='Method name.')
# Loading details
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')

parser.add_argument('--method_name', '-m', type=str, default='cifar100_wrn_MoSo', help='Method name.')
parser.add_argument('--in-data-dir', type=str, default='', help='in-distribution dataset.')
parser.add_argument('--load', '-l', type=str, default='/lustre/home/xmwang/codes/pruning_method/Influence-MoSo/saveroot/wen_exap/checkpoint/trial_0_300.pth', help='Checkpoint path to resume / test.')
parser.add_argument('--num-to-avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--out-put-dir', '-a', type=str, default='/lustre/home/xmwang/codes/code_cc_misd/basic_performance', help='output directory.')
parser.add_argument('--save', '-o', type=bool, default=True, help='Output to csv file.')
args = parser.parse_args()

# torch.manual_seed(1)
# np.random.seed(1)

# test_transform = trn.Compose([trn.ToTensor(),])

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

transform_train = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if not os.path.exists(args.out_put_dir):
    os.makedirs(args.out_put_dir)

if 'cifar10_' in args.method_name:
    args.in_data_dir = '/lustre/home/xmwang/share/data/cifar10'
    test_data = dset.CIFAR10(args.in_data_dir, train=False, transform=test_transform)
    num_classes = 10
    in_data = 'cifar10'
else:
    args.in_data_dir = '/lustre/home/xmwang/share/data/cifar100'
    test_data = dset.CIFAR100(args.in_data_dir, train=False, transform=test_transform)
    num_classes = 100
    in_data = 'cifar100'

test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                            num_workers=args.prefetch, pin_memory=True)

if 'res18' in args.method_name:
    from models.resnet import ResNet18
    net = ResNet18(num_classes)
elif 'res50' in args.method_name:
    from models.resnet import ResNet50
    net = ResNet50(num_classes)
elif 'wrn' in args.method_name:
    from models.wrn import WideResNet
    net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
    

if args.ngpu > 1:
    device = 'cuda'
    net.to(device)
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu == 1:
    device = 'cuda:1'
    net.to(device)
if args.load is not None:
    if os.path.isfile(args.load):
        print("=======> loading checkpoint '{}'".format(args.load))
        checkpoint = torch.load(args.load, map_location=device)
        try:
            net_load = checkpoint['net']
            print('load acc', checkpoint['acc'])
            print('load epoch', checkpoint['epoch'])
            net.load_state_dict(net_load)
        except:
            net.load_state_dict(checkpoint)
            
        

cudnn.benchmark = True  # fire on all cylinders

acc_list = []
auroc_list = []
aupr_success_list = []
aupr_list = []
fpr_list = []
aurc_list = []
eaurc_list = []
ece_list = []
nll_list = []
new_fpr_list = []
for i in range(args.num_to_avg):
    acc, auroc, aupr_success, aupr, fpr, aurc, eaurc, ece, nll, new_fpr = metrics.calc_metrics(test_loader,
                                                                                                 model=net,)
    

    
    acc_list.append(acc)
    auroc_list.append(auroc)
    aupr_success_list.append(aupr_success)
    aupr_list.append(aupr)
    fpr_list.append(fpr)
    aurc_list.append(aurc)
    eaurc_list.append(eaurc)
    ece_list.append(ece)
    nll_list.append(nll)
    new_fpr_list.append(new_fpr)
    # Define the metrics to be saved
    metrics_data = {
                    "name": args.method_name,
                    # "ratio": 0,
                    "acc": acc,  # Placeholder for actual accuracy value
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
    print(metrics_data)
    if args.save:
        append_to_csv_file(metrics_data, os.path.join(args.out_put_dir, 'base-eval.csv'))
        if i == args.num_to_avg - 1 and args.num_to_avg > 1:
            metrics_data = {
                "name": args.method_name,
                "acc": '{:.4f}\t+/- {:.4f}'.format(np.mean(acc_list), np.std(acc_list)),
                "auroc": '{:.4f}\t+/- {:.4f}'.format(np.mean(auroc_list), np.std(auroc_list)),
                "aupr_success": '{:.4f}\t+/- {:.4f}'.format(np.mean(aupr_success_list), np.std(aupr_success_list)),
                "aupr": '{:.4f}\t+/- {:.4f}'.format(np.mean(aupr_list), np.std(aupr_list)),
                "fpr": '{:.4f}\t+/- {:.4f}'.format(np.mean(fpr_list), np.std(fpr_list)),
                "aurc": '{:.4f}\t+/- {:.4f}'.format(np.mean(aurc_list), np.std(aurc_list)),
                "eaurc": '{:.4f}\t+/- {:.4f}'.format(np.mean(eaurc_list), np.std(eaurc_list)),
                "ece": '{:.4f}\t+/- {:.4f}'.format(np.mean(ece_list), np.std(ece_list)),
                "nll": '{:.4f}\t+/- {:.4f}'.format(np.mean(nll_list), np.std(nll_list)),
                "new_fpr": '{:.4f}\t+/- {:.4f}'.format(np.mean(new_fpr_list), np.std(new_fpr_list))
                }
            append_to_csv_file(metrics_data, os.path.join(args.out_put_dir, 'base-eval.csv'))
