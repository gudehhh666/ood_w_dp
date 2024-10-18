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
# from skimage.filters import gaussian as gblur

# sys.path.append("/lustre/home/xmwang/codes/code_cc_misd/utils")
# from utils import *
# from utils.
from utils.display_results import print_measures, print_measures_with_std, get_measures, show_performance

recall_level_default = 0.95

parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=200, help='test batch size')
parser.add_argument('--num_to_avg', type=int, default=10, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
# parser.add_argument('--method_name', '-m', type=str, default='cifar100_resnet_baseline', help='Method name.')
# Loading details
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')


parser.add_argument('--method_name', '-m', type=str, default='tiny-imagenet-res18-baseline', help='Method name.')
parser.add_argument('--in-data-dir', type=str, default='', help='in-distribution dataset.')
parser.add_argument('--load', '-l', type=str, default='/root/code_cc/checkpoint/res18-tiny-imagenet/baseline/r0.5-d0.875-ls0/40-70-120-base/r0.5-d0.875-ls0/model_100_baseline_train_r0.5.pth', help='Checkpoint path to resume / test.')
parser.add_argument('--ood-save-dir', type=str, default='/root/code_cc/checkpoint/ood_result/june_9th.csv', help='output directory.')
args = parser.parse_args()
# /root/code_cc/checkpoint/res50-cifar100/infobatch_grad/r0.5/model_200_infobatch_grad_train_r0.5.pth


all_dir = {'cifar10': '/lustre/home/xmwang/share/data/cifar10',
           'cifar100': '/root/dataset/cifar100',
           'svhn': '/root/ood/svhn',
        #    'imagenet': '/lustre/home/xmwang/share/data/imagenet',
           'isun': '/root/ood/isun-ood/train',
           'textures': '/root/ood/dtd/dtd/images',
           'places365': '/root/ood/Places_365',
        #    'tinyimagenet': '/lustre/home/xmwang/share/data/tiny-imagenet',
           'lsun_c': '/root/ood/lsun_c-ood/train',
           'lsun_r': '/root/ood/lsun_r-ood/train'}

if 'cifar10' in args.method_name:
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

    ood_transform = trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)])

if 'tiny-imagenet' in args.method_name:
    test_transform = trn.Compose([trn.ToTensor()])

    ood_transform = trn.Compose([trn.Resize(64), trn.CenterCrop(64), trn.ToTensor()])
# test_transform = trn.Compose([trn.ToTensor(),])
# cifar100_wrn_loss_10_rate_0
if args.load == '':
    model = args.method_name.split('_')[1]
    method = args.method_name.split('_')[2]
    rate = args.method_name.split('_')[-1]
    dataset = args.method_name.split('_')[0]
    ep = args.method_name.split('_')[3]

if 'cifar10-' in args.method_name:
    args.in_data_dir = '/lustre/home/xmwang/share/data/cifar10'

    test_data = dset.CIFAR10(args.in_data_dir, train=False, transform=test_transform)
    num_classes = 10
    in_data = 'cifar10'
    if args.load == '':
        load = f'/lustre/home/xmwang/codes/code_cc_misd/checkpoint/{model}-cifar10'
elif 'cifar100-' in args.method_name:
    args.in_data_dir = all_dir['cifar100']
    test_data = dset.CIFAR100(args.in_data_dir, train=False, transform=test_transform)
    num_classes = 100
    in_data = 'cifar100'
    if args.load == '':
        load = f'/lustre/home/xmwang/codes/code_cc_misd/checkpoint/{model}-cifar100'
if 'tiny-imagenet' in args.method_name:
    num_classes = 200
    in_data = 'tiny-imagenet'
    from data.Tiny_imagenet import get_tiny
    root_dir = {'train': '/root/tiny-imagenet/train',
            'test': '/root/tiny-imagenet/valid'}
    _, test_data = get_tiny(root_dir, True , True)
print('In-distribution: {}'.format(in_data))




# args.ood_save_dir = f'/lustre/home/xmwang/codes/code_cc_misd/ood_result/new_result_.csv'
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


# if 'loss' in args.method_name:

if args.ood_save_dir == '':
    args.ood_save_dir = f"/lustre/home/xmwang/codes/code_cc_misd/ood_result/{dataset}_{method}_new.csv"

print('OOD Save Dir: {}'.format(args.ood_save_dir))


if args.ngpu == 1:
    device = 'cuda:0'
    print('Using 1 GPU.')


print("=======>checkpoint found at '{}'".format(args.load))
checkpoint = torch.load(args.load, map_location=device)
net.load_state_dict(checkpoint)
        
net.eval()

if args.ngpu > 1:
    device = 'cuda'
    net.to(device)
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    print('Using', args.ngpu, 'GPUs:', list(range(args.ngpu)))
if args.ngpu == 1:
    net.to(device)

cudnn.benchmark = True  # fire on all cylinders

ood_num_examples = len(test_data) // 5

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

def get_ood_scores(loader, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []
    # in_dist = True,处理分布内数据？

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                # 首先检查是否已经处理分布外数据所需的批次 
                # idx= test_data/5/test_batch_size 
                # 这里的目的是将ood数据处理控制在0.2
                break

            data = data.to(device)

            output = net(data)
            smax = to_np(F.softmax(output, dim=1))

            if args.use_xent:
                _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            else:
                # 得到每个softmax的最大值
                # why use -?
                _score.append(-np.max(smax, axis=1))

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                
                wrong_indices = np.invert(right_indices)

                if args.use_xent:
                    _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                else:
                    # return the softmax score of the prediction
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()


in_score, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)
# print('In-distribution score shape: {}'.format(in_score.shape))
# print('in_score: ', in_score)
num_right = len(right_score)
num_wrong = len(wrong_score)
print('Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))


print('\nUsing CIFAR-10 as typical data') if num_classes == 10 else print('\nUsing CIFAR-100 as typical data')

print('\n\nError Detection')
show_performance(wrong_score, right_score, in_data, 'Error Detection', method_name=args.method_name, ood_save_dir= args.ood_save_dir)


# /////////////// OOD Detection ///////////////
auroc_list, aupr_list, fpr_list = [], [], []


def get_and_print_results(ood_loader, in_data=in_data, ood_data='', num_to_avg=args.num_to_avg):

    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_score = get_ood_scores(ood_loader)
        measures = get_measures(out_score, in_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
        # print_measures(measures[0], measures[1], measures[2], in_data, ood_data, args.method_name)
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    if num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs, in_data, ood_data + '_mean', args.method_name, ood_save_dir=args.ood_save_dir)        
    else:
        print_measures(auroc, aupr, fpr, in_data, ood_data + '_mean', args.method_name, ood_save_dir=args.ood_save_dir)



# /////////////// Places365 ///////////////
# from pytorch_ood.dataset.img import Places365
ood_data = dset.Places365(root=all_dir['places365'], 
                        split='val', 
                        small=True, 
                        download=False, 
                        transform=ood_transform)

ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                        num_workers=args.prefetch, pin_memory=True)

print('\n\nPlaces365 Detection')
get_and_print_results(ood_loader, ood_data='Places365')

# /////////////// Textures ///////////////

ood_data = dset.ImageFolder(root=all_dir['textures'],
                            transform=ood_transform)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                        num_workers=args.prefetch, pin_memory=True)

print('\n\nTexture Detection')
get_and_print_results(ood_loader, ood_data='Textures')

# /////////////// SVHN ///////////////
from torchvision.datasets import svhn
ood_data = svhn.SVHN(root=all_dir['svhn'], split="test",
                    transform=ood_transform, download=False)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nSVHN Detection')
get_and_print_results(ood_loader, ood_data='SVHN')






# /////////////// huggingface ///////////////
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk
import PIL.Image as Image
class HuggingFaceDataset(Dataset):
    def __init__(self, hf_dataset, transforms):
        self.dataset = hf_dataset
        self.transform = transforms
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # print(item.items())
        # 转换PIL图像为张量
        if isinstance(item['image'], Image.Image):
            item['image'] = self.transform(item['image'])
        return item['image'], 0

# /////////////// LSUN ///////////////
from torchvision.datasets import lsun
ood_data = load_from_disk(all_dir['lsun_c'])
ood_data = HuggingFaceDataset(ood_data, ood_transform)

ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nLSUN_crop Detection')
get_and_print_results(ood_loader, ood_data='LSUN_crop')



ood_data = load_from_disk(all_dir['lsun_r'])
ood_data = HuggingFaceDataset(ood_data, ood_transform)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nLSUN_resize Detection')
get_and_print_results(ood_loader, ood_data='LSUN_resize')



#  /////////////// iSUN ///////////////
ood_data = load_from_disk(all_dir['isun'])
ood_data = HuggingFaceDataset(ood_data, ood_transform)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\niSUN Detection')
get_and_print_results(ood_loader, ood_data='iSUN')


# # /////////////// TinyImgNet Data ///////////////
# ood_data = dset.ImageFolder(root="/lustre/home/xmwang/share/data/Imagenet_crop",
#                             transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
#                                                    trn.ToTensor(), trn.Normalize(mean, std)]))
# ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
#                                          num_workers=args.prefetch, pin_memory=True)

# print('\n\nTinyImageNet_crop Detection')
# get_and_print_results(ood_loader, ood_data='TinyImageNet_crop')

# ood_data = dset.ImageFolder(root="/lustre/home/xmwang/share/data/Imagenet_resize",
#                             transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
#                                                    trn.ToTensor(), trn.Normalize(mean, std)]))
# ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
#                                          num_workers=args.prefetch, pin_memory=True)


# print('\n\nTinyImageNet_resize Detection')
# get_and_print_results(ood_loader, ood_data='TinyImageNet_resize')


# /////////////// Mean Results ///////////////

print('\n\nMean Test Results')
print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), in_data=in_data, ood_data='mean', method_name=args.method_name, ood_save_dir=args.ood_save_dir)
