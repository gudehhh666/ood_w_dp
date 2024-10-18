import argparse
import datetime
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
# from infobatch import InfoBatch

from data.infobatch import InfoBatch
from torchvision import transforms
from models.resnet import ResNet18, ResNet50, ResNet101
from models.wrn import WideResNet
import torch.distributed as dist
import metrics 
from utils.save_items import append_to_csv_file


RANK = int(os.getenv('RANK', -1))
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
print('RANK', RANK, 'LOCAL_RANK', LOCAL_RANK)

def safe_print(*args, **kwargs):
    if RANK in (-1, 0):
        print(*args)

def setup_ddp():
    world_size = int(os.getenv('WORLD_SIZE', 1))
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group('nccl', rank=RANK, world_size=world_size)

def destroy_ddp():
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.2, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--use_info_batch', default=True, action='store_true',
                        help='whether use info batch or not.')
    parser.add_argument('--use_ddp', action='store_true', help='whether use ddp or not.')
    parser.add_argument('--fp16', action='store_true', help='use mix precision training')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W',
                        help='SGD weight decay (default: 5e-4)')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='different optimizers')
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    # onecycle scheduling arguments
    parser.add_argument('--max-lr', default=0.05, type=float)
    parser.add_argument('--div-factor', default=25, type=float)
    parser.add_argument('--final-div', default=10000, type=float)
    parser.add_argument('--num_epoch', default=200,
                        type=int, help='training epochs')
    parser.add_argument('--pct-start', default=0.3, type=float)
    parser.add_argument('--shuffle', default=True, action='store_true')
    parser.add_argument('--ratio', default=0.5, type=float, help='prune ratio')
    parser.add_argument('--delta', default=0.875, type=float)
    parser.add_argument('--model', default='r18', type=str)
    parser.add_argument('--dataset', '-d', default='cifar10', type=str)
    parser.add_argument('--metrics', default=True, type=bool)

    args = parser.parse_args()
   
    if not torch.cuda.is_available():
        device = 'cpu'
    elif args.use_ddp:
        device = 'cuda:%d' % LOCAL_RANK
        setup_ddp()
    else:
        device = 'cuda:0'
    
    try:
        criterion = nn.CrossEntropyLoss(
            label_smoothing=args.label_smoothing, reduction='none').to(device)
    except:
        safe_print('warning! This version has no label smooth.')
        criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    test_criterion = nn.CrossEntropyLoss().to(device)

    best_acc = 0  # best test accuracy
    best_loss = 1e3 # best test loss
    best_epoch = 0
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch


    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), 
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.ToTensor(), 
                                          transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([transforms.ToTensor(), 
                                         transforms.Normalize(mean, std)])

    
    # stats = ((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025))
    # train_transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomCrop(32, padding=4, padding_mode="constant"),
    #     transforms.ToTensor(),
    #     transforms.Normalize(*stats)
    # ])

    # test_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(*stats)
    # ])
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='/lustre/home/xmwang/share/data/cifar10', train=True, transform=train_transform,
                                            download=True)
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='/lustre/home/xmwang/share/data/cifar100', train=True, transform=train_transform,
                                            download=True)
    # 1.Substitute dataset with InfoBatch dataset, optionally set r; delta and num_epoch are needed to anneal with full data
    if args.use_info_batch:
        safe_print('Use info batch.')
        trainset = InfoBatch(trainset, args.num_epoch, args.ratio, args.delta)
    else:
        safe_print('Use normal full batch.')

    # 2.Substitute sampler
    sampler = None
    train_shuffle = True
    if args.use_info_batch:
        sampler = trainset.sampler
        train_shuffle = False
    if args.use_ddp and not args.use_info_batch:
        sampler = DistributedSampler(trainset, shuffle=True)
        train_shuffle = False
    safe_print(type(sampler))
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=train_shuffle, sampler=sampler, num_workers=4, pin_memory=True)

    if args.dataset == 'cifar10':
        testset = torchvision.datasets.CIFAR10(
            root='/lustre/home/xmwang/share/data/cifar10', train=False, download=True, transform=test_transform)
        num_classes = 10
    elif args.dataset == 'cifar100':
        testset = torchvision.datasets.CIFAR100(
            root='/lustre/home/xmwang/share/data/cifar100', train=False, download=True, transform=test_transform)
        num_classes = 10

    testloader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    print('==> Building model..', args.model.lower())

    if args.model.lower() == 'r18':
        net = ResNet18(num_classes)
    elif args.model.lower() == 'r50':
        net = ResNet50(num_classes)
    elif args.model.lower() == 'r101':
        net = ResNet101(num_classes)
    else:
        net = ResNet50(num_classes)
    net = net.to(device)
    # if args.use_ddp:
    #     safe_print('use ddp')
    #     net = torch.nn.parallel.DistributedDataParallel(net, [LOCAL_RANK], LOCAL_RANK)
    # else:
    #     safe_print('use normal data parallel')
    #     net = torch.nn.DataParallel(net)
    # Model
   

    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
   
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.max_lr, steps_per_epoch=len(trainloader),
                                                    epochs=args.num_epoch, div_factor=args.div_factor,
                                                    final_div_factor=args.final_div, pct_start=args.pct_start)

    train_acc = []
    valid_acc = []
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
 
    def train_info_batch(epoch):
        safe_print('\nEpoch: %d, iterations %d' % (epoch, len(trainloader)))
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        print('lr', optimizer.param_groups[0]['lr'])
        for batch_idx, blobs in enumerate(trainloader):
            inputs, targets = blobs
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(args.fp16):
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                # 3. use <InfoBatch>.update(loss), all scoring/rescaling/getting mean is now conducted at the backend, see previous (research version) code for details.
                loss = trainset.update(loss)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        safe_print('epoch:', epoch, '  Training Accuracy:', round(100. * correct /
            total, 3), '  Train loss:', round(train_loss / len(trainloader), 4))
        # train_acc.append(correct / total)
        return train_loss/ len(trainloader), round(100. * correct / total, 3)


    def test(epoch):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        global best_acc
        global best_loss
        global best_epoch
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = test_criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        cur_acc = round(100. * correct / total, 3)
        cur_loss = round(test_loss / len(testloader), 4)
        safe_print('epoch: %d' % epoch, '  Test Acc: %.3f' % cur_acc, 
        '  Test loss: %.4f' % cur_loss, ' Best info epoch %d, acc %.3f, loss %.4f' % (best_epoch, best_acc, best_loss))
        if cur_acc > best_acc:
            best_acc = cur_acc
            best_epoch = epoch
        if cur_loss < best_loss:
            best_loss = cur_loss
        # valid_acc.append(cur_acc)
        return cur_loss, cur_acc


    total_time = 0

    for epoch in range(args.num_epoch):
        if args.use_ddp:
            trainloader.sampler.set_epoch(epoch)
        # 4. For epoch-based implementation, update the corresponding learning rate schedule according to steps of this epoch
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.max_lr,
                                                        steps_per_epoch=len(trainloader),
                                                        epochs=args.num_epoch, div_factor=args.div_factor,
                                                        final_div_factor=args.final_div, pct_start=args.pct_start,
                                                        last_epoch=epoch * len(trainloader) - 1)
        end = time.time()
        train_loss, train_acc = train_info_batch(epoch) 
        total_time += time.time() - end
        test_loss, test_acc = test(epoch)
        if args.metrics:
            acc, auroc, aupr_success, aupr, fpr, aurc, eaurc, ece, nll, new_fpr = metrics.calc_metrics(testloader, net)
            metrics_data = {
                "epoch": epoch,
                "cutting rate" : args.ratio ,
                'train_time': total_time,
                "model": args.model,
                "train-loss": train_loss,  # Placeholder for actual loss value
                "train-acc": train_acc,  # Placeholder for actual accuracy value
                "test-loss": test_loss,  # Placeholder for actual loss value
                "test-acc": test_acc,  # Placeholder for actual accuracy value
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
            append_to_csv_file(metrics_data, 'test_cifar_example.csv')
            print('successfully save the metrics')
            

    if args.use_info_batch:
        safe_print('Total saved sample forwarding: ', trainset.get_pruned_count())
    safe_print('Total training time: ', total_time)
    
    # pref = 'full_batch' if not args.use_info_batch else 'info_batch'
    # fn = '{}{}-{}-epoch{}-batchsize{}-pct{}-labelsm{}-{}_cifar10_{}_log.json'.format(
    #     args.model,
    #     str(args.max_lr/args.div_factor),
    #     str(args.max_lr),
    #     args.num_epoch,
    #     args.batch_size,
    #     args.pct_start,
    #     args.label_smoothing,
    #     datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), pref)
    # if LOCAL_RANK in (-1, 0):
    #     file = open(fn, 'w+')
    #     json.dump([total_time, trainset.get_pruned_count() if args.use_info_batch else 0,
    #             args.ratio, train_acc, valid_acc], file)
    # if args.use_ddp:
    #     destroy_ddp()