import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle
import torchvision
import torchvision.transforms as transforms
from models import resnet
from data.dataset import CIFAR10_loss
import argparse
from models.allconv import AllConvNet
from models.wrn import WideResNet

parser = argparse.ArgumentParser(description='PyTorch score calculation')

model_options = ['res18', 'wrn']
dataset_options = ['cifar10', 'cifar100', 'tiny-imagenet']
parser.add_argument('--dataset', '-d', default='cifar100', choices=dataset_options)
parser.add_argument('--model', '-a', default='res18', choices=model_options)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--load-dir', default='', type=str)
parser.add_argument('--save-dir', default='/lustre/home/xmwang/codes/code_cc_misd/scores/cifar100/res18-0.05', type=str)
parser.add_argument('--trials', default=10, type=int)
parser.add_argument('--cal-methods', default='EL2N', type=str)
# choices=['loss', 'EL2N', 'Grad']
args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ', device)


mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

transform_train = transforms.Compose([transforms.RandomHorizontalFlip(), 
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.ToTensor(), 
                                        transforms.Normalize(mean, std)])
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])# transform_test = transforms.Compose([])

# transform_test = transform_train

batch_size = 128
test_batch_size = 4
kwargs = {'num_workers': 0, 'pin_memory': True}

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if args.dataset == 'cifar10':
    print('use dataset ===========> cifar10')
    data_dir = '/lustre/home/xmwang/share/data/cifar10'

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    num_classes = 10
if args.dataset == 'cifar100':
    print('use dataset ===========> cifar100')

    data_dir = '/lustre/home/xmwang/share/data/cifar100'
    trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    
    testset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)

    num_classes = 100

# if args.model =='res18':
#     print('use model ===========> res18')

#     model = resnet.ResNet18(num_classes=num_classes).to(device)
#     if args.load_dir != "":
#         print('load the checkpoints')
#         model.load_state_dict(torch.load(args.load_dir))
# elif args.model == 'wrn':
#     print('use model ===========> wrn')
#     model = WideResNet(40, num_classes, widen_factor=2, dropRate=0.3).to(device)
#     if args.load_dir != "":
#         print('load the checkpoints')

#         model.load_state_dict(torch.load(args.load_dir))


def train(model, train_loader, test_loader):
    for epoch in range(20):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
        model.train()
        cretirion = nn.CrossEntropyLoss()
        loss_total = 0
        acc = 0
        num = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            out = model(data)
            loss = cretirion(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            # print('target', target.size(0)
            num += target.size(0)
            acc += (out.argmax(dim=1) == target).sum().item()
        print('epoch: ', epoch )
        print('loss: ', loss_total / len(train_loader))
        print('acc: ', acc/num)
        
    return model
    
        
        
    


def cal_scores(model, test_loader, id):

    if 'loss' in args.cal_methods:

        cretirion = nn.CrossEntropyLoss(reduction='none')

        loss_list = []
        for idx, (input, target) in enumerate(test_loader):
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            loss = cretirion(output, target)
            # print(loss.data)
            loss_list.extend(loss.data.cpu().numpy())
        # break
    # save the loss_list in a file
        with open(os.path.join(args.save_dir, f'loss_{idx}.pkl'), 'wb') as f:
            pickle.dump(loss_list, f)
        print('loss_list saved')

    if 'EL2N' in args.cal_methods:
        print('methods selectetd ==========> EL2N')
        model.eval()
        with torch.no_grad():
            model = model.to(device)
            score_list = []
            for idx, (input, target) in enumerate(test_loader):
                input = input.to(device)
                target = target.to(device)
                # print('target: ', target)
                target = F.one_hot(target, num_classes=num_classes).float()
                output = model(input)

                errors = F.softmax(output, dim=-1) - target
                # print('errors: ', errors.data)
                score = torch.linalg.norm(errors, ord=2, dim=-1)
                # print('scores: ', score.data.cpu().numpy())
                # score = score.unsqueeze(0) if input.shape[0] == 1 else score
                # print('score_list: ', score.data.cpu().numpy())
                score_list.extend(score.data.cpu().numpy())
                # print('score_list: ', score_list)
                # break
        with open(os.path.join(args.save_dir, f'EL2N_{id}.pkl'), 'wb') as f:
            pickle.dump(score_list, f)
        print('EL2N saved at', os.path.join(args.save_dir, f'EL2N_{id}.pkl'))

    if 'Grad' in args.cal_methods:

        print('methods selectetd ==========> Grad')
        model.eval()
        grad_list = []
        for idx, (input, target) in enumerate(test_loader):
            input = input.to(device)

            target = target.to(device)
            input.requires_grad = True
            output = model(input)
            loss = F.cross_entropy(output, target)
            model.zero_grad()
            loss.backward()
            grad = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
            # print('grad: ', grad.shape)
            # grad = grad.unsqueeze()
            grad = torch.linalg.norm(grad, ord=2, dim=-1)
            # print(grad.data.cpu().numpy())  
            # grad = grad.unsqueeze(0) if input.shape[0] == 1 else grad
            
            # print('grad: ', grad.data.cpu().numpy())
            # grad_list.append(grad.data.cpu().numpy())
            # print('grad_list: ', grad_list)
            # if idx > 10:
            #     break
        with open(os.path.join(args.save_dir, f'Grad_{idx}.pkl'), 'wb') as f:
            pickle.dump(grad_list, f)
        print('Grad_list saved')
    

if __name__ == '__main__':
    for idx in range(args.trials):
        
        if args.model =='res18':
            print('use model ===========> res18')

            model = resnet.ResNet18(num_classes=num_classes).to(device)
            if args.load_dir != "":
                print('load the checkpoints')
                model.load_state_dict(torch.load(args.load_dir))
        elif args.model == 'wrn':
            print('use model ===========> wrn')
            model = WideResNet(40, num_classes, widen_factor=2, dropRate=0.3).to(device)
            if args.load_dir != "":
                print('load the checkpoints')

                model.load_state_dict(torch.load(args.load_dir))

        print('=========================> Start Traing')
        model = train(model, train_loader, test_loader)
        print('=========================> Start Calculate')

        cal_scores(model, test_loader, idx)
        print(f'trial {idx} finished')
    
    
    score_list = []
    for idx in range(args.trials):
        with open(os.path.join(args.save_dir, f'EL2N_{idx}.pkl'), 'rb') as f:
            list = pickle.load(f)
            score_list.append(list)
    score_list = np.array(score_list)
    score_mean = np.mean(score_list, axis=0).tolist()
    
    with open(os.path.join(args.save_dir, 'EL2N_mean.pkl'), 'wb') as f:
        pickle.dump(score_mean, f)
    

