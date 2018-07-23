import torch
import json
import numpy as np

from dataset import create_train_dataset, create_test_dataset
from my_snip.base import EpochDataset
from base_model.cifar_resnet18 import cifar_resnet18
from my_snip.config import MultiStageLearningRatePolicy, save_args
import torch.optim as optim
import time
from tensorboardX import SummaryWriter
import argparse
from my_snip.torch_checkpoint import save_checkpoint
from my_snip.clock import TrainClock, AvgMeter
from attack import IPGD
import torch.nn as nn
import os
from train import adversairal_train_one_epoch, adversarial_val
from collections import OrderedDict
from test import evalRoboustness
parser = argparse.ArgumentParser()
parser.add_argument('--weight_decay', default=5e-4, type = float, help='weight decay (default: 5e-4)')
parser.add_argument('--epochs', default=180, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (if has resume, this is not needed')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--exp', default='wide0', type = str, help = 'the name of this experiment')

parser.add_argument('--no_adv', action = 'store_true', default=False, help = 'if True, no adversarial training was used!')
parser.add_argument('--adv_freq', type = int, default=1, help = 'The frequencies of training one batch of adversarial examples')
parser.add_argument('--eps', default=8, type = int, help = 'the maximum boundary of adversarial perturbations')
parser.add_argument('--iter', default=20, type = int, help = 'the number of iterations take to generate adversarial examples for using IPGD')
parser.add_argument('--wide', default = 10, type = int, help = 'using wider resnet18')
args = parser.parse_args()


log_dir = os.path.join('../logs', args.exp)
exp_dir = os.path.join('../eexps', args.exp)
train_res_path = os.path.join(exp_dir, 'train_results.txt')
val_res_path = os.path.join(exp_dir, 'val_results.txt')
final_res_path = os.path.join(exp_dir, 'final_results.txt')
if not os.path.exists(exp_dir):
    os.mkdir(exp_dir)

save_args(args, exp_dir)
writer = SummaryWriter(log_dir)

clock = TrainClock()

#learning_rate_policy = [[80, 0.08],
#                        [50, 0.008],
#                        [50, 0.0008]
#                        ]
learning_rate_policy = [[80, 0.01],
                        [50, 0.001],
                        [50, 0.0001]
                        ]
get_learing_rate = MultiStageLearningRatePolicy(learning_rate_policy)

def adjust_learning_rate(optimzier, epoch):
    #global get_lea
    lr = get_learing_rate(epoch)
    for param_group in optimizer.param_groups:

        param_group['lr'] = lr

torch.backends.cudnn.benchmark = True
ds_train = create_train_dataset()

ds_val = create_test_dataset()

net = cifar_resnet18(expansion = args.wide)
net.cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr = get_learing_rate(0), momentum = 0.9, weight_decay=args.weight_decay)

PgdAttack = IPGD(eps = args.eps, sigma = args.eps // 2, nb_iter = args.iter, norm = np.inf)


best_prec = 0.0
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        check_point = torch.load(args.resume)
        args.start_epoch = check_point['epoch']
        net.load_state_dict(check_point['state_dict'])
        best_prec = check_point['best_prec']

        print('Modeled loaded from {} with metrics:'.format(args.resume))
        print(results)
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

clock.epoch = args.start_epoch
#for epoch in ds_train.epoch_generator():
while True:
    if clock.epoch > args.epochs:
        break
    adjust_learning_rate(optimizer, clock.epoch)

    Trainresults = adversairal_train_one_epoch(net, optimizer, ds_train, criterion, PgdAttack, clock, attack_freq = args.adv_freq, use_adv = not args.no_adv)
    Trainresults['epoch'] = clock.epoch
    with open(train_res_path, 'a') as f:
        json.dump(Trainresults, f)


    #val_epoch = next(ds_val.epoch_generator())
    Valresults = adversarial_val(net, ds_val, criterion, PgdAttack, clock, attack_freq = args.adv_freq)
    Valresults['epoch'] = clock.epoch
    with open(val_res_path, 'a') as f:
        f.write('\n')
        json.dump(Valresults, f)

    prec = Valresults['clean_acc']
    if prec > best_prec:
        best_prec = prec
        save_checkpoint(
            {"epoch": clock.epoch,
             'state_dict': net.state_dict(),
             'best_prec': best_prec}, is_best=True, prefix=exp_dir)
    else:
        save_checkpoint(
            {"epoch": clock.epoch,
             'state_dict': net.state_dict(),
             'best_prec': best_prec}, is_best=False, prefix=exp_dir)
    for name, val in Trainresults.items():
        vval = Valresults[name]

        writer.add_scalars(main_tag = name, tag_scalar_dict = {
            "Train": val,
            'Val': vval},
                           global_step = clock.epoch)

    if clock.epoch % 30 == 0 and clock.epoch > (args.epochs // 2):
        #val_epoch = next(ds_val.epoch_generator())

        roboustness = evalRoboustness(net, ds_val)
        with open(val_res_path, 'a') as f:
            f.write('\n')
            json.dump({'Roboustness': roboustness}, f)


print('Final prec: {:.2f} --- roboustness: {:.2f}'.format(Valresults['clean_acc'], roboustness))

with open(final_res_path, 'a') as f:
    f.write('\n')
    json.dump({
        "prec": Valresults['clean_acc'],
        'roboustness': roboustness
    }, f)
