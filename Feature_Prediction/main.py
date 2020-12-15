import argparse
import os
import shutil
import time
import math

import torch
import torch.nn as nn
import torch.optim
import torchvision.transforms as transforms

import torch.nn.parallel
import torch.distributed as dist

import numpy as np



def parse():

    model_names = ['ResNet18_Counter', 'ResNet18_Custom']

    parser = argparse.ArgumentParser(description='Nanopore Translocation Feature Prediction Training')
    parser.add_argument('data', metavar='DIR', nargs='*',
                        help='path(s) to dataset (if one path is provided, it is assumed\n' +
                       'to have subdirectories named "train" and "val"; alternatively,\n' +
                       'train and val paths can be specified directly by providing both paths as arguments)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='ResNet18_Counter',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: ResNet18_Counter)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size per process (default: 16)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='Initial learning rate.  Will be scaled by the value 1/learning rate modulator every learning-rate-scheduler-period epochs.')
    parser.add_argument('--lrm', '--learning-rate-modulator', default=0.1, type=float, metavar='MOD',
                        help='In the learning rate schedule, this is the value by which the learning rate will be multiplied every learning-rate-scheduler-period epochs (default: 0.1)')
    parser.add_argument('--lrsp', '--learning-rate-scheduler-period', default=100, type=int, metavar='PERIOD',
                        help='In the learning rate schedule, this is the number of epochs that has to pass in order to modulate the learning rate (default: 100)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    args = parser.parse_args()
    return args


def main():
    global best_prec1, args
    best_prec1 = 0
    args = parse()


















if __name__ == '__main__':
    main()
