import argparse
import sys
import os
import shutil
import time
import math
import h5py

import torch
import torch.nn as nn
import torch.optim
import torchvision.transforms as transforms
import torch.nn.functional as F

import torch.nn.parallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np

sys.path.append('../ResNet')
import ResNet1d as rn
sys.path.append('../')
import Model_Util
import Utilities
from Dataset_Management import Artificial_DataLoader

def parse():

    model_names = ['ResNet18_Counter', 'ResNet18_Custom']

    optimizers = ['sgd', 'adam']

    parser = argparse.ArgumentParser(description='Nanopore Translocation Feature Prediction Training')
    parser.add_argument('data', metavar='DIR', nargs='*',
                        help='path(s) to dataset (if one path is provided, it is assumed\n' +
                       'to have subdirectories named "train" and "val"; alternatively,\n' +
                       'train and val paths can be specified directly by providing both paths as arguments)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='ResNet18_Custom',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: ResNet_Toy_Custom)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=6, type=int,
                        metavar='N', help='mini-batch size per process (default: 6)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='Initial learning rate.  Will be scaled by the value 1/learning rate modulator every learning-rate-scheduler-period epochs.')
    parser.add_argument('--lrs', '--learning-rate-scaling', default='linear', type=str,
                        metavar='LRS', help='Function to scale the learning rate value (default: \'linear\').')
    parser.add_argument('--lrm', '--learning-rate-modulator', default=0.1, type=float, metavar='MOD',
                        help='In the learning rate schedule, this is the value by which the learning rate will be multiplied every learning-rate-scheduler-period epochs (default: 0.1)')
    parser.add_argument('--lrsp', '--learning-rate-scheduler-period', default=100, type=int, metavar='PERIOD',
                        help='In the learning rate schedule, this is the number of epochs that has to pass in order to modulate the learning rate (default: 100)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--warmup_epochs', default=10, type=int, metavar='W',
                        help='Number of warmup epochs (default: 10)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--cpu', action='store_true',
                        help='Runs CPU based version of the workflow.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='provides additional details as to what the program is doing')
    parser.add_argument('--optimizer', default='adam', type=str, metavar='OPTIM',
                        choices=optimizers,
                        help='optimizer for training the network\n' +
                             'Choices are: ' +
                             ' | '.join(optimizers) +
                             ' (default: adam)')
    parser.add_argument('-t', '--test', action='store_true',
                        help='Launch test mode with preset arguments')

    args = parser.parse_args()
    return args


def main():
    global best_prec1, args
    best_prec1 = 0
    args = parse()


    if not len(args.data):
        raise Exception("error: No data set provided")


    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1


    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank

        if not args.cpu:
            torch.cuda.set_device(args.gpu)

        torch.distributed.init_process_group(backend='gloo',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    args.total_batch_size = args.world_size * args.batch_size

    # Set the device
    device = torch.device('cpu' if args.cpu else 'cuda:' + str(args.gpu))

    # create model
    if args.local_rank==0:
        print("=> creating model '{}'".format(args.arch))

    if args.arch == 'ResNet18_Counter':
        model = rn.ResNet18_Counter()
    elif args.arch == 'ResNet18_Custom':
        model = rn.ResNet18_Custom()
    else:
        print("Unrecognized {} architecture" .format(args.arch))


    model = model.to(device)

 
    # For distributed training, wrap the model with torch.nn.parallel.DistributedDataParallel.
    if args.distributed:
        if args.cpu:
            model = DDP(model)
        else:
            model = DDP(model, device_ids=[args.gpu], output_device=args.gpu)

        if args.verbose:
            print('Since we are in a distributed setting the model is replicated here in local rank {}'
                                    .format(args.local_rank))



    # Set optimizer
    optimizer = Model_Util.get_optimizer(model, args)
    if args.local_rank==0 and args.verbose:
        print('Optimizer used for this run is {}'.format(args.optimizer))




    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'" .format(args.resume))
                checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
                start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                                .format(args.resume, checkpoint['epoch']))
                print("Model best precision saved was {}" .format(best_prec1))
                return start_epoch, best_prec1, model, optimizer
            else:
                print("=> no checkpoint found at '{}'" .format(args.resume))
    
        args.start_epoch, best_prec1, model, optimizer = resume()


    # Data loading code
    if len(args.data) == 1:
        traindir = os.path.join(args.data[0], 'train')
        valdir = os.path.join(args.data[0], 'val')
    else:
        traindir = args.data[0]
        valdir= args.data[1]

    training_f = h5py.File(traindir + '/Pulse_var_Amp_Cnp_1027_train_toy.h5', 'r')
    validation_f = h5py.File(valdir + '/Pulse_var_Amp_Cnp_1027_validation_toy.h5', 'r')


    sampling_rate = 10000               # This is the number of samples per second of the signals in the dataset
    number_of_concentrations = 2        # This is the number of different concentrations in the dataset
    number_of_durations = 2             # This is the number of different translocation durations per concentration in the dataset
    number_of_diameters = 4             # This is the number of different translocation durations per concentration in the dataset
    window = 0.5                        # This is the time window in seconds
    length = 20                         # This is the time of a complete signal for certain concentration and duration

    #sampling_rate = 10000               # This is the number of samples per second of the signals in the dataset
    #number_of_concentrations = 20       # This is the number of different concentrations in the dataset
    #number_of_durations = 5             # This is the number of different translocation durations per concentration in the dataset
    #number_of_diameters = 15            # This is the number of different translocation durations per concentration in the dataset
    #window = 0.5                        # This is the time window in seconds
    #length = 20                         # This is the time of a complete signal for certain concentration and duration

    # Training Artificial Data Loader
    TADL = Artificial_DataLoader(args.world_size, args.local_rank, device, training_f, sampling_rate,
                                 number_of_concentrations, number_of_durations, number_of_diameters,
                                 window, length, args.batch_size)




    total_time = Utilities.AverageMeter()
    for epoch in range(args.start_epoch, args.epochs):
        
        arguments = {'model': model,
                     'optimizer': optimizer,
                     'device': device,
                     'epoch': epoch,
                     'TADL': TADL}

        # train for one epoch
        avg_train_time = train(args, arguments)
        total_time.update(avg_train_time)
        if args.test:
            break






















def train(args, arguments):
    batch_time = Utilities.AverageMeter()
    losses = Utilities.AverageMeter()

    # switch to train mode
    arguments['model'].train()
    end = time.time()

    i = 0
    arguments['TADL'].reset_avail_winds(arguments['epoch'])
    while arguments['TADL'].get_number_of_avail_windows() > args.batch_size:
        # get the noisy inputs and the labels
        _, inputs, _, _, labels = arguments['TADL'].get_batch()
        mean = torch.mean(inputs, 1, True)
        inputs = inputs-mean
            
        labels[:,1] = labels[:,1] * 10**3
        labels[:,2] = labels[:,2] * 10**10

        train_loader_len = int(math.ceil(arguments['TADL'].shard_size / args.batch_size))

        # zero the parameter gradients
        arguments['optimizer'].zero_grad()

        # forward + backward + optimize
        inputs = inputs.unsqueeze(1)
        external = torch.reshape(labels[:,0],[args.batch_size,1])
        outputs = arguments['model'](inputs, external)

        # Compute Huber loss
        loss = F.smooth_l1_loss(outputs, labels[:,1:])

        # Adjust learning rate
        Model_Util.learning_rate_schedule(args, arguments)

        # compute gradient and do SGD step
        loss.backward()
        arguments['optimizer'].step()

        if args.test:
            if i > 10:
                break

        if i%args.print_freq == 0:
            # Every print_freq iterations, check the loss and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Average loss across processes for logging
            if args.distributed:
                reduced_loss = Utilities.reduce_tensor(loss.data, args.world_size)
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(Utilities.to_python_float(reduced_loss), args.batch_size)

            if not args.cpu:
                torch.cuda.synchronize()

            batch_time.update((time.time() - end)/args.print_freq)
            end = time.time()

            if args.local_rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss {loss.val:.10f} ({loss.avg:.4f})'.format(
                      arguments['epoch'], i, train_loader_len,
                      args.world_size*args.batch_size/batch_time.val,
                      args.world_size*args.batch_size/batch_time.avg,
                      batch_time=batch_time,
                      loss=losses))

        i += 1

    return batch_time.avg




















if __name__ == '__main__':
    main()
