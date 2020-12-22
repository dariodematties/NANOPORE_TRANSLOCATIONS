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

    model_names = ['ResNet10', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']

    optimizers = ['sgd', 'adam']

    parser = argparse.ArgumentParser(description='Nanopore Translocation Feature Prediction Training')
    parser.add_argument('data', metavar='DIR', nargs='*',
                        help='path(s) to dataset (if one path is provided, it is assumed\n' +
                       'to have subdirectories named "train" and "val"; alternatively,\n' +
                       'train and val paths can be specified directly by providing both paths as arguments)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='ResNet18',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: ResNet18_Custom)')
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
    parser.add_argument('-pth', '--plot-training-history', action='store_true',
                        help='Only plots the training history of a trained model: Loss and validation errors')

    args = parser.parse_args()
    return args


def main():
    global best_error, args
    best_error = math.inf
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
    if args.test:
        args.arch = 'ResNet10'

    if args.local_rank==0:
        print("=> creating model '{}'".format(args.arch))

    if args.arch == 'ResNet18':
        model = rn.ResNet18_Custom()
    elif args.arch == 'ResNet34':
        model = rn.ResNet34_Custom()
    elif args.arch == 'ResNet50':
        model = rn.ResNet50_Custom()
    elif args.arch == 'ResNet101':
        model = rn.ResNet101_Custom()
    elif args.arch == 'ResNet152':
        model = rn.ResNet152_Custom()
    elif args.arch == 'ResNet10':
        model = rn.ResNet10_Custom()
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




    loss_history = []
    duration_error_history = []
    amplitude_error_history = []
    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'" .format(args.resume))
                if args.cpu:
                    checkpoint = torch.load(args.resume)
                else:
                    checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))

                loss_history = checkpoint['loss_history']
                duration_error_history = checkpoint['duration_error_history']
                amplitude_error_history = checkpoint['amplitude_error_history']
                start_epoch = checkpoint['epoch']
                best_error = checkpoint['best_error']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                                .format(args.resume, checkpoint['epoch']))
                print("Model best precision saved was {}" .format(best_error))
                return start_epoch, best_error, model, optimizer, loss_history, duration_error_history, amplitude_error_history
            else:
                print("=> no checkpoint found at '{}'" .format(args.resume))
    
        args.start_epoch, best_error, model, optimizer, loss_history, duration_error_history, amplitude_error_history = resume()


    # Data loading code
    if len(args.data) == 1:
        traindir = os.path.join(args.data[0], 'train')
        valdir = os.path.join(args.data[0], 'val')
    else:
        traindir = args.data[0]
        valdir= args.data[1]

    if args.test:
        training_f = h5py.File(traindir + '/Pulse_var_Amp_Cnp_1027_train_toy.h5', 'r')
        validation_f = h5py.File(valdir + '/Pulse_var_Amp_Cnp_1027_validation_toy.h5', 'r')
    else:
        training_f = h5py.File(traindir + '/Pulse_var_Amp_Cnp_1027_train.h5', 'r')
        validation_f = h5py.File(valdir + '/Pulse_var_Amp_Cnp_1027_validation.h5', 'r')


    # this is the dataset for training
    sampling_rate = 10000                   # This is the number of samples per second of the signals in the dataset
    if args.test:
        number_of_concentrations = 2        # This is the number of different concentrations in the dataset
        number_of_durations = 2             # This is the number of different translocation durations per concentration in the dataset
        number_of_diameters = 4             # This is the number of different translocation durations per concentration in the dataset
        window = 0.5                        # This is the time window in seconds
        length = 20                         # This is the time of a complete signal for certain concentration and duration
    else:
        number_of_concentrations = 20       # This is the number of different concentrations in the dataset
        number_of_durations = 5             # This is the number of different translocation durations per concentration in the dataset
        number_of_diameters = 15            # This is the number of different translocation durations per concentration in the dataset
        window = 0.5                        # This is the time window in seconds
        length = 20                         # This is the time of a complete signal for certain concentration and duration

    # Training Artificial Data Loader
    TADL = Artificial_DataLoader(args.world_size, args.local_rank, device, training_f, sampling_rate,
                                 number_of_concentrations, number_of_durations, number_of_diameters,
                                 window, length, args.batch_size)


    # this is the dataset for validating
    if args.test:
        number_of_concentrations = 2        # This is the number of different concentrations in the dataset
        number_of_durations = 2             # This is the number of different translocation durations per concentration in the dataset
        number_of_diameters = 4             # This is the number of different translocation durations per concentration in the dataset
        window = 0.5                        # This is the time window in seconds
        length = 10                         # This is the time of a complete signal for certain concentration and duration
    else:
        number_of_concentrations = 20       # This is the number of different concentrations in the dataset
        number_of_durations = 5             # This is the number of different translocation durations per concentration in the dataset
        number_of_diameters = 15            # This is the number of different translocation durations per concentration in the dataset
        window = 0.5                        # This is the time window in seconds
        length = 10                         # This is the time of a complete signal for certain concentration and duration

    # Validating Artificial Data Loader
    VADL = Artificial_DataLoader(args.world_size, args.local_rank, device, validation_f, sampling_rate,
                                 number_of_concentrations, number_of_durations, number_of_diameters,
                                 window, length, args.batch_size)


    if args.verbose:
        print('From rank {} training shard size is {}'. format(args.local_rank, TADL.get_number_of_avail_windows()))
        print('From rank {} validation shard size is {}'. format(args.local_rank, VADL.get_number_of_avail_windows()))

    if args.evaluate:
        arguments = {'model': model,
                     'device': device,
                     'epoch': 0,
                     'VADL': VADL}

        validate(args, arguments)
        return

    if args.plot_training_history and args.local_rank == 0:
        Model_Util.plot_stats(loss_history, duration_error_history, amplitude_error_history)
        return


    total_time = Utilities.AverageMeter()
    for epoch in range(args.start_epoch, args.epochs):
        
        arguments = {'model': model,
                     'optimizer': optimizer,
                     'device': device,
                     'epoch': epoch,
                     'TADL': TADL,
                     'VADL': VADL,
                     'loss_history': loss_history,
                     'duration_error_history': duration_error_history,
                     'amplitude_error_history': amplitude_error_history}

        # train for one epoch
        avg_train_time = train(args, arguments)
        total_time.update(avg_train_time)

        # evaluate on validation set
        [duration_error, amplitude_error] = validate(args, arguments)

        error = (duration_error + amplitude_error) / 2

        #if args.test:
            #break

        # remember the best model and save checkpoint
        if args.local_rank == 0:
            print('From validation we have error is {} while best_error is {}'.format(error, best_error))
            is_best = error < best_error
            best_error = min(error, best_error)
            Model_Util.save_checkpoint({
                    'arch': args.arch,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_error': best_error,
                    'optimizer': optimizer.state_dict(),
                    'loss_history': loss_history,
                    'duration_error_history': duration_error_history,
                    'amplitude_error_history': amplitude_error_history,
            }, is_best)

            print('##Duration error {0}\n'
                  '##Amplitude error {1}\n'
                  '##Perf {2}'.format(
                  duration_error,
                  amplitude_error,
                  args.total_batch_size / total_time.avg))



















def train(args, arguments):
    batch_time = Utilities.AverageMeter()
    losses = Utilities.AverageMeter()

    # switch to train mode
    arguments['model'].train()
    end = time.time()

    i = 0
    arguments['TADL'].reset_avail_winds(arguments['epoch'])
    while i * arguments['TADL'].batch_size < arguments['TADL'].shard_size:
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

    arguments['loss_history'].append(losses.avg)

    return batch_time.avg




















def validate(args, arguments):
    batch_time = Utilities.AverageMeter()
    average_duration_error = Utilities.AverageMeter()
    average_amplitude_error = Utilities.AverageMeter()

    # switch to evaluate mode
    arguments['model'].eval()

    end = time.time()

    i = 0
    arguments['VADL'].reset_avail_winds(arguments['epoch'])
    while i * arguments['VADL'].batch_size < arguments['VADL'].shard_size:
        # bring a new batch
        times, noisy_signals, clean_signals, _, labels = arguments['VADL'].get_batch()
        
        mean = torch.mean(noisy_signals, 1, True)
        noisy_signals = noisy_signals-mean

        with torch.no_grad():
            noisy_signals = noisy_signals.unsqueeze(1)
            external = torch.reshape(labels[:,0],[arguments['VADL'].batch_size,1])
            outputs = arguments['model'](noisy_signals, external)
            noisy_signals = noisy_signals.squeeze(1)

            errors=abs((labels[:,1:] - outputs.data.to('cpu')*torch.Tensor([10**(-3), 10**(-10)]).repeat(arguments['VADL'].batch_size,1)) / labels[:,1:])*100
            errors=torch.mean(errors,dim=0)

            duration_error = errors[0]
            amplitude_error = errors[1]

            val_loader_len = int(math.ceil(arguments['VADL'].shard_size / args.batch_size))



        if args.distributed:
            reduced_duration_error = Utilities.reduce_tensor(duration_error.data, args.world_size)
            reduced_amplitude_error = Utilities.reduce_tensor(amplitude_error.data, args.world_size)
        else:
            reduced_duration_error = duration_error.data
            reduced_amplitude_error = amplitude_error.data


        average_duration_error.update(Utilities.to_python_float(reduced_duration_error), args.batch_size)
        average_amplitude_error.update(Utilities.to_python_float(reduced_amplitude_error), args.batch_size)

        # measure elapsed time
        batch_time.update((time.time() - end)/args.print_freq)
        end = time.time()

        if args.test:
            if i > 10:
                break

        if args.local_rank == 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Duration Error {dur_error.val:.4f} ({dur_error.avg:.4f})\t'
                  'Amplitude Error {amp_error.val:.4f} ({amp_error.avg:.4f})'.format(
                  i, val_loader_len,
                  args.world_size*args.batch_size/batch_time.val,
                  args.world_size*args.batch_size/batch_time.avg,
                  batch_time=batch_time,
                  dur_error=average_duration_error,
                  amp_error=average_amplitude_error))
        
        i += 1

    if not args.evaluate:
        arguments['duration_error_history'].append(average_duration_error.avg)
        arguments['amplitude_error_history'].append(average_amplitude_error.avg)

    return [average_duration_error.avg, average_amplitude_error.avg]

















if __name__ == '__main__':
    main()
