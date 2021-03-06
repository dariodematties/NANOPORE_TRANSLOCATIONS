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
import matplotlib.pyplot as plt

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
    parser.add_argument('-stats', '--statistics', dest='statistics', action='store_true',
                        help='Compute statistics about errors of a trained model on validation set')
    parser.add_argument('-r', '--run', dest='run', action='store_true',
                        help='Run a trained model and plots a batch of predictions in noisy signals')
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


    # Set learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lrsp,
                                                              args.lrm)



    total_time = Utilities.AverageMeter()
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
                    checkpoint = torch.load(args.resume, map_location='cpu')
                else:
                    checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))

                loss_history = checkpoint['loss_history']
                duration_error_history = checkpoint['duration_error_history']
                amplitude_error_history = checkpoint['amplitude_error_history']
                start_epoch = checkpoint['epoch']
                best_error = checkpoint['best_error']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                total_time = checkpoint['total_time']
                print("=> loaded checkpoint '{}' (epoch {})"
                                .format(args.resume, checkpoint['epoch']))
                print("Model best precision saved was {}" .format(best_error))
                return start_epoch, best_error, model, optimizer, lr_scheduler, loss_history, duration_error_history, amplitude_error_history, total_time 
            else:
                print("=> no checkpoint found at '{}'" .format(args.resume))
    
        args.start_epoch, best_error, model, optimizer, lr_scheduler, loss_history, duration_error_history, amplitude_error_history, total_time = resume()


    # Data loading code
    if len(args.data) == 1:
        traindir = os.path.join(args.data[0], 'train')
        valdir = os.path.join(args.data[0], 'val')
    else:
        traindir = args.data[0]
        valdir= args.data[1]

    if args.test:
        training_f = h5py.File(traindir + '/train_toy.h5', 'r')
        validation_f = h5py.File(valdir + '/validation_toy.h5', 'r')
    else:
        training_f = h5py.File(traindir + '/train.h5', 'r')
        validation_f = h5py.File(valdir + '/validation.h5', 'r')


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


    if args.run:
        arguments = {'model': model,
                     'device': device,
                     'epoch': 0,
                     'VADL': VADL}

        if args.local_rank == 0:
            run_model(args, arguments)

        return

    if args.statistics:
        arguments = {'model': model,
                     'device': device,
                     'epoch': 0,
                     'VADL': VADL}

        [duration_errors, amplitude_errors] = compute_error_stats(args, arguments)
        if args.local_rank == 0:
            plot_stats(VADL, duration_errors, amplitude_errors)

        return


    if args.evaluate:
        arguments = {'model': model,
                     'device': device,
                     'epoch': 0,
                     'VADL': VADL}

        [duration_error, amplitude_error] = validate(args, arguments)
        print('##Duration error {0}\n'
              '##Amplitude error {1}'.format(
              duration_error,
              amplitude_error))

        return

    if args.plot_training_history and args.local_rank == 0:
        Model_Util.plot_features_stats(loss_history, duration_error_history, amplitude_error_history)
        hours = int(total_time.sum / 3600)
        minutes = int((total_time.sum % 3600) / 60)
        seconds = int((total_time.sum % 3600) % 60)
        print('The total training time was {} hours {} minutes and {} seconds' .format(hours, minutes, seconds))
        hours = int(total_time.avg / 3600)
        minutes = int((total_time.avg % 3600) / 60)
        seconds = int((total_time.avg % 3600) % 60)
        print('while the average time during one epoch of training was {} hours {} minutes and {} seconds' .format(hours, minutes, seconds))
        return


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
        epoch_time, avg_batch_time = train(args, arguments)
        total_time.update(epoch_time)

        # evaluate on validation set
        [duration_error, amplitude_error] = validate(args, arguments)

        error = (duration_error + amplitude_error) / 2

        #if args.test:
            #break

        lr_scheduler.step()
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
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'total_time': total_time
            }, is_best)

            print('##Duration error {0}\n'
                  '##Amplitude error {1}\n'
                  '##Perf {2}'.format(
                  duration_error,
                  amplitude_error,
                  args.total_batch_size / avg_batch_time))



















def train(args, arguments):
    batch_time = Utilities.AverageMeter()
    losses = Utilities.AverageMeter()

    # switch to train mode
    arguments['model'].train()
    end = time.time()

    train_loader_len = int(math.ceil(arguments['TADL'].shard_size / args.batch_size))
    i = 0
    arguments['TADL'].reset_avail_winds(arguments['epoch'])
    while i * arguments['TADL'].batch_size < arguments['TADL'].shard_size:
        # get the noisy inputs and the labels
        _, inputs, _, _, labels = arguments['TADL'].get_batch()

        mean = torch.mean(inputs, 1, True)
        inputs = inputs-mean
            
        labels[:,1] = labels[:,1] * 10**3
        labels[:,2] = labels[:,2] * 10**10

        # zero the parameter gradients
        arguments['optimizer'].zero_grad()

        # forward + backward + optimize
        inputs = inputs.unsqueeze(1)
        external = torch.reshape(labels[:,0],[args.batch_size,1])
        outputs = arguments['model'](inputs, external)

        # Compute Huber loss
        loss = F.smooth_l1_loss(outputs, labels[:,1:])

        # Adjust learning rate
        #Model_Util.learning_rate_schedule(args, arguments)

        # compute gradient and do SGD step
        loss.backward()
        arguments['optimizer'].step()

        #if args.test:
            #if i > 10:
                #break

        if i%args.print_freq == 0 and i != 0:
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

            batch_time.update((time.time() - end)/args.print_freq, args.print_freq)
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

    return batch_time.sum, batch_time.avg




















def validate(args, arguments):
    batch_time = Utilities.AverageMeter()
    average_duration_error = Utilities.AverageMeter()
    average_amplitude_error = Utilities.AverageMeter()

    # switch to evaluate mode
    arguments['model'].eval()

    end = time.time()

    val_loader_len = int(math.ceil(arguments['VADL'].shard_size / args.batch_size))
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

            errors=abs((labels[:,1:].to('cpu') - outputs.data.to('cpu')*torch.Tensor([10**(-3), 10**(-10)]).repeat(arguments['VADL'].batch_size,1)) / labels[:,1:].to('cpu'))*100
            errors=torch.mean(errors,dim=0)

            duration_error = errors[0]
            amplitude_error = errors[1]



        if args.distributed:
            reduced_duration_error = Utilities.reduce_tensor(duration_error.data, args.world_size)
            reduced_amplitude_error = Utilities.reduce_tensor(amplitude_error.data, args.world_size)
        else:
            reduced_duration_error = duration_error.data
            reduced_amplitude_error = amplitude_error.data


        average_duration_error.update(Utilities.to_python_float(reduced_duration_error), args.batch_size)
        average_amplitude_error.update(Utilities.to_python_float(reduced_amplitude_error), args.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #if args.test:
            #if i > 10:
                #break

        if args.local_rank == 0 and i % args.print_freq == 0 and i != 0:
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




















def compute_error_stats(args, arguments):
    # switch to evaluate mode
    arguments['model'].eval()
    duration_errors = torch.zeros(arguments['VADL'].shape)
    amplitude_errors = torch.zeros(arguments['VADL'].shape)
    arguments['VADL'].reset_avail_winds(arguments['epoch'])
    for i in range(arguments['VADL'].total_number_of_windows):
        if i % args.world_size == args.local_rank:
            (Cnp, Duration, Dnp, window) = np.unravel_index(i, arguments['VADL'].shape)

            # bring a new window
            times, noisy_signals, clean_signals, _, labels = arguments['VADL'].get_signal_window(Cnp, Duration, Dnp, window)

            if labels[0] > 0:
                times = times.unsqueeze(0)
                noisy_signals = noisy_signals.unsqueeze(0)
                clean_signals = clean_signals.unsqueeze(0)
                labels = labels.unsqueeze(0)

                mean = torch.mean(noisy_signals, 1, True)
                noisy_signals = noisy_signals-mean

                with torch.no_grad():
                    noisy_signals = noisy_signals.unsqueeze(1)
                    external = torch.reshape(labels[:,0],[1,1])
                    outputs = arguments['model'](noisy_signals, external)
                    noisy_signals = noisy_signals.squeeze(1)

                    errors=abs((labels[:,1:].to('cpu') - outputs.data.to('cpu')*torch.Tensor([10**(-3), 10**(-10)]).repeat(1,1)) / labels[:,1:].to('cpu'))*100
                    errors=torch.mean(errors,dim=0)

                    duration_errors[Cnp, Duration, Dnp, window] = errors[0]
                    amplitude_errors[Cnp, Duration, Dnp, window] = errors[1]
            else:
                duration_errors[Cnp, Duration, Dnp, window] = torch.tensor(float('nan'))
                amplitude_errors[Cnp, Duration, Dnp, window] = torch.tensor(float('nan'))

        #if args.test:
            #if i > 10:
                #break

    if args.distributed:
        reduced_duration_error = Utilities.reduce_tensor_sum_dest(duration_errors.data, 0)
        reduced_amplitude_error = Utilities.reduce_tensor_sum_dest(amplitude_errors.data, 0)
    else:
        reduced_duration_error = duration_errors.data
        reduced_amplitude_error = amplitude_errors.data

    return [reduced_duration_error, reduced_amplitude_error]








def plot_stats(VADL, reduced_duration_error, reduced_amplitude_error):
    mean_duration_error = reduced_duration_error.numpy()
    mean_duration_error = np.nanmean(mean_duration_error, 3)

    std_duration_error = reduced_duration_error.numpy()
    std_duration_error = np.nanstd(std_duration_error, 3)

    mean_amplitude_error = reduced_amplitude_error.numpy()
    mean_amplitude_error = np.nanmean(mean_amplitude_error, 3)

    std_amplitude_error = reduced_amplitude_error.numpy()
    std_amplitude_error = np.nanstd(std_amplitude_error, 3)

    (Cnp, Duration, Dnp) = VADL.shape[:3]

    ave1 = []
    std1 = []
    ave2 = []
    std2 = []
    # setup the figure and axes for duration errors
    fig = plt.figure(figsize=(10, 2*Duration*3.2))
    for i in range(Duration):
        ave1.append(fig.add_subplot(Duration,2,2*i+1, projection='3d'))
        std1.append(fig.add_subplot(Duration,2,2*i+2, projection='3d'))

    # prepare the data
    _x = np.arange(Cnp)
    _y = np.arange(Dnp)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    width = depth = 1
    for i in range(Duration):
        top = mean_duration_error[:,i,:].ravel()
        bottom = np.zeros_like(top)
        ave1[i].bar3d(x, y, bottom, width, depth, top, shade=True)
        ave1[i].set_title('Mean Duration Error for Duration {}' .format(i+1))
        ave1[i].set_xlabel('Cnp')
        ave1[i].set_ylabel('Dnp')

        top = std_duration_error[:,i,:].ravel()
        bottom = np.zeros_like(top)
        std1[i].bar3d(x, y, bottom, width, depth, top, shade=True, color='r')
        std1[i].set_title('STD Duration Error for Duration {}' .format(i+1))
        std1[i].set_xlabel('Cnp')
        std1[i].set_ylabel('Dnp')

    plt.show()


    # setup the figure and axes for amplitude errors
    fig = plt.figure(figsize=(10, 2*Duration*3.2))
    for i in range(Duration):
        ave2.append(fig.add_subplot(Duration,2,2*i+1, projection='3d'))
        std2.append(fig.add_subplot(Duration,2,2*i+2, projection='3d'))

    # prepare the data
    _x = np.arange(Cnp)
    _y = np.arange(Dnp)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    width = depth = 1
    for i in range(Duration):
        top = mean_amplitude_error[:,i,:].ravel()
        bottom = np.zeros_like(top)
        ave2[i].bar3d(x+1, y+1, bottom, width, depth, top, shade=True)
        ave2[i].set_title('Mean Amplitude Error for Duration {}' .format(i+1))
        ave2[i].set_xlabel('Cnp')
        ave2[i].set_ylabel('Dnp')

        top = std_amplitude_error[:,i,:].ravel()
        bottom = np.zeros_like(top)
        std2[i].bar3d(x+1, y+1, bottom, width, depth, top, shade=True, color='r')
        std2[i].set_title('STD Amplitude Error for Duration {}' .format(i+1))
        std2[i].set_xlabel('Cnp')
        std2[i].set_ylabel('Dnp')

    plt.show()


    ave1 = []
    std1 = []
    ave2 = []
    std2 = []
    duration_error = reduced_duration_error.numpy()
    amplitude_error = reduced_amplitude_error.numpy()
    for i in range(Duration):
        ave1.append(np.nanmean(duration_error[:,i,:,:].ravel()))
        std1.append(np.nanstd(duration_error[:,i,:,:].ravel()))
        ave2.append(np.nanmean(amplitude_error[:,i,:,:].ravel()))
        std2.append(np.nanstd(amplitude_error[:,i,:,:].ravel()))


    fig, axs = plt.subplots(2, 2, figsize=(10,10))
    fig.tight_layout(pad=4.0)
    durations = [i for i in range(Duration)]

    axs[0,0].plot(durations,ave1)
    axs[0,0].set_title("Average duration error: {}" .format(np.nanmean(duration_error.ravel())))
    axs[0,0].set_xlabel("Duration")
    axs[0,0].set_ylabel("Average Error")

    axs[0,1].plot(durations,std1, color='r')
    axs[0,1].set_title("STD duration error")
    axs[0,1].set_xlabel("Duration")
    axs[0,1].set_ylabel("STD Error")

    axs[1,0].plot(durations,ave2)
    axs[1,0].set_title("Average amplitude error: {}" .format(np.nanmean(amplitude_error.ravel())))
    axs[1,0].set_xlabel("Duration")
    axs[1,0].set_ylabel("Average Error")

    axs[1,1].plot(durations,std2, color='r')
    axs[1,1].set_title("STD amplitude error")
    axs[1,1].set_xlabel("Duration")
    axs[1,1].set_ylabel("STD Error")

    plt.show()























def run_model(args, arguments):
    # switch to evaluate mode
    arguments['model'].eval()

    arguments['VADL'].reset_avail_winds(arguments['epoch'])

    # bring a new batch
    times, noisy_signals, clean_signals, _, labels = arguments['VADL'].get_batch()
    
    mean = torch.mean(noisy_signals, 1, True)
    noisy_signals = noisy_signals-mean

    with torch.no_grad():
        noisy_signals = noisy_signals.unsqueeze(1)
        external = torch.reshape(labels[:,0],[arguments['VADL'].batch_size,1])
        outputs = arguments['model'](noisy_signals, external)
        noisy_signals = noisy_signals.squeeze(1)


    times = times.cpu()
    noisy_signals = noisy_signals.cpu()
    clean_signals = clean_signals.cpu()
    labels = labels.cpu()


    if arguments['VADL'].batch_size < 21:
        fig, axs = plt.subplots(arguments['VADL'].batch_size, 1, figsize=(10,arguments['VADL'].batch_size*3))
        fig.tight_layout(pad=4.0)
        for i, batch_element in enumerate(range(arguments['VADL'].batch_size)):
            mean = torch.mean(noisy_signals[batch_element])
            axs[i].plot(times[batch_element],noisy_signals[batch_element]-mean)
            mean = torch.mean(clean_signals[batch_element])
            axs[i].plot(times[batch_element],clean_signals[batch_element]-mean)
            axs[i].set_title("Average translocation time: {}, presiction is {}\nAverage aplitude: {}, prediction is {}"
            .format(labels[batch_element,1], outputs[batch_element,0]*10**(-3),\
                        labels[batch_element,2], outputs[batch_element,1]*10**(-10)))
    else:
        print('This will not show more than 20 plots')

    plt.show()

    errors=abs((labels[:,1:] - outputs.data.to('cpu')*torch.Tensor([10**(-3), 10**(-10)]).repeat(arguments['VADL'].batch_size,1)) / labels[:,1:])*100
    av_errors=torch.mean(errors,dim=0)
    print("Average translocation time error: {0:.2f}\nAverage translocation amplitude error:{1:.2f}".format(av_errors[0], av_errors[1]))

















if __name__ == '__main__':
    main()
