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
from matplotlib.ticker import MaxNLocator

sys.path.append('../ResNet')
import ResNet1d as rn
sys.path.append('../')
import Model_Util
import Utilities
from Dataset_Management import Artificial_DataLoader

def parse():

    model_names = ['ResNet10', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']

    parser = argparse.ArgumentParser(description='Nanopore Translocation Feature Prediction Training')
    parser.add_argument('data', metavar='DIR', type=str,
                        help='path to validation dataset')
    parser.add_argument('counter', metavar='COUNTER', type=str,
                        help='path to translocation counter')
    parser.add_argument('predictor', metavar='PREDICTOR', type=str,
                        help='path to translocation feature predictor')
    parser.add_argument('--arch-1', '-a-1', metavar='ARCH_1', default='ResNet18',
                        choices=model_names,
                        help='model architecture for translocation counter: ' +
                        ' | '.join(model_names) +
                        ' (default: ResNet18)')
    parser.add_argument('--arch-2', '-a-2', metavar='ARCH_2', default='ResNet18',
                        choices=model_names,
                        help='model architecture for translocation feature predictions: ' +
                        ' | '.join(model_names) +
                        ' (default: ResNet18_Custom)')
    parser.add_argument('-b', '--batch-size', default=6, type=int,
                        metavar='N', help='mini-batch size per process (default: 6)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-save-stats', default='', type=str, metavar='STATS_PATH',
                        help='path to save the stats produced during validation (default: none)')
    parser.add_argument('-stats-from-file', default='', type=str, metavar='STATS_FROM_FILE',
                        help='path to load the stats produced during validation from a file (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-stats', '--statistics', dest='statistics', action='store_true',
                        help='Compute statistics about errors of a trained model on validation set')
    parser.add_argument('-out-stats', '--output-statistics', dest='output_statistics', action='store_true',
                        help='Compute statistics about outputs of a trained model on validation set')
    parser.add_argument('-r', '--run', dest='run', action='store_true',
                        help='Run a trained model and plots a batch of predictions in noisy signals')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--cpu', action='store_true',
                        help='Runs CPU based version of the workflow.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='provides additional details as to what the program is doing')
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

    if not len(args.counter):
        raise Exception("error: No path to counter model provided")

    if not len(args.predictor):
        raise Exception("error: No path to predictor model provided")


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

    # create model_1
    if args.test:
        args.arch_1 = 'ResNet10'

    if args.local_rank==0:
        print("=> creating model_1 '{}'".format(args.arch_1))

    if args.arch_1 == 'ResNet18':
        model_1 = rn.ResNet18_Counter()
    elif args.arch_1 == 'ResNet34':
        model_1 = rn.ResNet34_Counter()
    elif args.arch_1 == 'ResNet50':
        model_1 = rn.ResNet50_Counter()
    elif args.arch_1 == 'ResNet101':
        model_1 = rn.ResNet101_Counter()
    elif args.arch_1 == 'ResNet152':
        model_1 = rn.ResNet152_Counter()
    elif args.arch_1 == 'ResNet10':
        model_1 = rn.ResNet10_Counter()
    else:
        print("Unrecognized {} for translocations counter architecture" .format(args.arch_1))

    # create model_2
    if args.test:
        args.arch_2 = 'ResNet10'

    if args.local_rank==0:
        print("=> creating model_2 '{}'".format(args.arch_2))

    if args.arch_2 == 'ResNet18':
        model_2 = rn.ResNet18_Custom()
    elif args.arch_2 == 'ResNet34':
        model_2 = rn.ResNet34_Custom()
    elif args.arch_2 == 'ResNet50':
        model_2 = rn.ResNet50_Custom()
    elif args.arch_2 == 'ResNet101':
        model_2 = rn.ResNet101_Custom()
    elif args.arch_2 == 'ResNet152':
        model_2 = rn.ResNet152_Custom()
    elif args.arch_2 == 'ResNet10':
        model_2 = rn.ResNet10_Custom()
    else:
        print("Unrecognized {} for translocation feature prediction architecture" .format(args.arch_2))



    model_1 = model_1.to(device)
    model_2 = model_2.to(device)

 
    # For distributed training, wrap the model with torch.nn.parallel.DistributedDataParallel.
    if args.distributed:
        if args.cpu:
            model_1 = DDP(model_1)
            model_2 = DDP(model_2)
        else:
            model_1 = DDP(model_1, device_ids=[args.gpu], output_device=args.gpu)
            model_2 = DDP(model_2, device_ids=[args.gpu], output_device=args.gpu)

        if args.verbose:
            print('Since we are in a distributed setting the model is replicated here in local rank {}'
                                    .format(args.local_rank))


    total_time = Utilities.AverageMeter()

    # bring counter from a checkpoint
    if args.counter:
        # Use a local scope to avoid dangling references
        def bring_counter():
            if os.path.isfile(args.counter):
                print("=> loading counter '{}'" .format(args.counter))
                if args.cpu:
                    checkpoint = torch.load(args.counter, map_location='cpu')
                else:
                    checkpoint = torch.load(args.counter, map_location = lambda storage, loc: storage.cuda(args.gpu))

                loss_history_1 = checkpoint['loss_history']
                counter_error_history = checkpoint['Counter_error_history']
                best_error_1 = checkpoint['best_error']
                model_1.load_state_dict(checkpoint['state_dict'])
                total_time_1 = checkpoint['total_time']
                print("=> loaded counter '{}' (epoch {})"
                                .format(args.counter, checkpoint['epoch']))
                print("Model best precision saved was {}" .format(best_error_1))
                return best_error_1, model_1, loss_history_1, counter_error_history, total_time_1
            else:
                print("=> no counter found at '{}'" .format(args.counter))
    
        best_error_1, model_1, loss_history_1, counter_error_history, total_time_1 = bring_counter()
    else:
        raise Exception("error: No counter path provided")




    # bring predictor from a checkpoint
    if args.predictor:
        # Use a local scope to avoid dangling references
        def bring_predictor():
            if os.path.isfile(args.predictor):
                print("=> loading predictor '{}'" .format(args.predictor))
                if args.cpu:
                    checkpoint = torch.load(args.predictor, map_location='cpu')
                else:
                    checkpoint = torch.load(args.predictor, map_location = lambda storage, loc: storage.cuda(args.gpu))

                loss_history_2 = checkpoint['loss_history']
                duration_error_history = checkpoint['duration_error_history']
                amplitude_error_history = checkpoint['amplitude_error_history']
                best_error_2 = checkpoint['best_error']
                model_2.load_state_dict(checkpoint['state_dict'])
                total_time_2 = checkpoint['total_time']
                print("=> loaded predictor '{}' (epoch {})"
                                .format(args.predictor, checkpoint['epoch']))
                print("Model best precision saved was {}" .format(best_error_2))
                return best_error_2, model_2, loss_history_2, duration_error_history, amplitude_error_history, total_time_2 
            else:
                print("=> no predictor found at '{}'" .format(args.predictor))

        best_error_2, model_2, loss_history_2, duration_error_history, amplitude_error_history, total_time_2 = bring_predictor()
    else:
        raise Exception("error: No predictor path provided")




    # plots validation stats from a file
    if args.stats_from_file and args.local_rank == 0:
        # Use a local scope to avoid dangling references
        def bring_stats_from_file():
            if os.path.isfile(args.stats_from_file):
                print("=> loading stats from file '{}'" .format(args.stats_from_file))
                if args.cpu:
                    stats = torch.load(args.stats_from_file, map_location='cpu')
                else:
                    stats = torch.load(args.stats_from_file, map_location = lambda storage, loc: storage.cuda(args.gpu))

                count_errors = stats['count_errors']
                duration_errors = stats['duration_errors']
                amplitude_errors = stats['amplitude_errors']
                Cnp = stats['Cnp']
                Duration = stats['Duration']
                Dnp = stats['Dnp']
                Arch = stats['Arch']

                print("=> loaded stats '{}'" .format(args.stats_from_file))
                return count_errors, duration_errors, amplitude_errors, Cnp, Duration, Dnp, Arch 
            else:
                print("=> no stats found at '{}'" .format(args.stats_from_file))

        count_errors, duration_errors, amplitude_errors, Cnp, Duration, Dnp, Arch = bring_stats_from_file()
        plot_stats(Cnp, Duration, Dnp, count_errors, duration_errors, amplitude_errors)

        return





    # Data loading code
    valdir = os.path.join(args.data, 'test')

    if args.test:
        validation_f = h5py.File(valdir + '/test_toy.h5', 'r')
    else:
        validation_f = h5py.File(valdir + '/test.h5', 'r')


    # this is the dataset for validating
    sampling_rate = 10000                   # This is the number of samples per second of the signals in the dataset
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
        print('From rank {} validation shard size is {}'. format(args.local_rank, VADL.get_number_of_avail_windows()))


    if args.run:
        arguments = {'model_1': model_1,
                     'model_2': model_2,
                     'device': device,
                     'epoch': 0,
                     'VADL': VADL}

        if args.local_rank == 0:
            run_model(args, arguments)

        return

    if args.statistics:
        arguments = {'model_1': model_1,
                     'model_2': model_2,
                     'device': device,
                     'epoch': 0,
                     'VADL': VADL}

        [count_errors, duration_errors, amplitude_errors, improper_measures] = compute_error_stats(args, arguments)
        if args.local_rank == 0:
            (Cnp, Duration, Dnp) = VADL.shape[:3]
            plot_stats(Cnp, Duration, Dnp, count_errors, duration_errors, amplitude_errors)
            print("This backbone produces {} improper measures.\nImproper measures are produced when the ground truth establishes 0 number of pulses but the network predicts one or more pulses."\
                    .format(improper_measures))

            if args.save_stats:
                Model_Util.save_stats({'count_errors': count_errors,
                                       'duration_errors': duration_errors,
                                       'amplitude_errors': amplitude_errors,
                                       'Cnp': VADL.shape[0],
                                       'Duration': VADL.shape[1],
                                       'Dnp': VADL.shape[2],
                                       'Arch': args.arch_2},
                                       args.save_stats)

        return

    if args.output_statistics:
        arguments = {'model_1': model_1,
                     'model_2': model_2,
                     'device': device,
                     'epoch': 0,
                     'VADL': VADL}

        [counts, durations, amplitudes] = compute_output_stats(args, arguments)
        if args.local_rank == 0:
            (Cnp, Duration, Dnp) = VADL.shape[:3]
            plot_stats(Cnp, Duration, Dnp, counts, durations, amplitudes, Error=False)

            if args.save_stats:
                Model_Util.save_stats({'counts': counts,
                                       'durations': durations,
                                       'amplitudes': amplitudes,
                                       'Cnp': VADL.shape[0],
                                       'Duration': VADL.shape[1],
                                       'Dnp': VADL.shape[2],
                                       'Arch': args.arch_2},
                                       args.save_stats)

        return

























































def compute_error_stats(args, arguments, include_improper_on_error_computation=True):
    # switch to evaluate mode
    arguments['model_1'].eval()
    arguments['model_2'].eval()
    improper_measures = 0
    count_errors = torch.zeros(arguments['VADL'].shape)
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
                    num_of_pulses = arguments['model_1'](noisy_signals)
                    external = torch.reshape(num_of_pulses ,[1,1]).round()
                    outputs = arguments['model_2'](noisy_signals, external)
                    noisy_signals = noisy_signals.squeeze(1)

                    errors=abs((labels[:,1:].to('cpu') - outputs.data.to('cpu')*torch.Tensor([10**(-3), 10**(-10)]).repeat(1,1)) / labels[:,1:].to('cpu'))*100
                    errors=torch.mean(errors,dim=0)

                    duration_errors[Cnp, Duration, Dnp, window] = errors[0]
                    amplitude_errors[Cnp, Duration, Dnp, window] = errors[1]

                    error=abs((labels[:,0].to('cpu') - external.data.to('cpu')) / labels[:,0].to('cpu'))*100
                    error=torch.mean(error,dim=0)

                    count_errors[Cnp, Duration, Dnp, window] = error

            else:
                times = times.unsqueeze(0)
                noisy_signals = noisy_signals.unsqueeze(0)
                clean_signals = clean_signals.unsqueeze(0)
                labels = labels.unsqueeze(0)

                mean = torch.mean(noisy_signals, 1, True)
                noisy_signals = noisy_signals-mean

                with torch.no_grad():
                    noisy_signals = noisy_signals.unsqueeze(1)
                    num_of_pulses = arguments['model_1'](noisy_signals)
                    external = torch.reshape(num_of_pulses ,[1,1]).round()
                    noisy_signals = noisy_signals.squeeze(1)

                    if external.data.to('cpu') > 0.0:
                        if include_improper_on_error_computation:
                            count_errors[Cnp, Duration, Dnp, window] = 100.0
                            duration_errors[Cnp, Duration, Dnp, window] = 100.0
                            amplitude_errors[Cnp, Duration, Dnp, window] = 100.0
                        else:
                            count_errors[Cnp, Duration, Dnp, window] = torch.tensor(float('nan'))
                            duration_errors[Cnp, Duration, Dnp, window] = torch.tensor(float('nan'))
                            amplitude_errors[Cnp, Duration, Dnp, window] = torch.tensor(float('nan'))

                        improper_measures += 1
                    else:
                        count_errors[Cnp, Duration, Dnp, window] = 0.0
                        duration_errors[Cnp, Duration, Dnp, window] = 0.0
                        amplitude_errors[Cnp, Duration, Dnp, window] = 0.0

        #if args.test:
            #if i > 10:
                #break

    if args.distributed:
        reduced_count_error = Utilities.reduce_tensor_sum_dest(count_errors.data, 0)
        reduced_duration_error = Utilities.reduce_tensor_sum_dest(duration_errors.data, 0)
        reduced_amplitude_error = Utilities.reduce_tensor_sum_dest(amplitude_errors.data, 0)
    else:
        reduced_count_error = count_errors.data
        reduced_duration_error = duration_errors.data
        reduced_amplitude_error = amplitude_errors.data

    return [reduced_count_error, reduced_duration_error, reduced_amplitude_error, improper_measures]








def plot_stats(Cnp, Duration, Dnp, reduced_count, reduced_duration, reduced_amplitude, Error=True):
    plt.rcParams.update({'font.size': 25})
    fontsize=22
    
    mean_count = reduced_count.numpy()
    mean_count = np.nanmean(mean_count, 3)

    std_count = reduced_count.numpy()
    std_count = np.nanstd(std_count, 3)

    mean_duration = reduced_duration.numpy()
    mean_duration = np.nanmean(mean_duration, 3)

    std_duration = reduced_duration.numpy()
    std_duration = np.nanstd(std_duration, 3)

    mean_amplitude = reduced_amplitude.numpy()
    mean_amplitude = np.nanmean(mean_amplitude, 3)

    std_amplitude = reduced_amplitude.numpy()
    std_amplitude = np.nanstd(std_amplitude, 3)

    ave0 = []
    # setup the figure and axes for count errors
    fig = plt.figure(figsize=(2*Duration*3.2, 7))
    for i in range(Duration):
        ave0.append(fig.add_subplot(1,Duration,i+1, projection='3d'))

    # prepare the data
    _x = np.arange(Cnp)+1
    _y = np.arange(Dnp)+1
    x, y = np.meshgrid(_x, _y)
    width = depth = 1
    for i in range(Duration):
        top = mean_count[:,i,:]
        top = top.transpose()
        ave0[i].plot_surface(x, y, top, alpha=0.9)

        std_surface = std_count[:,i,:]
        top1 = top+std_surface.transpose()
        top2 = top-std_surface.transpose()
        ave0[i].plot_surface(x, y, top1, alpha=0.2, color='r')
        ave0[i].plot_surface(x, y, top2, alpha=0.2, color='r')

        if Error==True:
            ave0[i].set_title('Mean Count Error (Dur. {})' .format(i+1), fontsize=fontsize)
        else:
            ave0[i].set_title('Mean Count (Dur. {})' .format(i+1), fontsize=fontsize)

        ave0[i].set_xlabel('Cnp', fontsize=fontsize)
        ave0[i].set_ylabel('Dnp', fontsize=fontsize)
        ave0[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        ave0[i].yaxis.set_major_locator(MaxNLocator(integer=True))
        ave0[i].set_yticklabels([])
        ave0[i].set_xticklabels([])


    plt.show()


    ave1 = []
    # setup the figure and axes for count errors
    fig = plt.figure(figsize=(2*Duration*3.2, 7))
    for i in range(Duration):
        ave1.append(fig.add_subplot(1,Duration,i+1, projection='3d'))

    # prepare the data
    _x = np.arange(Cnp)+1
    _y = np.arange(Dnp)+1
    x, y = np.meshgrid(_x, _y)
    width = depth = 1
    for i in range(Duration):
        top = mean_duration[:,i,:]
        top = top.transpose()
        ave1[i].plot_surface(x, y, top, alpha=0.9)

        std_surface = std_duration[:,i,:]
        top1 = top+std_surface.transpose()
        top2 = top-std_surface.transpose()
        ave1[i].plot_surface(x, y, top1, alpha=0.2, color='r')
        ave1[i].plot_surface(x, y, top2, alpha=0.2, color='r')

        if Error==True:
            ave1[i].set_title('Mean Dur. Error (Dur. {})' .format(i+1), fontsize=fontsize)
        else:
            ave1[i].set_title('Mean Dur. (Dur. {})' .format(i+1), fontsize=fontsize)

        ave1[i].set_xlabel('Cnp', fontsize=fontsize)
        ave1[i].set_ylabel('Dnp', fontsize=fontsize)
        ave1[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        ave1[i].yaxis.set_major_locator(MaxNLocator(integer=True))
        ave1[i].set_yticklabels([])
        ave1[i].set_xticklabels([])


    plt.show()


    ave2 = []
    # setup the figure and axes for count errors
    fig = plt.figure(figsize=(2*Duration*3.2, 7))
    for i in range(Duration):
        ave2.append(fig.add_subplot(1,Duration,i+1, projection='3d'))

    # prepare the data
    _x = np.arange(Cnp)+1
    _y = np.arange(Dnp)+1
    x, y = np.meshgrid(_x, _y)
    width = depth = 1
    for i in range(Duration):
        top = mean_amplitude[:,i,:]
        top = top.transpose()
        ave2[i].plot_surface(x, y, top, alpha=0.9)

        std_surface = std_amplitude[:,i,:]
        top1 = top+std_surface.transpose()
        top2 = top-std_surface.transpose()
        ave2[i].plot_surface(x, y, top1, alpha=0.2, color='r')
        ave2[i].plot_surface(x, y, top2, alpha=0.2, color='r')

        if Error==True:
            ave2[i].set_title('Mean Amp. Error (Dur. {})' .format(i+1), fontsize=fontsize)
        else:
            ave2[i].set_title('Mean Amp. (Dur. {})' .format(i+1), fontsize=fontsize)

        ave2[i].set_xlabel('Cnp', fontsize=fontsize)
        ave2[i].set_ylabel('Dnp', fontsize=fontsize)
        ave2[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        ave2[i].yaxis.set_major_locator(MaxNLocator(integer=True))
        ave2[i].set_yticklabels([])
        ave2[i].set_xticklabels([])


    plt.show()



    plt.rcParams.update({'font.size': 20})
    fontsize=30


    ave0 = []
    std0 = []
    ave1 = []
    std1 = []
    ave2 = []
    std2 = []
    count = reduced_count.numpy()
    duration = reduced_duration.numpy()
    amplitude = reduced_amplitude.numpy()
    for i in range(Duration):
        ave0.append(np.nanmean(count[:,i,:,:].ravel()))
        std0.append(np.nanstd(count[:,i,:,:].ravel()))
        ave1.append(np.nanmean(duration[:,i,:,:].ravel()))
        std1.append(np.nanstd(duration[:,i,:,:].ravel()))
        ave2.append(np.nanmean(amplitude[:,i,:,:].ravel()))
        std2.append(np.nanstd(amplitude[:,i,:,:].ravel()))


    fig, axs = plt.subplots(3, 1, figsize=(10,15))
    fig.tight_layout(pad=4.0)
    durations = [i+1 for i in range(Duration)]

    axs[0].errorbar(durations,ave0,std0, linestyle='None', marker='o', linewidth=1.5, markeredgewidth=2.0, capsize=10)
    if Error==True:
        axs[0].set_title("Avg. count error: {:.2f}\nSTD: {:.2f}" .format(np.nanmean(count.ravel()),np.nanstd(count.ravel())), fontsize=fontsize)
    else:
        axs[0].set_title("Avg. count: {:.2f}\nSTD: {:.2f}" .format(np.nanmean(count.ravel()),np.nanstd(count.ravel())), fontsize=fontsize)

    if Error==True:
        axs[0].set_ylabel("Avg. Error", fontsize=fontsize)
    else:
        axs[0].set_ylabel("Average", fontsize=fontsize)

    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5) 

    axs[1].errorbar(durations,ave1,std1, linestyle='None', marker='o', linewidth=1.5, markeredgewidth=2.0, capsize=10)
    if Error==True:
        axs[1].set_title("Avg. duration error: {:.2f}\nSTD: {:.2f}" .format(np.nanmean(duration.ravel()),np.nanstd(duration.ravel())), fontsize=fontsize)
    else:
        axs[1].set_title("Avg. duration: {:.2f}\nSTD: {:.2f}" .format(np.nanmean(duration.ravel()),np.nanstd(duration.ravel())), fontsize=fontsize)

    if Error==True:
        axs[1].set_ylabel("Avg. Error", fontsize=fontsize)
    else:
        axs[1].set_ylabel("Average", fontsize=fontsize)

    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5) 

    axs[2].errorbar(durations,ave2,std2, linestyle='None', marker='o', linewidth=1.5, markeredgewidth=2.0, capsize=10)
    if Error==True:
        axs[2].set_title("Avg. amplitude error: {:.2f}\nSTD: {:.2f}" .format(np.nanmean(amplitude.ravel()),np.nanstd(amplitude.ravel())), fontsize=fontsize)
    else:
        axs[2].set_title("Avg. amplitude: {:.2f}\nSTD: {:.2f}" .format(np.nanmean(amplitude.ravel()),np.nanstd(amplitude.ravel())), fontsize=fontsize)

    axs[2].set_xlabel("Duration", fontsize=fontsize)
    if Error==True:
        axs[2].set_ylabel("Avg. Error", fontsize=fontsize)
    else:
        axs[2].set_ylabel("Average", fontsize=fontsize)

    axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[2].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5) 

    plt.show()
    
    print("Average count error: {}\nSTD: {}" .format(np.nanmean(count.ravel()),np.nanstd(count.ravel())))
    print("Average duration error: {}\nSTD: {}" .format(np.nanmean(duration.ravel()),np.nanstd(duration.ravel())))
    print("Average amplitude error: {}\nSTD: {}" .format(np.nanmean(amplitude.ravel()),np.nanstd(amplitude.ravel())))












    # setup the figure and axes for count errors
    fig = plt.figure(figsize=(2*3.2, 3*7))

    ave0 = fig.add_subplot(3,1,1, projection='3d')

    # prepare the data
    _x = np.arange(Cnp)+1
    _y = np.arange(Dnp)+1
    x, y = np.meshgrid(_x, _y)
    width = depth = 1

    top = np.nanmean(mean_count, 1)
    top = top.transpose()
    ave0.plot_surface(x, y, top, alpha=0.9)

    std_surface = np.nanstd(std_count, 1)
    top1 = top+std_surface.transpose()
    top2 = top-std_surface.transpose()
    ave0.plot_surface(x, y, top1, alpha=0.2, color='r')
    ave0.plot_surface(x, y, top2, alpha=0.2, color='r')

    if Error==True:
        ave0.set_title('Mean Count Error', fontsize=fontsize)
    else:
        ave0.set_title('Mean Count', fontsize=fontsize)

    ave0.set_xlabel('Cnp', fontsize=fontsize)
    ave0.set_ylabel('Dnp', fontsize=fontsize)
    ave0.xaxis.set_major_locator(MaxNLocator(integer=True))
    ave0.yaxis.set_major_locator(MaxNLocator(integer=True))
    ave0.set_yticklabels([])
    ave0.set_xticklabels([])

    ave1 = fig.add_subplot(3,1,2, projection='3d')

    # prepare the data
    _x = np.arange(Cnp)+1
    _y = np.arange(Dnp)+1
    x, y = np.meshgrid(_x, _y)
    width = depth = 1

    top = np.nanmean(mean_duration, 1)
    top = top.transpose()
    ave1.plot_surface(x, y, top, alpha=0.9)

    #std_surface = std_count[:,i,:]
    std_surface = np.nanstd(std_duration, 1)
    top1 = top+std_surface.transpose()
    top2 = top-std_surface.transpose()
    ave1.plot_surface(x, y, top1, alpha=0.2, color='r')
    ave1.plot_surface(x, y, top2, alpha=0.2, color='r')

    if Error==True:
        ave1.set_title('Mean Duration Error', fontsize=fontsize)
    else:
        ave1.set_title('Mean Duration', fontsize=fontsize)

    ave1.set_xlabel('Cnp', fontsize=fontsize)
    ave1.set_ylabel('Dnp', fontsize=fontsize)
    ave1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ave1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ave1.set_yticklabels([])
    ave1.set_xticklabels([])

    ave2 = fig.add_subplot(3,1,3, projection='3d')

    # prepare the data
    _x = np.arange(Cnp)+1
    _y = np.arange(Dnp)+1
    x, y = np.meshgrid(_x, _y)
    width = depth = 1

    top = np.nanmean(mean_amplitude, 1)
    top = top.transpose()
    ave2.plot_surface(x, y, top, alpha=0.9)

    #std_surface = std_count[:,i,:]
    std_surface = np.nanstd(std_amplitude, 1)
    top1 = top+std_surface.transpose()
    top2 = top-std_surface.transpose()
    ave2.plot_surface(x, y, top1, alpha=0.2, color='r')
    ave2.plot_surface(x, y, top2, alpha=0.2, color='r')

    if Error==True:
        ave2.set_title('Mean Amplitude Error', fontsize=fontsize)
    else:
        ave2.set_title('Mean Amplitude', fontsize=fontsize)

    ave2.set_xlabel('Cnp', fontsize=fontsize)
    ave2.set_ylabel('Dnp', fontsize=fontsize)
    ave2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ave2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ave2.set_yticklabels([])
    ave2.set_xticklabels([])





    plt.show()

































def run_model(args, arguments, include_improper_on_error_computation=True):
    plt.rcParams.update({'font.size': 14})
    
    fontsize=22

    # switch to evaluate mode
    arguments['model_1'].eval()
    arguments['model_2'].eval()

    arguments['VADL'].reset_avail_winds(arguments['epoch'])

    # bring a new batch
    times, noisy_signals, clean_signals, _, labels = arguments['VADL'].get_batch(descart_empty_windows=False)
    
    mean = torch.mean(noisy_signals, 1, True)
    noisy_signals = noisy_signals-mean

    with torch.no_grad():
        noisy_signals = noisy_signals.unsqueeze(1)
        num_of_pulses = arguments['model_1'](noisy_signals)
        zero_pulse_indices = torch.where(num_of_pulses.round()==0.0)[0].data
        external = torch.reshape(num_of_pulses ,[arguments['VADL'].batch_size,1]).round()
        outputs = arguments['model_2'](noisy_signals, external)
        noisy_signals = noisy_signals.squeeze(1)

    outputs[zero_pulse_indices,:] = 0.0

    times = times.cpu()
    noisy_signals = noisy_signals.cpu()
    clean_signals = clean_signals.cpu()
    labels = labels.cpu()


    if arguments['VADL'].batch_size < 21:
        fig, axs = plt.subplots(arguments['VADL'].batch_size, 1, figsize=(10,arguments['VADL'].batch_size*3))
        fig.tight_layout(pad=6.0)
        for i, batch_element in enumerate(range(arguments['VADL'].batch_size)):
            mean = torch.mean(noisy_signals[batch_element])
            axs[i].plot(times[batch_element],noisy_signals[batch_element]-mean)
            #mean = torch.mean(clean_signals[batch_element])
            #axs[i].plot(times[batch_element],clean_signals[batch_element]-mean)
            axs[i].set_title("Avg. duration: {:.2E}, prediction {:.2E}\nAvg. aplitude: {:.2E}, prediction {:.2E}\nNum. of pulses: {}, prediction {}."
            .format(labels[batch_element,1], outputs[batch_element,0]*10**(-3),\
                        labels[batch_element,2], outputs[batch_element,1]*10**(-10),\
                        round(labels[batch_element,0].item()), round(num_of_pulses[batch_element,0].item())), fontsize=fontsize)
            axs[i].set_yticklabels([])
            axs[i].set_xticklabels([])

    else:
        print('This will not show more than 20 plots')

    plt.show()


    count_error = 0.0
    duration_error = 0.0
    amplitude_error = 0.0
    measures = 0.0
    improper_measures = 0.0
    for i in range(arguments['VADL'].batch_size):
        if labels[i,0] == 0.0:
            if (i == zero_pulse_indices).any():
                measures += 1.0
            else:
                if include_improper_on_error_computation:
                    measures += 1.0
                    count_error += 100.0
                    duration_error += 100.0
                    amplitude_error += 100.0

                improper_measures += 1.0
        else:
            measures += 1.0
            count_error += abs((labels[i,0] - external[i,0].data.to('cpu')) / labels[i,0])*100
            duration_error += abs((labels[i,1] - outputs[i,0].data.to('cpu')*10**(-3)) / labels[i,1])*100
            amplitude_error += abs((labels[i,2] - outputs[i,1].data.to('cpu')*10**(-10)) / labels[i,2])*100


    print("Average translocation duration error: {0:.1f}%\nAverage translocation amplitude error: {1:.1f}%\nAverage translocation counter error: {2:.1f}%"\
            .format(duration_error.item()/measures, amplitude_error.item()/measures, count_error.item()/measures))

    print("In this batch we has {} improper measures.\nImproper measures are produced when the ground truth establishes 0 number of pulses but the network predicts one or more pulses."\
            .format(int(improper_measures)))





































def compute_output_stats(args, arguments):
    # switch to evaluate mode
    arguments['model_1'].eval()
    arguments['model_2'].eval()
    counts = torch.zeros(arguments['VADL'].shape)
    durations = torch.zeros(arguments['VADL'].shape)
    amplitudes = torch.zeros(arguments['VADL'].shape)
    arguments['VADL'].reset_avail_winds(arguments['epoch'])
    for i in range(arguments['VADL'].total_number_of_windows):
        if i % args.world_size == args.local_rank:
            (Cnp, Duration, Dnp, window) = np.unravel_index(i, arguments['VADL'].shape)

            # bring a new window
            times, noisy_signals, clean_signals, _, _ = arguments['VADL'].get_signal_window(Cnp, Duration, Dnp, window)

            times = times.unsqueeze(0)
            noisy_signals = noisy_signals.unsqueeze(0)
            clean_signals = clean_signals.unsqueeze(0)

            mean = torch.mean(noisy_signals, 1, True)
            noisy_signals = noisy_signals-mean

            with torch.no_grad():
                noisy_signals = noisy_signals.unsqueeze(1)
                num_of_pulses = arguments['model_1'](noisy_signals)
                external = torch.reshape(num_of_pulses ,[1,1]).round()
                outputs = arguments['model_2'](noisy_signals, external)
                noisy_signals = noisy_signals.squeeze(1)

                features=outputs.data.to('cpu')*torch.Tensor([10**(-3), 10**(-10)]).repeat(1,1)
                features=torch.mean(features,dim=0)

                durations[Cnp, Duration, Dnp, window] = features[0]
                amplitudes[Cnp, Duration, Dnp, window] = features[1]

                count=external.data.to('cpu')
                count=torch.mean(count,dim=0)

                counts[Cnp, Duration, Dnp, window] = count


        #if args.test:
            #if i > 10:
                #break

    if args.distributed:
        reduced_count = Utilities.reduce_tensor_sum_dest(counts.data, 0)
        reduced_duration = Utilities.reduce_tensor_sum_dest(durations.data, 0)
        reduced_amplitude = Utilities.reduce_tensor_sum_dest(amplitudes.data, 0)
    else:
        reduced_count = counts.data
        reduced_duration = durations.data
        reduced_amplitude = amplitudes.data

    return [reduced_count, reduced_duration, reduced_amplitude]

















if __name__ == '__main__':
    main()
