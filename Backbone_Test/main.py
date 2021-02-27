import argparse
import sys
import os
import shutil
import time
import math
import h5py
from random import randint

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
from Dataset_Management import Unlabeled_Real_DataLoader

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
    parser.add_argument('-tl', '--trace-length', default=10, type=int,
                        metavar='TL', help='length of a trace in the dataset in seconds (default: 10)')

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

                count_translocations = stats['count_translocations']
                duration_translocations = stats['duration_translocations']
                amplitude_translocations = stats['amplitude_translocations']
                Trace = stats['Trace']
                Arch = stats['Arch']

                print("=> loaded stats '{}'" .format(args.stats_from_file))
                return count_translocations, duration_translocations, amplitude_translocations, Trace, Arch 
            else:
                print("=> no stats found at '{}'" .format(args.stats_from_file))

        count_translocations, duration_translocations, amplitude_translocations, Trace, Arch = bring_stats_from_file()
        plot_stats(Trace, count_translocations, duration_translocations, amplitude_translocations)

        return








    # Data loading code
    testdir = args.data

    if args.test:
        test_f = h5py.File(testdir + '/test_toy.h5', 'r')
    else:
        test_f = h5py.File(testdir + '/test.h5', 'r')


    # this is the dataset for validating
    if args.test:
        num_of_traces = 2                   # This is the number of different traces in the dataset
        window = 0.5                        # This is the time window in seconds
        length = args.trace_length          # This is the time of a complete signal for certain concentration and duration
    else:
        num_of_traces = 6                   # This is the number of different traces in the dataset
        window = 0.5                        # This is the time window in seconds
        length = args.trace_length          # This is the time of a complete signal for certain concentration and duration

    # Validating Artificial Data Loader
    TRDL = Unlabeled_Real_DataLoader(device, test_f,
                                     num_of_traces,
                                     window, length)

    if args.run:
        arguments = {'model_1': model_1,
                     'model_2': model_2,
                     'device': device,
                     'TRDL': TRDL}

        if args.local_rank == 0:
            run_model(args, arguments)

        return

    if args.statistics:
        arguments = {'model_1': model_1,
                     'model_2': model_2,
                     'device': device,
                     'TRDL': TRDL}

        [count_translocations, duration_translocations, amplitude_translocations] = compute_value_stats(args, arguments)
        if args.local_rank == 0:
            Trace = TRDL.shape[0]
            plot_stats(Trace, count_translocations, duration_translocations, amplitude_translocations)

            if args.save_stats:
                Model_Util.save_stats({'count_translocations': count_translocations,
                                       'duration_translocations': duration_translocations,
                                       'amplitude_translocations': amplitude_translocations,
                                       'Trace': TRDL.shape[0],
                                       'Arch': args.arch_2},
                                       args.save_stats)

        return



























































def compute_value_stats(args, arguments, include_improper_on_error_computation=True):
    # switch to evaluate mode
    arguments['model_1'].eval()
    arguments['model_2'].eval()
    count_translocations = torch.zeros(arguments['TRDL'].shape)
    duration_translocations = torch.zeros(arguments['TRDL'].shape)
    amplitude_translocations = torch.zeros(arguments['TRDL'].shape)
    for i in range(arguments['TRDL'].total_number_of_windows):
        if i % args.world_size == args.local_rank:
            (trace, window) = np.unravel_index(i, arguments['TRDL'].shape)

            # bring a new window
            times, signals = arguments['TRDL'].get_signal_window(trace, window)

            times = times.unsqueeze(0)
            signals = signals.unsqueeze(0)

            mean = torch.mean(signals, 1, True)
            signals = signals-mean

            with torch.no_grad():
                signals = signals.unsqueeze(1)
                num_of_pulses = arguments['model_1'](signals)
                external = torch.reshape(num_of_pulses ,[1,1]).round()
                outputs = arguments['model_2'](signals, external)
                signals = signals.squeeze(1)

                values=outputs.data.to('cpu')
                values=torch.mean(values,dim=0)

                duration_translocations[trace, window] = values[0]
                amplitude_translocations[trace, window] = values[1]

                pulses=external.data.to('cpu')
                pulses=torch.mean(pulses,dim=0)

                count_translocations[trace, window] = pulses

        #if args.test:
            #if i > 10:
                #break

    if args.distributed:
        reduced_count_translocations = Utilities.reduce_tensor_sum_dest(count_translocations.data, 0)
        reduced_duration_translocations = Utilities.reduce_tensor_sum_dest(duration_translocations.data, 0)
        reduced_amplitude_translocations = Utilities.reduce_tensor_sum_dest(amplitude_translocations.data, 0)
    else:
        reduced_count_translocations = count_translocations.data
        reduced_duration_translocations = duration_translocations.data
        reduced_amplitude_translocations = amplitude_translocations.data

    return [reduced_count_translocations, reduced_duration_translocations, reduced_amplitude_translocations]








def plot_stats(Trace, reduced_count_translocations, reduced_duration_translocations, reduced_amplitude_translocations):
    ave0 = []
    std0 = []
    ave1 = []
    std1 = []
    ave2 = []
    std2 = []
    count_translocation = reduced_count_translocations.numpy()
    duration_translocation = reduced_duration_translocations.numpy()
    amplitude_translocation = reduced_amplitude_translocations.numpy()

    for i in range(Trace):
        ave0.append(np.mean(count_translocation[i,:]))
        std0.append(np.std(count_translocation[i,:]))
        ave1.append(np.mean(duration_translocation[i,:]))
        std1.append(np.std(duration_translocation[i,:]))
        ave2.append(np.mean(amplitude_translocation[i,:]))
        std2.append(np.std(amplitude_translocation[i,:]))


    fig, axs = plt.subplots(3, 1, figsize=(10,15))
    fig.tight_layout(pad=4.0)
    traces = [i+1 for i in range(Trace)]

    axs[0].errorbar(traces,ave0,std0, linestyle='None', marker='o', linewidth=1.0)
    axs[0].set_title("Average translocation count: {}" .format(np.mean(count_translocation)))
    axs[0].set_xlabel("Trace")
    axs[0].set_ylabel("Average")
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    axs[1].errorbar(traces,ave1,std1, linestyle='None', marker='o', linewidth=1.0)
    axs[1].set_title("Average translocation duration: {}" .format(np.mean(duration_translocation)))
    axs[1].set_xlabel("Trace")
    axs[1].set_ylabel("Average")
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    axs[2].errorbar(traces,ave2,std2, linestyle='None', marker='o', linewidth=1.0)
    axs[2].set_title("Average translocation amplitude: {}" .format(np.mean(amplitude_translocation)))
    axs[2].set_xlabel("Trace")
    axs[2].set_ylabel("Average")
    axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()

























def run_model(args, arguments):
    # switch to evaluate mode
    arguments['model_1'].eval()
    arguments['model_2'].eval()

    # bring a new batch
    times = []
    signals = []
    traces = []
    for i in range(args.batch_size):
        # bring a new window
        trace = randint(0, arguments['TRDL'].num_of_traces-1)
        window = randint(0, arguments['TRDL'].windows_per_trace-1)
        partial_times, partial_signals = arguments['TRDL'].get_signal_window(trace, window)

        times.append(partial_times)
        signals.append(partial_signals)
        traces.append(torch.tensor(trace))

    times=torch.stack(times)
    signals=torch.stack(signals)
    traces=torch.stack(traces)

    mean = torch.mean(signals, 1, True)
    signals = signals-mean

    with torch.no_grad():
        signals = signals.unsqueeze(1)
        num_of_pulses = arguments['model_1'](signals)
        zero_pulse_indices = torch.where(num_of_pulses.round()==0.0)[0].data
        external = torch.reshape(num_of_pulses ,[args.batch_size,1]).round()
        outputs = arguments['model_2'](signals, external)
        signals = signals.squeeze(1)

    outputs[zero_pulse_indices,:] = 0.0

    times = times.cpu()
    signals = signals.cpu()


    if args.batch_size < 21:
        fig, axs = plt.subplots(args.batch_size, 1, figsize=(10,args.batch_size*3))
        fig.tight_layout(pad=6.0)
        for i, batch_element in enumerate(range(args.batch_size)):
            mean = torch.mean(signals[batch_element])
            axs[i].plot(times[batch_element],signals[batch_element]-mean)
            axs[i].set_title("Average translocation time prediction is {}\nAverage aplitude prediction is {}\nNumber of pulses prediction is {}."
            .format(outputs[batch_element,0]*10**(-3),\
                    outputs[batch_element,1]*10**(-10),\
                    round(num_of_pulses[batch_element,0].item())))
    else:
        print('This will not show more than 20 plots')

    plt.show()

















if __name__ == '__main__':
    main()
