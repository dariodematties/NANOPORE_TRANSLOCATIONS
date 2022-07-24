import argparse
import sys
import os
import shutil
import time
import math
import h5py
import random

import torch
import torch.nn as nn
import torch.optim
import torchvision.transforms as transforms
import torch.nn.functional as F

import torch.nn.parallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter

sys.path.append('../ResNet')
import ResNet1d as rn
sys.path.append('../')
import Model_Util
import Utilities
from Dataset_Management import Labeled_Real_DataLoader

sys.path.append('../Translocations_Detector/models')
from backbone import build_backbone 
from transformer import build_transformer
import detr as DT

sys.path.append('./Evaluator')
from Evaluator import mean_average_precision_and_errors 

def parse():

    model_names = ['ResNet10', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']

    parser = argparse.ArgumentParser(description='Nanopore Translocation Detector Training')
    parser.add_argument('data', metavar='DIR', type=str,
                        help='path to experimental validation dataset')
    parser.add_argument('counter', metavar='COUNTER', type=str,
                        help='path to translocation counter')
    parser.add_argument('predictor', metavar='PREDICTOR', type=str,
                        help='path to translocation feature predictor')
    parser.add_argument('detector', metavar='DETECTOR', type=str,
                        help='path to translocation detector')
    parser.add_argument('--feature_predictor_arch', '-fpa', metavar='FEATURE_PREDICTOR_ARCH', default='ResNet18',
                        choices=model_names,
                        help='This is the architecture of the feature_predictor section in the backbone: ' +
                        ' | '.join(model_names) +
                        ' (default: ResNet18_Custom)')
    parser.add_argument('--pulse_counter_arch', '-pca', metavar='PULSE_COUNTER_ARCH', default='ResNet18',
                        choices=model_names,
                        help='This is the architecture of the pulse_counter section in the backbone: ' +
                        ' | '.join(model_names) +
                        ' (default: ResNet18_Counter)')
    parser.add_argument('-b', '--batch-size', default=6, type=int,
                        metavar='N', help='mini-batch size per process (default: 6)')
    parser.add_argument('-save-stats', default='', type=str, metavar='STATS_PATH',
                        help='path to save the stats produced during evaluation (default: none)')
    parser.add_argument('-stats', '--statistics', dest='statistics', action='store_true',
                        help='Compute statistics about contrast between a trained and a traditional model on validation set')
    parser.add_argument('-stats-from-file', default='', type=str, metavar='STATS_FROM_FILE',
                        help='path to load the stats produced during validation from a file (default: none)')
    parser.add_argument('-c', '--compute-predictions', default='', type=str, metavar='COMPUTE_PREDICTIONS',
                        help='Run a trained model and compute and save all its predictions in noisy traces')
    parser.add_argument('-r', '--run', dest='run', action='store_true',
                        help='Run a trained model and plots a window of predictions in a noisy trace')
    parser.add_argument('--run-plot-window', default=1.0, type=float, metavar='RPW',
                        help='the percentage of the window width the you want to actually plot (default: 1; which means 100%%)')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--cpu', action='store_true',
                        help='Runs CPU based version of the workflow.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='provides additional details as to what the program is doing')
    parser.add_argument('-t', '--test', action='store_true',
                        help='Launch test mode with preset arguments')

    parser.add_argument('--transformer-hidden-dim', default=512, type=int, metavar='TRANSFORMER-HIDDEN-DIM',
                        help='Hidden dimension of transformer on DETR model (default: 512)')
    parser.add_argument('--transformer-dropout', default=0.1, type=float, metavar='TRANSFORMER_DROPOUT',
                        help='Dropout of transformer on DETR model (default: 0.1)')
    parser.add_argument('--transformer-num-heads', default=8, type=int, metavar='TRANSFORMER_NUM_HEADS',
                        help='Number of heads of transformer on DETR model (default: 8)')
    parser.add_argument('--transformer-dim-feedforward', default=2048, type=int, metavar='TRANSFORMER_DIM_FEEDFORWARD',
                        help='Feedforward dimension inside transformer on DETR model (default: 2048)')
    parser.add_argument('--transformer-num-enc-layers', default=6, type=int, metavar='TRANSFORMER_NUM_ENC_LAYERS',
                        help='Number of encoder layers inside transformer on DETR model (default: 6)')
    parser.add_argument('--transformer-num-dec-layers', default=6, type=int, metavar='TRANSFORMER_NUM_DEC_LAYERS',
                        help='Number of decoder layers inside transformer on DETR model (default: 6)')
    parser.add_argument('--transformer-pre-norm', dest='transformer-pre-norm', action='store_true',
                        help='Configurization of transformer on DETR model (default: False)')

    parser.add_argument('--num-classes', default=1, type=int, metavar='NUM_CLASSES',
                        help='The number of different translocation classes that DETR has to classify (default: 1)')
    parser.add_argument('--num-queries', default=75, type=int, metavar='NUM_QUERIES',
                        help='The maximum number of translocations that DETR considers could exist in a window (default: 75)')

    parser.add_argument('--cost-class', default=1.0, type=float, metavar='COST_CLASS',
                        help='This is the relative weight of the classification error in the Hungarian matching cost (default: 1.0)')
    parser.add_argument('--cost-bsegment', default=1.0, type=float, metavar='COST_BSEGMENT',
                        help='This is the relative weight of the L1 error of the bounding segment coordinates in the Hungarian matching cost (default: 1.0)')
    parser.add_argument('--cost-giou', default=0.0, type=float, metavar='COST_GIOU',
                        help='This is the relative weight of the giou loss of the bounding segment in the Hungarian matching cost (default: 0.0)')

    parser.add_argument('--loss_ce', default=1.0, type=float, metavar='LOSS_CE',
                        help='This is the relative weight of the classification error in loss (default: 1.0)')
    parser.add_argument('--loss_bsegment', default=1.0, type=float, metavar='LOSS_BSEGMENT',
                        help='This is the relative weight of the L1 error of the bounding segment coordinates in loss (default: 1.0)')
    parser.add_argument('--loss_giou', default=0.0, type=float, metavar='LOSS_GIOU',
                        help='This is the relative weight of the giou loss of the bounding segment in the loss (default: 0.0)')
    parser.add_argument('--eos-coef', default=0.1, type=float, metavar='EOS_COEF',
                        help='This is relative classification weight applied to the no-translocation category in the loss (default: 0.1)')

    parser.add_argument('--start-threshold', default=0.5, type=float, metavar='START_THRESHOLD',
                        help='This is the start threshold for the mAP computation (default: 0.5)')
    parser.add_argument('--end-threshold', default=0.95, type=float, metavar='END_THRESHOLD',
                        help='This is the end threshold for the mAP computation (default: 0.95)')
    parser.add_argument('--step-threshold', default=0.05, type=float, metavar='STEP_THRESHOLD',
                        help='This is the step threshold for the mAP computation (default: 0.05)')

    parser.add_argument('--trace_number', default=0, type=int,
                        metavar='TN', help='trace number to plot (default: 0)')
    parser.add_argument('--window_number', default=0, type=int,
                        metavar='WN', help='window number to plot (default: 0)')

    args = parser.parse_args()
    return args


def main():
    global best_precision, args
    best_precision = 0
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





    #######################################################################
    #   Start DETR contruction
    #######################################################################

    # create DETR backbone

    # create backbone pulse counter
    if args.test:
        args.pulse_counter_arch = 'ResNet10'

    if args.local_rank==0 and args.verbose:
        print("=> creating backbone pulse counter '{}'".format(args.pulse_counter_arch))

    if args.pulse_counter_arch == 'ResNet18':
        backbone_pulse_counter = rn.ResNet18_Counter()
    elif args.pulse_counter_arch == 'ResNet34':
        backbone_pulse_counter = rn.ResNet34_Counter()
    elif args.pulse_counter_arch == 'ResNet50':
        backbone_pulse_counter = rn.ResNet50_Counter()
    elif args.pulse_counter_arch == 'ResNet101':
        backbone_pulse_counter = rn.ResNet101_Counter()
    elif args.pulse_counter_arch == 'ResNet152':
        backbone_pulse_counter = rn.ResNet152_Counter()
    elif args.pulse_counter_arch == 'ResNet10':
        backbone_pulse_counter = rn.ResNet10_Counter()
    else:
        print("Unrecognized {} architecture for the backbone pulse counter" .format(args.pulse_counter_arch))


    backbone_pulse_counter = backbone_pulse_counter.to(device)

    # create backbone feature predictor
    if args.test:
        args.feature_predictor_arch = 'ResNet10'

    if args.local_rank==0 and args.verbose:
        print("=> creating backbone feature predictor '{}'".format(args.feature_predictor_arch))

    if args.feature_predictor_arch == 'ResNet18':
        backbone_feature_predictor = rn.ResNet18_Custom()
    elif args.feature_predictor_arch == 'ResNet34':
        backbone_feature_predictor = rn.ResNet34_Custom()
    elif args.feature_predictor_arch == 'ResNet50':
        backbone_feature_predictor = rn.ResNet50_Custom()
    elif args.feature_predictor_arch == 'ResNet101':
        backbone_feature_predictor = rn.ResNet101_Custom()
    elif args.feature_predictor_arch == 'ResNet152':
        backbone_feature_predictor = rn.ResNet152_Custom()
    elif args.feature_predictor_arch == 'ResNet10':
        backbone_feature_predictor = rn.ResNet10_Custom()
    else:
        print("Unrecognized {} architecture for the backbone feature predictor" .format(args.feature_predictor_arch))


    backbone_feature_predictor = backbone_feature_predictor.to(device)



    # For distributed training, wrap the model with torch.nn.parallel.DistributedDataParallel.
    if args.distributed:
        if args.cpu:
            backbone_pulse_counter = DDP(backbone_pulse_counter)
            backbone_feature_predictor = DDP(backbone_feature_predictor)
        else:
            backbone_pulse_counter = DDP(backbone_pulse_counter, device_ids=[args.gpu], output_device=args.gpu)
            backbone_feature_predictor = DDP(backbone_feature_predictor, device_ids=[args.gpu], output_device=args.gpu)

        if args.verbose:
            print('Since we are in a distributed setting the backbone componets are replicated here in local rank {}'
                                    .format(args.local_rank))



    # bring counter from a checkpoint
    if args.counter:
        # Use a local scope to avoid dangling references
        def bring_counter():
            if os.path.isfile(args.counter):
                print("=> loading backbone pulse counter '{}'" .format(args.counter))
                if args.cpu:
                    checkpoint = torch.load(args.counter, map_location='cpu')
                else:
                    checkpoint = torch.load(args.counter, map_location = lambda storage, loc: storage.cuda(args.gpu))

                loss_history_1 = checkpoint['loss_history']
                counter_error_history = checkpoint['Counter_error_history']
                best_error_1 = checkpoint['best_error']
                backbone_pulse_counter.load_state_dict(checkpoint['state_dict'])
                total_time_1 = checkpoint['total_time']
                print("=> loaded counter '{}' (epoch {})"
                                .format(args.counter, checkpoint['epoch']))
                print("Counter best precision saved was {}" .format(best_error_1))
                return best_error_1, backbone_pulse_counter, loss_history_1, counter_error_history, total_time_1
            else:
                print("=> no counter found at '{}'" .format(args.counter))
    
        best_error_1, backbone_pulse_counter, loss_history_1, counter_error_history, total_time_1 = bring_counter()
    else:
        raise Exception("error: No counter path provided")




    # bring predictor from a checkpoint
    if args.predictor:
        # Use a local scope to avoid dangling references
        def bring_predictor():
            if os.path.isfile(args.predictor):
                print("=> loading backbone feature predictor '{}'" .format(args.predictor))
                if args.cpu:
                    checkpoint = torch.load(args.predictor, map_location='cpu')
                else:
                    checkpoint = torch.load(args.predictor, map_location = lambda storage, loc: storage.cuda(args.gpu))

                loss_history_2 = checkpoint['loss_history']
                duration_error_history = checkpoint['duration_error_history']
                amplitude_error_history = checkpoint['amplitude_error_history']
                best_error_2 = checkpoint['best_error']
                backbone_feature_predictor.load_state_dict(checkpoint['state_dict'])
                total_time_2 = checkpoint['total_time']
                print("=> loaded predictor '{}' (epoch {})"
                                .format(args.predictor, checkpoint['epoch']))
                print("Predictor best precision saved was {}" .format(best_error_2))
                return best_error_2, backbone_feature_predictor, loss_history_2, duration_error_history, amplitude_error_history, total_time_2 
            else:
                print("=> no predictor found at '{}'" .format(args.predictor))

        best_error_2, backbone_feature_predictor, loss_history_2, duration_error_history, amplitude_error_history, total_time_2 = bring_predictor()
    else:
        raise Exception("error: No predictor path provided")



    # create backbone
    if args.local_rank==0 and args.verbose:
        print("=> creating backbone")

    if args.feature_predictor_arch == 'ResNet18':
        backbone=build_backbone(pulse_counter=backbone_pulse_counter,
                                feature_predictor=backbone_feature_predictor,
                                num_channels=512)
    elif args.feature_predictor_arch == 'ResNet34':
        backbone=build_backbone(pulse_counter=backbone_pulse_counter,
                                feature_predictor=backbone_feature_predictor,
                                num_channels=512)
    elif args.feature_predictor_arch == 'ResNet50':
        backbone=build_backbone(pulse_counter=backbone_pulse_counter,
                                feature_predictor=backbone_feature_predictor,
                                num_channels=2048)
    elif args.feature_predictor_arch == 'ResNet101':
        backbone=build_backbone(pulse_counter=backbone_pulse_counter,
                                feature_predictor=backbone_feature_predictor,
                                num_channels=2048)
    elif args.feature_predictor_arch == 'ResNet152':
        backbone=build_backbone(pulse_counter=backbone_pulse_counter,
                                feature_predictor=backbone_feature_predictor,
                                num_channels=2048)
    elif args.feature_predictor_arch == 'ResNet10':
        backbone=build_backbone(pulse_counter=backbone_pulse_counter,
                                feature_predictor=backbone_feature_predictor,
                                num_channels=512)
    else:
        print("Unrecognized {} architecture for the backbone feature predictor" .format(args.feature_predictor_arch))


    backbone = backbone.to(device)











    # create DETR transformer
    if args.local_rank==0 and args.verbose:
        print("=> creating transformer")

    if args.test:
        args.transformer_hidden_dim = 64
        args.transformer_num_heads = 2
        args.transformer_dim_feedforward = 256
        args.transformer_num_enc_layers = 2
        args.transformer_num_dec_layers = 2

    args.transformer_pre_norm = True
    transformer = build_transformer(hidden_dim=args.transformer_hidden_dim,
                                    dropout=args.transformer_dropout,
                                    nheads=args.transformer_num_heads,
                                    dim_feedforward=args.transformer_dim_feedforward,
                                    enc_layers=args.transformer_num_enc_layers,
                                    dec_layers=args.transformer_num_dec_layers,
                                    pre_norm=args.transformer_pre_norm)






    # create DETR in itself
    if args.local_rank==0 and args.verbose:
        print("=> creating DETR")

    detr = DT.DETR(backbone=backbone,
                   transformer=transformer,
                   num_classes=args.num_classes,
                   num_queries=args.num_queries)

    detr = detr.to(device)

    # For distributed training, wrap the model with torch.nn.parallel.DistributedDataParallel.
    if args.distributed:
        if args.cpu:
            detr = DDP(detr)
        else:
            detr = DDP(detr, device_ids=[args.gpu], output_device=args.gpu)

        if args.verbose:
            print('Since we are in a distributed setting DETR model is replicated here in local rank {}'
                                    .format(args.local_rank))












    total_time = Utilities.AverageMeter()
    loss_history = []
    precision_history = []
    # bring detector from a checkpoint
    if args.detector:
        # Use a local scope to avoid dangling references
        def bring_detector():
            if os.path.isfile(args.detector):
                print("=> loading detector '{}'" .format(args.detector))
                if args.cpu:
                    checkpoint = torch.load(args.detector, map_location='cpu')
                else:
                    checkpoint = torch.load(args.detector, map_location = lambda storage, loc: storage.cuda(args.gpu))

                loss_history = checkpoint['loss_history']
                precision_history = checkpoint['precision_history']
                best_precision = checkpoint['best_precision']
                detr.load_state_dict(checkpoint['state_dict'])
                total_time = checkpoint['total_time']
                print("=> loaded checkpoint '{}' (epoch {})"
                                .format(args.detector, checkpoint['epoch']))
                print("Detector best precision saved was {}" .format(best_precision))
                return detr, loss_history, precision_history, total_time, best_precision 
            else:
                print("=> no checkpoint found at '{}'" .format(args.detector))
    
        detr, loss_history, precision_history, total_time, best_precision = bring_detector()
    else:
        raise Exception("error: No detector path provided")








    # plots validation stats from a file
    if args.stats_from_file:
        # Use a local scope to avoid dangling references
        def bring_stats_from_file():
            if os.path.isfile(args.stats_from_file):
                print("=> loading stats from file '{}'" .format(args.stats_from_file))
                if args.cpu:
                    stats = torch.load(args.stats_from_file, map_location='cpu')
                else:
                    stats = torch.load(args.stats_from_file, map_location = lambda storage, loc: storage.cuda(args.gpu))

                mAPs = stats['mAPs']
                mean_duration_errors = stats['mean_duration_errors']
                mean_start_time_errors = stats['mean_start_time_errors']
                mean_end_time_errors = stats['mean_end_time_errors']
                mean_duration_biases = stats['mean_duration_biases']
                mean_start_time_biases = stats['mean_start_time_biases']
                mean_end_time_biases = stats['mean_end_time_biases']
                mean_coverages = stats['mean_coverages']
                num_of_traces = stats['num_of_traces']

                print("=> loaded stats '{}'" .format(args.stats_from_file))
                return mAPs, mean_duration_errors, mean_start_time_errors, mean_end_time_errors,\
                             mean_duration_biases, mean_start_time_biases, mean_end_time_biases, mean_coverages,\
                             num_of_traces 
            else:
                print("=> no stats found at '{}'" .format(args.stats_from_file))

        mAPs, mean_duration_errors, mean_start_time_errors, mean_end_time_errors,\
              mean_duration_biases, mean_start_time_biases, mean_end_time_biases, mean_coverages, num_of_traces = bring_stats_from_file()
        plot_error_stats(num_of_traces, mAPs, mean_duration_errors, mean_start_time_errors, mean_end_time_errors, mean_coverages)
        plot_bias_stats(num_of_traces, mAPs, mean_duration_biases, mean_start_time_biases, mean_end_time_biases, mean_coverages)

        return










    # Data loading code
    testdir = os.path.join(args.data)

    if args.test:
        test_f = h5py.File(testdir + '/test_toy.h5', 'r')
    else:
        test_f = h5py.File(testdir + '/test.h5', 'r')


    # this is the dataset for testing
    sampling_rate = 100000                   # This is the number of samples per second of the signals in the dataset
    if args.test:
        number_of_traces = 2                # This is the number of different traces in the dataset
        window = 0.5                        # This is the time window in seconds
        length = 71                         # This is the time of a complete signal for certain concentration and duration
        #length = 126                         # This is the time of a complete signal for certain concentration and duration
    else:
        number_of_traces = 6                # This is the number of different traces in the dataset
        window = 0.5                        # This is the time window in seconds
        #window = 0.05                        # This is the time window in seconds
        #length = 71                         # This is the time of a complete signal for certain concentration and duration
        length = 126                         # This is the time of a complete signal for certain concentration and duration

    # Testing Artificial Data Loader
    TADL = Labeled_Real_DataLoader(device, test_f, number_of_traces, window, length)

    if args.verbose:
        print('From rank {} test data set loaded'. format(args.local_rank))












    if args.run:
        arguments = {'model': detr,
                     'device': device,
                     'epoch': 0,
                     'TADL': TADL,
                     'trace': args.trace_number,
                     'window': args.window_number}

        if args.local_rank == 0:
            run_model(args, arguments)

        return






    if args.compute_predictions:
        arguments = {'model': detr,
                     'pulse_counter': backbone_pulse_counter,
                     'device': device,
                     'epoch': 0,
                     'TADL': TADL,
                     'trace': args.trace_number,
                     'window': args.window_number}

        if args.local_rank == 0:
            starts, ends = compute_predictions(args, arguments)
        
        assert len(starts) == len(ends)
        pulse_starts = np.empty((len(starts),), dtype=np.object)
        pulse_ends = np.empty((len(ends),), dtype=np.object)
        for i in range(len(starts)):
            pulse_starts[i] = starts[i]
            pulse_ends[i] = ends[i]
    
        directory = os.path.join(args.compute_predictions)
        if not os.path.exists(directory):
            os.mkdir(directory)

        savemat(directory + 'predictions.mat', {"pulse_starts":pulse_starts, "pulse_ends":pulse_ends})

        return






    if args.statistics:
        arguments = {'model': detr,
                     'device': device,
                     'epoch': 0,
                     'TADL': TADL}

        [mAPs, mean_duration_errors, mean_start_time_errors, mean_end_time_errors,\
               mean_duration_biases, mean_start_time_biases, mean_end_time_biases, mean_coverages,\
         mAPs_I, mean_duration_errors_I, mean_start_time_errors_I, mean_end_time_errors_I,\
               mean_duration_biases_I, mean_start_time_biases_I, mean_end_time_biases_I, mean_coverages_I] = compute_error_stats(args, arguments)
        if args.save_stats:
            Model_Util.save_stats({'mAPs': mAPs,
                                   'mean_duration_errors': mean_duration_errors,
                                   'mean_start_time_errors': mean_start_time_errors,
                                   'mean_end_time_errors': mean_end_time_errors,
                                   'mean_duration_biases': mean_duration_biases,
                                   'mean_start_time_biases': mean_start_time_biases,
                                   'mean_end_time_biases': mean_end_time_biases,
                                   'mean_coverages': mean_coverages,
                                   'num_of_traces': TADL.num_of_traces,
                                   'Arch': 'DETR_' + args.feature_predictor_arch},
                                   args.save_stats, filename='Direct.pth.tar')

            Model_Util.save_stats({'mAPs': mAPs_I,
                                   'mean_duration_errors': mean_duration_errors_I,
                                   'mean_start_time_errors': mean_start_time_errors_I,
                                   'mean_end_time_errors': mean_end_time_errors_I,
                                   'mean_duration_biases': mean_duration_biases_I,
                                   'mean_start_time_biases': mean_start_time_biases_I,
                                   'mean_end_time_biases': mean_end_time_biases_I,
                                   'mean_coverages': mean_coverages_I,
                                   'num_of_traces': TADL.num_of_traces,
                                   'Arch': 'DETR_' + args.feature_predictor_arch},
                                   args.save_stats, filename='Indirect.pth.tar')

        return






































def run_model(args, arguments):
    plt.rcParams.update({'font.size': 14})

    # switch to evaluate mode
    arguments['model'].eval()

    # bring a new batch
    times, noisy_signals, targets, labels = arguments['TADL'].get_signal_window(arguments['trace'], arguments['window'])
    
    times = times.unsqueeze(0)
    noisy_signals = noisy_signals.unsqueeze(0)
    mean = torch.mean(noisy_signals, 1, True)
    noisy_signals = noisy_signals-mean

    with torch.no_grad():
        noisy_signals = noisy_signals.unsqueeze(1)
        outputs = arguments['model'](noisy_signals)
        noisy_signals = noisy_signals.squeeze(1)


    times = times.cpu()
    noisy_signals = noisy_signals.cpu()
    targets = targets.cpu()
    labels = labels.cpu()
    
    if args.run_plot_window < 1.0:
        width=int(args.run_plot_window*times[0].shape[0])
        start=random.randrange(0,times[0].shape[0])
        end=min(start+width,times[0].shape[0]-1)
    else:
        start=0
        end=times[0].shape[0]-1

    fig, axs = plt.subplots(1, 1, figsize=(10,1.5*3))
    fig.tight_layout(pad=4.0)

    # indices to be eliminated from the output (i.e. non-segments)
    idxs = torch.where(outputs['pred_logits'][0, :, :].argmax(-1) != 1)[0]
    segments=outputs['pred_segments'][0,idxs,:].detach()

    axs.plot(times[0][start:end],noisy_signals[0][start:end])

    x_points = (segments[:,0] * arguments['TADL'].window + times[0,0]).cpu().detach().numpy()
    to_delete = []
    for x_point in x_points:
        if not (times[0][start] <= x_point and x_point <= times[0][end]):
            to_delete.append(np.where(x_points==x_point)[0][0])
            
    x_points = np.delete(x_points, to_delete)
    y_points = np.repeat(0.5, len(x_points))
  
    axs.plot(x_points, y_points, 'r*')

    x_points = ((segments[:,1] + segments[:,0]) * arguments['TADL'].window + times[0,0]).cpu().detach().numpy()
    to_delete = []
    for x_point in x_points:
        if not (times[0][start] <= x_point and x_point <= times[0][end]):
            to_delete.append(np.where(x_points==x_point)[0][0])
            
    x_points = np.delete(x_points, to_delete)
    y_points = np.repeat(0.5, len(x_points))
  
    axs.plot(x_points, y_points, 'g*')

    # indices to be eliminated from the targets (i.e. non-segments)
    segments=targets[:, :].detach()

    x_points = (segments[0,:] * arguments['TADL'].window + times[0,0]).cpu().detach().numpy()
    to_delete = []
    for x_point in x_points:
        if not (times[0][start] <= x_point and x_point <= times[0][end]):
            to_delete.append(np.where(x_points==x_point)[0][0])
            
    x_points = np.delete(x_points, to_delete)
    y_points = np.repeat(0.25, len(x_points))
  
    axs.plot(x_points, y_points, 'ro')

    x_points = ((segments[0,:] + segments[1,:]) * arguments['TADL'].window + times[0,0]).cpu().detach().numpy()
    to_delete = []
    for x_point in x_points:
        if not (times[0][start] <= x_point and x_point <= times[0][end]):
            to_delete.append(np.where(x_points==x_point)[0][0])
        
    x_points = np.delete(x_points, to_delete)
    y_points = np.repeat(0.25, len(x_points))
  
    axs.plot(x_points, y_points, 'go')





        
    axs.set_xlabel("Time [s]", fontsize=22)
    axs.set_xticklabels((times[0]-times[0][0]).tolist(), fontsize=18)
    axs.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
    axs.set_ylabel("Current [nA]", fontsize=22)
            
    axs.set_yticklabels(noisy_signals[0].tolist(), fontsize=18)
    axs.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    plt.show()









def compute_error_stats(args, arguments):
    # switch to evaluate mode
    arguments['model'].eval()
    
    (traces, windows) = (arguments['TADL'].num_of_traces, arguments['TADL'].windows_per_trace)
    Ths = np.arange(args.start_threshold, args.end_threshold, args.step_threshold).shape[0]

    # performance metrics taken the targets as ground truth
    mAPs = torch.zeros((traces, Ths))
    mean_duration_errors = torch.zeros((traces, Ths))
    mean_start_time_errors = torch.zeros((traces, Ths))
    mean_end_time_errors = torch.zeros((traces, Ths))
    mean_duration_biases = torch.zeros((traces, Ths))
    mean_start_time_biases = torch.zeros((traces, Ths))
    mean_end_time_biases = torch.zeros((traces, Ths))
    mean_coverages = torch.zeros((traces, Ths))

    # performance metrics taken the predictions as ground truth I stands for Inverted
    mAPs_I = torch.zeros((traces, Ths))
    mean_duration_errors_I = torch.zeros((traces, Ths))
    mean_start_time_errors_I = torch.zeros((traces, Ths))
    mean_end_time_errors_I = torch.zeros((traces, Ths))
    mean_duration_biases_I = torch.zeros((traces, Ths))
    mean_start_time_biases_I = torch.zeros((traces, Ths))
    mean_end_time_biases_I = torch.zeros((traces, Ths))
    mean_coverages_I = torch.zeros((traces, Ths))
    
    for trace in range(traces):
        pred_segments = []
        true_segments = []
        for window in range(windows):
            # bring a new window
            times, noisy_signals, targets, labels = arguments['TADL'].get_signal_window(trace, window)

            if labels[0] > 0:
                times = times.unsqueeze(0)
                noisy_signals = noisy_signals.unsqueeze(0)
                targets = targets.unsqueeze(0)
                labels = labels.unsqueeze(0)

                mean = torch.mean(noisy_signals, 1, True)
                noisy_signals = noisy_signals-mean

                with torch.no_grad():
                    # forward
                    noisy_signals = noisy_signals.unsqueeze(1)
                    outputs = arguments['model'](noisy_signals)
                    noisy_signals = noisy_signals.squeeze(1)
                    
                train_idx = window
                
                probabilities = F.softmax(outputs['pred_logits'][0], dim=1)
                aux_pred_segments = outputs['pred_segments'][0]

                for probability, pred_segment in zip(probabilities.to('cpu'), aux_pred_segments.to('cpu')):
                    #if probability[-1] < 0.9:
                    if torch.argmax(probability) != args.num_classes:
                        segment = [train_idx, np.argmax(probability[:-1]).item(), 1.0 - probability[-1].item(),\
                                   pred_segment[0].item(), pred_segment[1].item()]
                        pred_segments.append(segment)
                        
                num_pulses = labels[0, 0]
    
                starts = targets[0, 0]
                widths = targets[0, 1]

                for k in range(int(num_pulses.item())):
                    segment = [train_idx, 0, 1.0, starts[k].item(), widths[k].item()]
                    true_segments.append(segment)


        for threshold in np.arange(args.start_threshold, args.end_threshold, args.step_threshold):
            errors = mean_average_precision_and_errors(device=arguments['device'],
                                                       pred_segments=pred_segments,                                                             
                                                       true_segments=true_segments,
                                                       iou_threshold=threshold,
                                                       seg_format="mix",
                                                       num_classes=1)
                                                                                   
            threshold_idx = np.where(np.arange(args.start_threshold, args.end_threshold, args.step_threshold)==threshold)[0][0]

            mAPs[trace, threshold_idx] = errors[0]
            mean_duration_errors[trace, threshold_idx] = errors[1]
            mean_start_time_errors[trace, threshold_idx] = errors[2]
            mean_end_time_errors[trace, threshold_idx] = errors[3]
            mean_duration_biases[trace, threshold_idx] = errors[4]
            mean_start_time_biases[trace, threshold_idx] = errors[5]
            mean_end_time_biases[trace, threshold_idx] = errors[6]
            mean_coverages[trace, threshold_idx] = errors[7]


        print('Direct computation finished for trace number {}' .format(trace))

        for threshold in np.arange(args.start_threshold, args.end_threshold, args.step_threshold):
            errors_I = mean_average_precision_and_errors(device=arguments['device'],
                                                         pred_segments=true_segments,
                                                         true_segments=pred_segments,
                                                         iou_threshold=threshold,
                                                         seg_format="mix",
                                                         num_classes=1)

            threshold_idx = np.where(np.arange(args.start_threshold, args.end_threshold, args.step_threshold)==threshold)[0][0]
                                                                                   
            mAPs_I[trace, threshold_idx] = errors_I[0]
            mean_duration_errors_I[trace, threshold_idx] = errors_I[1]
            mean_start_time_errors_I[trace, threshold_idx] = errors_I[2]
            mean_end_time_errors_I[trace, threshold_idx] = errors_I[3]
            mean_duration_biases_I[trace, threshold_idx] = errors_I[4]
            mean_start_time_biases_I[trace, threshold_idx] = errors_I[5]
            mean_end_time_biases_I[trace, threshold_idx] = errors_I[6]
            mean_coverages_I[trace, threshold_idx] = errors_I[7]

        print('Indirect computation finished for trace number {}' .format(trace))

            
    return [mAPs, mean_duration_errors, mean_start_time_errors, mean_end_time_errors,\
                  mean_duration_biases, mean_start_time_biases, mean_end_time_biases, mean_coverages,\
            mAPs_I, mean_duration_errors_I, mean_start_time_errors_I, mean_end_time_errors_I,\
                    mean_duration_biases_I, mean_start_time_biases_I, mean_end_time_biases_I, mean_coverages_I]










def compute_predictions(args, arguments):
    # switch to evaluate mode
    arguments['model'].eval()
    
    (traces, windows) = (arguments['TADL'].num_of_traces, arguments['TADL'].windows_per_trace)
    Ths = np.arange(args.start_threshold, args.end_threshold, args.step_threshold).shape[0]


    start_predictions = []
    end_predictions   = []
    for trace in range(traces):
        starts = np.array([])
        ends   = np.array([])
        for window in range(windows):
            # bring a new window
            times, noisy_signals, _, _ = arguments['TADL'].get_signal_window(trace, window)

            times = times.unsqueeze(0)
            noisy_signals = noisy_signals.unsqueeze(0)
            mean = torch.mean(noisy_signals, 1, True)
            noisy_signals = noisy_signals-mean

            with torch.no_grad():
                # forward
                noisy_signals = noisy_signals.unsqueeze(1)
                pred_num_pulses = int(arguments['pulse_counter'](noisy_signals))
                if pred_num_pulses > 0:
                    outputs = arguments['model'](noisy_signals)
                    #outputs = arguments['model'](noisy_signals)
                noisy_signals = noisy_signals.squeeze(1)
            
            if pred_num_pulses > 0:
                # indices to be eliminated from the output (i.e. non-segments)
                idxs = torch.where(outputs['pred_logits'][0, :, :].argmax(-1) != 1)[0]
                segments=outputs['pred_segments'][0,idxs,:].detach()
    
                start_time_marks = (segments[:,0] * arguments['TADL'].window + times[0,0]).cpu().detach().numpy()
                end_time_marks   = ((segments[:,1] + segments[:,0]) * arguments['TADL'].window + times[0,0]).cpu().detach().numpy()
                
                starts = np.append(starts, start_time_marks)
                ends   = np.append(ends, end_time_marks)
            
        start_predictions.append(starts)
        end_predictions.append(ends)
        
    return start_predictions, end_predictions






























def plot_error_stats(traces, mAPs, mean_duration_errors, mean_start_time_errors, mean_end_time_errors, mean_coverages):
    plt.rcParams.update({'font.size': 20})
    fontsize=30


    ave0 = []
    std0 = []
    ave1 = []
    std1 = []
    ave2 = []
    std2 = []
    ave3 = []
    std3 = []
    ave4 = []
    std4 = []
    mean_precision = mAPs.numpy()
    mean_duration = mean_duration_errors.numpy()
    mean_start_time = 1000*mean_start_time_errors.numpy()
    mean_end_time = 1000*mean_end_time_errors.numpy()
    mean_coverage = mean_coverages.numpy()
    for i in range(traces):
        ave0.append(np.nanmean(mean_precision[i,:].ravel()))
        std0.append(np.nanstd(mean_precision[i,:].ravel()))
        ave1.append(np.nanmean(mean_duration[i,:].ravel()))
        std1.append(np.nanstd(mean_duration[i,:].ravel()))
        ave2.append(np.nanmean(mean_start_time[i,:].ravel()))
        std2.append(np.nanstd(mean_start_time[i,:].ravel()))
        ave3.append(np.nanmean(mean_end_time[i,:].ravel()))
        std3.append(np.nanstd(mean_end_time[i,:].ravel()))
        ave4.append(np.nanmean(mean_coverage[i,:].ravel()))
        std4.append(np.nanstd(mean_coverage[i,:].ravel()))

    fig, axs = plt.subplots(5, 1, figsize=(10,25))
    fig.tight_layout(pad=4.0)
    #durations = [i+1 for i in range(Duration)]
    durations = [1,2,3,4,5,6]


    axs[0].errorbar(durations,ave0,std0, linestyle='None', marker='o', linewidth=1.5, markeredgewidth=2.0, capsize=10)
    axs[0].set_title("mAP: {:.2f} \nSTD: {:.2f}" .format(np.nanmean(mean_precision.ravel()),np.nanstd(mean_precision.ravel())), fontsize=fontsize)

    axs[0].set_xticks([1,2,3,4,5,6])
    #axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)


    axs[1].errorbar(durations,ave1,std1, linestyle='None', marker='o', linewidth=1.5, markeredgewidth=2.0, capsize=10)
    axs[1].set_title("Dur. err.: {:.2f} [%]\nSTD: {:.2f}" .format(np.nanmean(mean_duration.ravel()),np.nanstd(mean_duration.ravel())), fontsize=fontsize)

    axs[1].set_xticks([1,2,3,4,5,6])
    #axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)


    axs[2].errorbar(durations,ave2,std2, linestyle='None', marker='o', linewidth=1.5, markeredgewidth=2.0, capsize=10)
    axs[2].set_title("Start time err.: {:.2f} [ms]\nSTD: {:.2f}" .format(np.nanmean(mean_start_time.ravel()),np.nanstd(mean_start_time.ravel())), fontsize=fontsize)

    axs[2].set_xticks([1,2,3,4,5,6])
    #axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[2].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)


    axs[3].errorbar(durations,ave3,std3, linestyle='None', marker='o', linewidth=1.5, markeredgewidth=2.0, capsize=10)
    axs[3].set_title("End time err.: {:.2f} [ms]\nSTD: {:.2f}" .format(np.nanmean(mean_end_time.ravel()),np.nanstd(mean_end_time.ravel())), fontsize=fontsize)

    axs[3].set_xticks([1,2,3,4,5,6])
    #axs[3].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[3].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[3].axhline(y=0, color='k', linestyle='-', linewidth=0.5)


    axs[4].errorbar(durations,ave4,std4, linestyle='None', marker='o', linewidth=1.5, markeredgewidth=2.0, capsize=10)
    axs[4].set_title("Coverage: {:.2f} [%]\nSTD: {:.2f}" .format(np.nanmean(mean_coverage.ravel()),np.nanstd(mean_coverage.ravel())), fontsize=fontsize)

    axs[4].set_xticks([1,2,3,4,5,6])
    #axs[4].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[4].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[4].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    plt.show()
    
    print("mAP: {}\nSTD: {}" .format(np.nanmean(mean_precision.ravel()),np.nanstd(mean_precision.ravel())))
    print("Average duration error: {}\nSTD: {}" .format(np.nanmean(mean_duration.ravel()),np.nanstd(mean_duration.ravel())))
    print("Average start time error: {}\nSTD: {}" .format(np.nanmean(mean_start_time.ravel()),np.nanstd(mean_start_time.ravel())))
    print("Average end time error: {}\nSTD: {}" .format(np.nanmean(mean_end_time.ravel()),np.nanstd(mean_end_time.ravel())))
    print("Average coverage: {}\nSTD: {}" .format(np.nanmean(mean_coverage.ravel()),np.nanstd(mean_coverage.ravel())))

















def plot_bias_stats(traces, mAPs, mean_duration_errors, mean_start_time_errors, mean_end_time_errors, mean_coverages):
    plt.rcParams.update({'font.size': 20})
    fontsize=30


    ave0 = []
    std0 = []
    ave1 = []
    std1 = []
    ave2 = []
    std2 = []
    ave3 = []
    std3 = []
    ave4 = []
    std4 = []
    mean_precision = mAPs.numpy()
    mean_duration = mean_duration_errors.numpy()
    mean_start_time = 1000*mean_start_time_errors.numpy()
    mean_end_time = 1000*mean_end_time_errors.numpy()
    mean_coverage = mean_coverages.numpy()
    for i in range(traces):
        ave0.append(np.nanmean(mean_precision[i,:].ravel()))
        std0.append(np.nanstd(mean_precision[i,:].ravel()))
        ave1.append(np.nanmean(mean_duration[i,:].ravel()))
        std1.append(np.nanstd(mean_duration[i,:].ravel()))
        ave2.append(np.nanmean(mean_start_time[i,:].ravel()))
        std2.append(np.nanstd(mean_start_time[i,:].ravel()))
        ave3.append(np.nanmean(mean_end_time[i,:].ravel()))
        std3.append(np.nanstd(mean_end_time[i,:].ravel()))
        ave4.append(np.nanmean(mean_coverage[i,:].ravel()))
        std4.append(np.nanstd(mean_coverage[i,:].ravel()))

    fig, axs = plt.subplots(5, 1, figsize=(10,25))
    fig.tight_layout(pad=4.0)
    #durations = [i+1 for i in range(Duration)]
    durations = [1,2,3,4,5,6]


    axs[0].errorbar(durations,ave0,std0, linestyle='None', marker='o', linewidth=1.5, markeredgewidth=2.0, capsize=10)
    axs[0].set_title("mAP: {:.2f} \nSTD: {:.2f}" .format(np.nanmean(mean_precision.ravel()),np.nanstd(mean_precision.ravel())), fontsize=fontsize)

    axs[0].set_xticks([1,2,3,4,5,6])
    #axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)


    axs[1].errorbar(durations,ave1,std1, linestyle='None', marker='o', linewidth=1.5, markeredgewidth=2.0, capsize=10)
    axs[1].set_title("Dur. bias.: {:.2f} [%]\nSTD: {:.2f}" .format(np.nanmean(mean_duration.ravel()),np.nanstd(mean_duration.ravel())), fontsize=fontsize)

    axs[1].set_xticks([1,2,3,4,5,6])
    #axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)


    axs[2].errorbar(durations,ave2,std2, linestyle='None', marker='o', linewidth=1.5, markeredgewidth=2.0, capsize=10)
    axs[2].set_title("Start time bias.: {:.2f} [ms]\nSTD: {:.2f}" .format(np.nanmean(mean_start_time.ravel()),np.nanstd(mean_start_time.ravel())), fontsize=fontsize)

    axs[2].set_xticks([1,2,3,4,5,6])
    #axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[2].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)


    axs[3].errorbar(durations,ave3,std3, linestyle='None', marker='o', linewidth=1.5, markeredgewidth=2.0, capsize=10)
    axs[3].set_title("End time bias.: {:.2f} [ms]\nSTD: {:.2f}" .format(np.nanmean(mean_end_time.ravel()),np.nanstd(mean_end_time.ravel())), fontsize=fontsize)

    axs[3].set_xticks([1,2,3,4,5,6])
    #axs[3].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[3].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[3].axhline(y=0, color='k', linestyle='-', linewidth=0.5)


    axs[4].errorbar(durations,ave4,std4, linestyle='None', marker='o', linewidth=1.5, markeredgewidth=2.0, capsize=10)
    axs[4].set_title("Coverage: {:.2f} [%]\nSTD: {:.2f}" .format(np.nanmean(mean_coverage.ravel()),np.nanstd(mean_coverage.ravel())), fontsize=fontsize)

    axs[4].set_xticks([1,2,3,4,5,6])
    #axs[4].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[4].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[4].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    plt.show()
    
    print("mAP: {}\nSTD: {}" .format(np.nanmean(mean_precision.ravel()),np.nanstd(mean_precision.ravel())))
    print("Average duration bias: {}\nSTD: {}" .format(np.nanmean(mean_duration.ravel()),np.nanstd(mean_duration.ravel())))
    print("Average start time bias: {}\nSTD: {}" .format(np.nanmean(mean_start_time.ravel()),np.nanstd(mean_start_time.ravel())))
    print("Average end time bias: {}\nSTD: {}" .format(np.nanmean(mean_end_time.ravel()),np.nanstd(mean_end_time.ravel())))
    print("Average coverage: {}\nSTD: {}" .format(np.nanmean(mean_coverage.ravel()),np.nanstd(mean_coverage.ravel())))































# match the number of target segments to the maximum num_target_segments in the batch
def transform_targets(targets):
  aux = list([])
  for target in targets:
    dic = {}
    up_to = len(np.where((target[2,:].cpu() < 1.0))[0])
    dic['labels'] = target[2,:up_to]
    dic['segments'] = target[:2,:up_to]
    dic['segments'] = dic['segments'].permute(1, 0)
    aux.append(dic)
      
  return aux
















if __name__ == '__main__':
    main()
