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
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter

sys.path.append('../ResNet')
import ResNet1d as rn
sys.path.append('../')
import Model_Util
import Utilities
from Dataset_Management import Artificial_DataLoader

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
                        help='path to validation dataset')
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
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-stats', '--statistics', dest='statistics', action='store_true',
                        help='Compute statistics about errors of a trained model on validation set')
    parser.add_argument('-stats-from-file', default='', type=str, metavar='STATS_FROM_FILE',
                        help='path to load the stats produced during validation from a file (default: none)')
    parser.add_argument('-r', '--run', dest='run', action='store_true',
                        help='Run a trained model and plots a batch of predictions in noisy signals')
    parser.add_argument('--run-plot-window', default=1.0, type=float, metavar='RPW',
                        help='the percentage of the window width the you want to actually plot (default: 1; which means 100%%)')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--cpu', action='store_true',
                        help='Runs CPU based version of the workflow.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='provides additional details as to what the program is doing')
    parser.add_argument('-t', '--test', action='store_true',
                        help='Launch test mode with preset arguments')
    parser.add_argument('-pth', '--plot-training-history', action='store_true',
                        help='Only plots the training history of a trained model: Loss and validation errors')

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
                Cnp = stats['Cnp']
                Duration = stats['Duration']
                Dnp = stats['Dnp']

                print("=> loaded stats '{}'" .format(args.stats_from_file))
                return mAPs, mean_duration_errors, mean_start_time_errors, mean_end_time_errors,\
                             mean_duration_biases, mean_start_time_biases, mean_end_time_biases, mean_coverages,\
                             Cnp, Duration, Dnp 
            else:
                print("=> no stats found at '{}'" .format(args.stats_from_file))

        mAPs, mean_duration_errors, mean_start_time_errors, mean_end_time_errors,\
              mean_duration_biases, mean_start_time_biases, mean_end_time_biases, mean_coverages, Cnp, Duration, Dnp = bring_stats_from_file()
        plot_error_stats(Cnp, Duration, Dnp, mAPs, mean_duration_errors, mean_start_time_errors, mean_end_time_errors, mean_coverages)
        plot_bias_stats(Cnp, Duration, Dnp, mAPs, mean_duration_biases, mean_start_time_biases, mean_end_time_biases, mean_coverages)

        return










    # Data loading code
    testdir = os.path.join(args.data, 'test_i')
    #testdir = os.path.join(args.data, 'test')

    if args.test:
        test_f = h5py.File(testdir + '/test_toy.h5', 'r')
    else:
        test_f = h5py.File(testdir + '/test.h5', 'r')


    # this is the dataset for testing
    sampling_rate = 100000                   # This is the number of samples per second of the signals in the dataset
    #sampling_rate = 10000                   # This is the number of samples per second of the signals in the dataset
    if args.test:
        number_of_concentrations = 2        # This is the number of different concentrations in the dataset
        number_of_durations = 2             # This is the number of different translocation durations per concentration in the dataset
        number_of_diameters = 4             # This is the number of different translocation durations per concentration in the dataset
        window = 0.5                        # This is the time window in seconds
        length = 10                         # This is the time of a complete signal for certain concentration and duration
    else:
        number_of_concentrations = 20       # This is the number of different concentrations in the dataset
        number_of_durations = 1             # This is the number of different translocation durations per concentration in the dataset
        #number_of_durations = 5             # This is the number of different translocation durations per concentration in the dataset
        number_of_diameters = 15            # This is the number of different translocation durations per concentration in the dataset
        window = 0.049999                        # This is the time window in seconds
        #window = 0.5                        # This is the time window in seconds
        length = 4.9999                          # This is the time of a complete signal for certain concentration and duration
        #length = 5                          # This is the time of a complete signal for certain concentration and duration
        #length = 10                         # This is the time of a complete signal for certain concentration and duration

    # Testing Artificial Data Loader
    TADL = Artificial_DataLoader(args.world_size, args.local_rank, device, test_f, sampling_rate,
                                 number_of_concentrations, number_of_durations, number_of_diameters,
                                 window, length, args.batch_size)

    if args.verbose:
        print('From rank {} test shard size is {}'. format(args.local_rank, TADL.get_number_of_avail_windows()))












    if args.run:
        arguments = {'model': detr,
                     'device': device,
                     'epoch': 0,
                     'TADL': TADL}

        if args.local_rank == 0:
            run_model(args, arguments)

        return

    if args.statistics:
        arguments = {'model': detr,
                     'device': device,
                     'epoch': 0,
                     'TADL': TADL}

        [mAPs, mean_duration_errors, mean_start_time_errors, mean_end_time_errors,\
               mean_duration_biases, mean_start_time_biases, mean_end_time_biases, mean_coverages] = compute_error_stats(args, arguments)
        if args.save_stats:
            Model_Util.save_stats({'mAPs': mAPs,
                                   'mean_duration_errors': mean_duration_errors,
                                   'mean_start_time_errors': mean_start_time_errors,
                                   'mean_end_time_errors': mean_end_time_errors,
                                   'mean_duration_biases': mean_duration_biases,
                                   'mean_start_time_biases': mean_start_time_biases,
                                   'mean_end_time_biases': mean_end_time_biases,
                                   'mean_coverages': mean_coverages,
                                   'Cnp': TADL.shape[0],
                                   'Duration': TADL.shape[1],
                                   'Dnp': TADL.shape[2],
                                   'Arch': 'DETR_' + args.feature_predictor_arch},
                                   args.save_stats)

        return


    #if args.evaluate:
        #arguments = {'model': model,
                     #'device': device,
                     #'epoch': 0,
                     #'VADL': VADL}

        #[duration_error, amplitude_error] = validate(args, arguments)
        #print('##Duration error {0}\n'
              #'##Amplitude error {1}'.format(
              #duration_error,
              #amplitude_error))

        #return

    if args.plot_training_history and args.local_rank == 0:
        Model_Util.plot_detector_stats(loss_history, precision_history)
        hours = int(total_time.sum / 3600)
        minutes = int((total_time.sum % 3600) / 60)
        seconds = int((total_time.sum % 3600) % 60)
        print('The total training time was {} hours {} minutes and {} seconds' .format(hours, minutes, seconds))
        hours = int(total_time.avg / 3600)
        minutes = int((total_time.avg % 3600) / 60)
        seconds = int((total_time.avg % 3600) % 60)
        print('while the average time during one epoch of training was {} hours {} minutes and {} seconds' .format(hours, minutes, seconds))
        return









































def validate(args, arguments):
    average_precision = Utilities.AverageMeter()

    # switch to evaluate mode
    arguments['detr'].eval()

    end = time.time()

    val_loader_len = int(math.ceil(arguments['VADL'].shard_size / args.batch_size))
    i = 0
    arguments['VADL'].reset_avail_winds(arguments['epoch'])
    pred_segments = []
    true_segments = []
    while i * arguments['VADL'].batch_size < arguments['VADL'].shard_size:
        # get the noisy inputs and the labels
        _, inputs, _, targets, labels = arguments['TADL'].get_batch()

        mean = torch.mean(inputs, 1, True)
        inputs = inputs-mean
            
        with torch.no_grad():
            # forward
            inputs = inputs.unsqueeze(1)
            outputs = arguments['detr'](inputs)

        for j in range(arguments['VADL'].batch_size):
            train_idx = int(j + i * arguments['VADL'].batch_size)

            probabilities = F.softmax(outputs['pred_logits'][j], dim=1)
            aux_pred_segments = outputs['pred_segments'][j]

            for probability, pred_segment in zip(probabilities.to('cpu'), aux_pred_segments.to('cpu')):
                #if probability[-1] < 0.9:
                if torch.argmax(probability) != args.num_classes:
                    segment = [train_idx, np.argmax(probability[:-1]).item(), 1.0 - probability[-1].item(), pred_segment[0].item(), pred_segment[1].item()]
                    pred_segments.append(segment)


            num_pulses = labels[j, 0]

            starts = targets[j, 0]
            widths = targets[j, 1]
            categories = targets[j, 3]
            
            for k in range(int(num_pulses.item())):
                segment = [train_idx, categories[k].item(), 1.0, starts[k].item(), widths[k].item()]
                true_segments.append(segment)


        i += 1


    for threshold in np.arange(args.start_threshold, args.end_threshold, args.step_threshold):
        detection_precision=mean_average_precision(device=arguments['device'],
                                                   pred_segments=pred_segments,
                                                   true_segments=true_segments,
                                                   iou_threshold=threshold,
                                                   seg_format="mix",
                                                   num_classes=1)

        if args.distributed:
            reduced_detection_precision = Utilities.reduce_tensor(detection_precision.data, args.world_size)
        else:
            reduced_detection_precision = detection_precision.data


        average_precision.update(Utilities.to_python_float(reduced_detection_precision))

    if not args.evaluate:
        arguments['precision_history'].append(average_precision.avg)

    return average_precision.avg




















def compute_error_stats(args, arguments):
    # switch to evaluate mode
    arguments['model'].eval()
    
    (Cnps, Durations, Dnps, windows) = arguments['TADL'].shape
    Ths = np.arange(args.start_threshold, args.end_threshold, args.step_threshold).shape[0]
    mAPs = torch.zeros((Cnps, Durations, Dnps, Ths))
    mean_duration_errors = torch.zeros((Cnps, Durations, Dnps, Ths))
    mean_start_time_errors = torch.zeros((Cnps, Durations, Dnps, Ths))
    mean_end_time_errors = torch.zeros((Cnps, Durations, Dnps, Ths))
    mean_duration_biases = torch.zeros((Cnps, Durations, Dnps, Ths))
    mean_start_time_biases = torch.zeros((Cnps, Durations, Dnps, Ths))
    mean_end_time_biases = torch.zeros((Cnps, Durations, Dnps, Ths))
    mean_coverages = torch.zeros((Cnps, Durations, Dnps, Ths))
    
    arguments['TADL'].reset_avail_winds(arguments['epoch'])
    for Cnp in range(Cnps):
        for Duration in range(Durations):
            for Dnp in range(Dnps):
                pred_segments = []
                true_segments = []
                for window in range(windows):
                    # bring a new window
                    times, noisy_signals, clean_signals, targets, labels = arguments['TADL'].get_signal_window(Cnp, Duration, Dnp, window)
        
                    if labels[0] > 0:
                        times = times.unsqueeze(0)
                        noisy_signals = noisy_signals.unsqueeze(0)
                        clean_signals = clean_signals.unsqueeze(0)
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
                        categories = targets[0, 3]
                        
                        for k in range(int(num_pulses.item())):
                            segment = [train_idx, categories[k].item(), 1.0, starts[k].item(), widths[k].item()]
                            true_segments.append(segment)


                for threshold in np.arange(args.start_threshold, args.end_threshold, args.step_threshold):
                    errors = mean_average_precision_and_errors(device=arguments['device'],
                                                               pred_segments=pred_segments,                                                             
                                                               true_segments=true_segments,
                                                               iou_threshold=threshold,
                                                               seg_format="mix",
                                                               num_classes=1)
                                                                                           
                    threshold_idx = np.where(np.arange(args.start_threshold, args.end_threshold, args.step_threshold)==threshold)[0][0]
                    mAPs[Cnp, Duration, Dnp, threshold_idx] = errors[0]
                    mean_duration_errors[Cnp, Duration, Dnp, threshold_idx] = errors[1]
                    mean_start_time_errors[Cnp, Duration, Dnp, threshold_idx] = errors[2]
                    mean_end_time_errors[Cnp, Duration, Dnp, threshold_idx] = errors[3]
                    mean_duration_biases[Cnp, Duration, Dnp, threshold_idx] = errors[4]
                    mean_start_time_biases[Cnp, Duration, Dnp, threshold_idx] = errors[5]
                    mean_end_time_biases[Cnp, Duration, Dnp, threshold_idx] = errors[6]
                    mean_coverages[Cnp, Duration, Dnp, threshold_idx] = errors[7]

                
    return [mAPs, mean_duration_errors, mean_start_time_errors, mean_end_time_errors,\
                  mean_duration_biases, mean_start_time_biases, mean_end_time_biases, mean_coverages]








def plot_error_stats(Cnp, Duration, Dnp, mAPs, mean_duration_errors, mean_start_time_errors, mean_end_time_errors, mean_coverages):
    plt.rcParams.update({'font.size': 25})
    fontsize=22
    
    mean_precision = mAPs.numpy()
    mean_precision = np.nanmean(mean_precision, 3)

    std_precision = mAPs.numpy()
    std_precision = np.nanstd(std_precision, 3)


    mean_duration = mean_duration_errors.numpy()
    mean_duration = np.nanmean(mean_duration, 3)

    std_duration = mean_duration_errors.numpy()
    std_duration = np.nanstd(std_duration, 3)


    mean_start_time = 1000*mean_start_time_errors.numpy()
    mean_start_time = np.nanmean(mean_start_time, 3)

    std_start_time = 1000*mean_start_time_errors.numpy()
    std_start_time = np.nanstd(std_start_time, 3)


    mean_end_time = 1000*mean_end_time_errors.numpy()
    mean_end_time = np.nanmean(mean_end_time, 3)

    std_end_time = 1000*mean_end_time_errors.numpy()
    std_end_time = np.nanstd(std_end_time, 3)


    mean_coverage = mean_coverages.numpy()
    mean_coverage = np.nanmean(mean_coverage, 3)

    std_coverage = mean_coverages.numpy()
    std_coverage = np.nanstd(std_coverage, 3)


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
        top = mean_precision[:,i,:]
        top = top.transpose()
        ave0[i].plot_surface(x, y, top, alpha=0.9)

        std_surface = std_precision[:,i,:]
        top1 = top+std_surface.transpose()
        top2 = top-std_surface.transpose()
        ave0[i].plot_surface(x, y, top1, alpha=0.2, color='r')
        ave0[i].plot_surface(x, y, top2, alpha=0.2, color='r')

        ave0[i].set_title('mAP (Dur. {})' .format(i+1), fontsize=fontsize)

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

        ave1[i].set_title('Mean Dur. Error (Dur. {})' .format(i+1), fontsize=fontsize)

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
        top = mean_start_time[:,i,:]
        top = top.transpose()
        ave2[i].plot_surface(x, y, top, alpha=0.9)

        std_surface = std_start_time[:,i,:]
        top1 = top+std_surface.transpose()
        top2 = top-std_surface.transpose()
        ave2[i].plot_surface(x, y, top1, alpha=0.2, color='r')
        ave2[i].plot_surface(x, y, top2, alpha=0.2, color='r')

        ave2[i].set_title('Mean Start Time Err. [ms] (Dur. {})' .format(i+1), fontsize=fontsize-2)

        ave2[i].set_xlabel('Cnp', fontsize=fontsize)
        ave2[i].set_ylabel('Dnp', fontsize=fontsize)
        ave2[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        ave2[i].yaxis.set_major_locator(MaxNLocator(integer=True))
        ave2[i].set_yticklabels([])
        ave2[i].set_xticklabels([])


    plt.show()


    ave3 = []
    # setup the figure and axes for count errors
    fig = plt.figure(figsize=(2*Duration*3.2, 7))
    for i in range(Duration):
        ave3.append(fig.add_subplot(1,Duration,i+1, projection='3d'))

    # prepare the data
    _x = np.arange(Cnp)+1
    _y = np.arange(Dnp)+1
    x, y = np.meshgrid(_x, _y)
    width = depth = 1
    for i in range(Duration):
        top = mean_end_time[:,i,:]
        top = top.transpose()
        ave3[i].plot_surface(x, y, top, alpha=0.9)

        std_surface = std_end_time[:,i,:]
        top1 = top+std_surface.transpose()
        top2 = top-std_surface.transpose()
        ave3[i].plot_surface(x, y, top1, alpha=0.2, color='r')
        ave3[i].plot_surface(x, y, top2, alpha=0.2, color='r')

        ave3[i].set_title('Mean End Time Err. [ms] (Dur. {})' .format(i+1), fontsize=fontsize-2)

        ave3[i].set_xlabel('Cnp', fontsize=fontsize)
        ave3[i].set_ylabel('Dnp', fontsize=fontsize)
        ave3[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        ave3[i].yaxis.set_major_locator(MaxNLocator(integer=True))
        ave3[i].set_yticklabels([])
        ave3[i].set_xticklabels([])


    plt.show()


    ave4 = []
    # setup the figure and axes for count errors
    fig = plt.figure(figsize=(2*Duration*3.2, 7))
    for i in range(Duration):
        ave4.append(fig.add_subplot(1,Duration,i+1, projection='3d'))

    # prepare the data
    _x = np.arange(Cnp)+1
    _y = np.arange(Dnp)+1
    x, y = np.meshgrid(_x, _y)
    width = depth = 1
    for i in range(Duration):
        top = mean_coverage[:,i,:]
        top = top.transpose()
        ave4[i].plot_surface(x, y, top, alpha=0.9)

        std_surface = std_coverage[:,i,:]
        top1 = top+std_surface.transpose()
        top2 = top-std_surface.transpose()
        ave4[i].plot_surface(x, y, top1, alpha=0.2, color='r')
        ave4[i].plot_surface(x, y, top2, alpha=0.2, color='r')

        ave4[i].set_title('Mean Coverage (Dur. {})' .format(i+1), fontsize=fontsize)

        ave4[i].set_xlabel('Cnp', fontsize=fontsize)
        ave4[i].set_ylabel('Dnp', fontsize=fontsize)
        ave4[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        ave4[i].yaxis.set_major_locator(MaxNLocator(integer=True))
        ave4[i].set_yticklabels([])
        ave4[i].set_xticklabels([])


    plt.show()





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
    for i in range(Duration):
        ave0.append(np.nanmean(mean_precision[:,i,:,:].ravel()))
        std0.append(np.nanstd(mean_precision[:,i,:,:].ravel()))
        ave1.append(np.nanmean(mean_duration[:,i,:,:].ravel()))
        std1.append(np.nanstd(mean_duration[:,i,:,:].ravel()))
        ave2.append(np.nanmean(mean_start_time[:,i,:,:].ravel()))
        std2.append(np.nanstd(mean_start_time[:,i,:,:].ravel()))
        ave3.append(np.nanmean(mean_end_time[:,i,:,:].ravel()))
        std3.append(np.nanstd(mean_end_time[:,i,:,:].ravel()))
        ave4.append(np.nanmean(mean_coverage[:,i,:,:].ravel()))
        std4.append(np.nanstd(mean_coverage[:,i,:,:].ravel()))

    fig, axs = plt.subplots(5, 1, figsize=(10,25))
    fig.tight_layout(pad=4.0)
    #durations = [i+1 for i in range(Duration)]
    durations = [0.5,1.0,1.5,3.0,5.0]


    axs[0].errorbar(durations,ave0,std0, linestyle='None', marker='o', linewidth=1.5, markeredgewidth=2.0, capsize=10)
    axs[0].set_title("mAP: {:.2f} \nSTD: {:.2f}" .format(np.nanmean(mean_precision.ravel()),np.nanstd(mean_precision.ravel())), fontsize=fontsize)

    axs[0].set_xticks([0.5,1.0,1.5,3.0,5.0])
    #axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)


    axs[1].errorbar(durations,ave1,std1, linestyle='None', marker='o', linewidth=1.5, markeredgewidth=2.0, capsize=10)
    axs[1].set_title("Dur. err.: {:.2f} [%]\nSTD: {:.2f}" .format(np.nanmean(mean_duration.ravel()),np.nanstd(mean_duration.ravel())), fontsize=fontsize)

    axs[1].set_xticks([0.5,1.0,1.5,3.0,5.0])
    #axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)


    axs[2].errorbar(durations,ave2,std2, linestyle='None', marker='o', linewidth=1.5, markeredgewidth=2.0, capsize=10)
    axs[2].set_title("Start time err.: {:.2f} [ms]\nSTD: {:.2f}" .format(np.nanmean(mean_start_time.ravel()),np.nanstd(mean_start_time.ravel())), fontsize=fontsize)

    axs[2].set_xticks([0.5,1.0,1.5,3.0,5.0])
    #axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[2].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)


    axs[3].errorbar(durations,ave3,std3, linestyle='None', marker='o', linewidth=1.5, markeredgewidth=2.0, capsize=10)
    axs[3].set_title("End time err.: {:.2f} [ms]\nSTD: {:.2f}" .format(np.nanmean(mean_end_time.ravel()),np.nanstd(mean_end_time.ravel())), fontsize=fontsize)

    axs[3].set_xticks([0.5,1.0,1.5,3.0,5.0])
    #axs[3].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[3].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[3].axhline(y=0, color='k', linestyle='-', linewidth=0.5)


    axs[4].errorbar(durations,ave4,std4, linestyle='None', marker='o', linewidth=1.5, markeredgewidth=2.0, capsize=10)
    axs[4].set_title("Coverage: {:.2f} [%]\nSTD: {:.2f}" .format(np.nanmean(mean_coverage.ravel()),np.nanstd(mean_coverage.ravel())), fontsize=fontsize)

    axs[4].set_xticks([0.5,1.0,1.5,3.0,5.0])
    #axs[4].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[4].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[4].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    plt.show()
    
    print("mAP: {}\nSTD: {}" .format(np.nanmean(mean_precision.ravel()),np.nanstd(mean_precision.ravel())))
    print("Average duration error: {}\nSTD: {}" .format(np.nanmean(mean_duration.ravel()),np.nanstd(mean_duration.ravel())))
    print("Average start time error: {}\nSTD: {}" .format(np.nanmean(mean_start_time.ravel()),np.nanstd(mean_start_time.ravel())))
    print("Average end time error: {}\nSTD: {}" .format(np.nanmean(mean_end_time.ravel()),np.nanstd(mean_end_time.ravel())))
    print("Average coverage: {}\nSTD: {}" .format(np.nanmean(mean_coverage.ravel()),np.nanstd(mean_coverage.ravel())))



















def plot_bias_stats(Cnp, Duration, Dnp, mAPs, mean_duration_errors, mean_start_time_errors, mean_end_time_errors, mean_coverages):
    plt.rcParams.update({'font.size': 25})
    fontsize=22
    
    mean_precision = mAPs.numpy()
    mean_precision = np.nanmean(mean_precision, 3)

    std_precision = mAPs.numpy()
    std_precision = np.nanstd(std_precision, 3)


    mean_duration = mean_duration_errors.numpy()
    mean_duration = np.nanmean(mean_duration, 3)

    std_duration = mean_duration_errors.numpy()
    std_duration = np.nanstd(std_duration, 3)


    mean_start_time = 1000*mean_start_time_errors.numpy()
    mean_start_time = np.nanmean(mean_start_time, 3)

    std_start_time = 1000*mean_start_time_errors.numpy()
    std_start_time = np.nanstd(std_start_time, 3)


    mean_end_time = 1000*mean_end_time_errors.numpy()
    mean_end_time = np.nanmean(mean_end_time, 3)

    std_end_time = 1000*mean_end_time_errors.numpy()
    std_end_time = np.nanstd(std_end_time, 3)


    mean_coverage = mean_coverages.numpy()
    mean_coverage = np.nanmean(mean_coverage, 3)

    std_coverage = mean_coverages.numpy()
    std_coverage = np.nanstd(std_coverage, 3)


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
        top = mean_precision[:,i,:]
        top = top.transpose()
        ave0[i].plot_surface(x, y, top, alpha=0.9)

        std_surface = std_precision[:,i,:]
        top1 = top+std_surface.transpose()
        top2 = top-std_surface.transpose()
        ave0[i].plot_surface(x, y, top1, alpha=0.2, color='r')
        ave0[i].plot_surface(x, y, top2, alpha=0.2, color='r')

        ave0[i].set_title('mAP (Dur. {})' .format(i+1), fontsize=fontsize)

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

        ave1[i].set_title('Mean Dur. Bias (Dur. {})' .format(i+1), fontsize=fontsize)

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
        top = mean_start_time[:,i,:]
        top = top.transpose()
        ave2[i].plot_surface(x, y, top, alpha=0.9)

        std_surface = std_start_time[:,i,:]
        top1 = top+std_surface.transpose()
        top2 = top-std_surface.transpose()
        ave2[i].plot_surface(x, y, top1, alpha=0.2, color='r')
        ave2[i].plot_surface(x, y, top2, alpha=0.2, color='r')

        ave2[i].set_title('Mean Start Time Bias [ms] (Dur. {})' .format(i+1), fontsize=fontsize-2)

        ave2[i].set_xlabel('Cnp', fontsize=fontsize)
        ave2[i].set_ylabel('Dnp', fontsize=fontsize)
        ave2[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        ave2[i].yaxis.set_major_locator(MaxNLocator(integer=True))
        ave2[i].set_yticklabels([])
        ave2[i].set_xticklabels([])


    plt.show()


    ave3 = []
    # setup the figure and axes for count errors
    fig = plt.figure(figsize=(2*Duration*3.2, 7))
    for i in range(Duration):
        ave3.append(fig.add_subplot(1,Duration,i+1, projection='3d'))

    # prepare the data
    _x = np.arange(Cnp)+1
    _y = np.arange(Dnp)+1
    x, y = np.meshgrid(_x, _y)
    width = depth = 1
    for i in range(Duration):
        top = mean_end_time[:,i,:]
        top = top.transpose()
        ave3[i].plot_surface(x, y, top, alpha=0.9)

        std_surface = std_end_time[:,i,:]
        top1 = top+std_surface.transpose()
        top2 = top-std_surface.transpose()
        ave3[i].plot_surface(x, y, top1, alpha=0.2, color='r')
        ave3[i].plot_surface(x, y, top2, alpha=0.2, color='r')

        ave3[i].set_title('Mean End Time Bias [ms] (Dur. {})' .format(i+1), fontsize=fontsize-2)

        ave3[i].set_xlabel('Cnp', fontsize=fontsize)
        ave3[i].set_ylabel('Dnp', fontsize=fontsize)
        ave3[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        ave3[i].yaxis.set_major_locator(MaxNLocator(integer=True))
        ave3[i].set_yticklabels([])
        ave3[i].set_xticklabels([])


    plt.show()


    ave4 = []
    # setup the figure and axes for count errors
    fig = plt.figure(figsize=(2*Duration*3.2, 7))
    for i in range(Duration):
        ave4.append(fig.add_subplot(1,Duration,i+1, projection='3d'))

    # prepare the data
    _x = np.arange(Cnp)+1
    _y = np.arange(Dnp)+1
    x, y = np.meshgrid(_x, _y)
    width = depth = 1
    for i in range(Duration):
        top = mean_coverage[:,i,:]
        top = top.transpose()
        ave4[i].plot_surface(x, y, top, alpha=0.9)

        std_surface = std_coverage[:,i,:]
        top1 = top+std_surface.transpose()
        top2 = top-std_surface.transpose()
        ave4[i].plot_surface(x, y, top1, alpha=0.2, color='r')
        ave4[i].plot_surface(x, y, top2, alpha=0.2, color='r')

        ave4[i].set_title('Mean Coverage (Dur. {})' .format(i+1), fontsize=fontsize)

        ave4[i].set_xlabel('Cnp', fontsize=fontsize)
        ave4[i].set_ylabel('Dnp', fontsize=fontsize)
        ave4[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        ave4[i].yaxis.set_major_locator(MaxNLocator(integer=True))
        ave4[i].set_yticklabels([])
        ave4[i].set_xticklabels([])


    plt.show()





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
    for i in range(Duration):
        ave0.append(np.nanmean(mean_precision[:,i,:,:].ravel()))
        std0.append(np.nanstd(mean_precision[:,i,:,:].ravel()))
        ave1.append(np.nanmean(mean_duration[:,i,:,:].ravel()))
        std1.append(np.nanstd(mean_duration[:,i,:,:].ravel()))
        ave2.append(np.nanmean(mean_start_time[:,i,:,:].ravel()))
        std2.append(np.nanstd(mean_start_time[:,i,:,:].ravel()))
        ave3.append(np.nanmean(mean_end_time[:,i,:,:].ravel()))
        std3.append(np.nanstd(mean_end_time[:,i,:,:].ravel()))
        ave4.append(np.nanmean(mean_coverage[:,i,:,:].ravel()))
        std4.append(np.nanstd(mean_coverage[:,i,:,:].ravel()))

    fig, axs = plt.subplots(5, 1, figsize=(10,25))
    fig.tight_layout(pad=4.0)
    #durations = [i+1 for i in range(Duration)]
    durations = [0.5,1.0,1.5,3.0,5.0]


    axs[0].errorbar(durations,ave0,std0, linestyle='None', marker='o', linewidth=1.5, markeredgewidth=2.0, capsize=10)
    axs[0].set_title("mAP: {:.2f} \nSTD: {:.2f}" .format(np.nanmean(mean_precision.ravel()),np.nanstd(mean_precision.ravel())), fontsize=fontsize)

    axs[0].set_xticks([0.5,1.0,1.5,3.0,5.0])
    #axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)


    axs[1].errorbar(durations,ave1,std1, linestyle='None', marker='o', linewidth=1.5, markeredgewidth=2.0, capsize=10)
    axs[1].set_title("Dur. bias.: {:.2f} [%]\nSTD: {:.2f}" .format(np.nanmean(mean_duration.ravel()),np.nanstd(mean_duration.ravel())), fontsize=fontsize)

    axs[1].set_xticks([0.5,1.0,1.5,3.0,5.0])
    #axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)


    axs[2].errorbar(durations,ave2,std2, linestyle='None', marker='o', linewidth=1.5, markeredgewidth=2.0, capsize=10)
    axs[2].set_title("Start time bias.: {:.2f} [ms]\nSTD: {:.2f}" .format(np.nanmean(mean_start_time.ravel()),np.nanstd(mean_start_time.ravel())), fontsize=fontsize)

    axs[2].set_xticks([0.5,1.0,1.5,3.0,5.0])
    #axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[2].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)


    axs[3].errorbar(durations,ave3,std3, linestyle='None', marker='o', linewidth=1.5, markeredgewidth=2.0, capsize=10)
    axs[3].set_title("End time bias.: {:.2f} [ms]\nSTD: {:.2f}" .format(np.nanmean(mean_end_time.ravel()),np.nanstd(mean_end_time.ravel())), fontsize=fontsize)

    axs[3].set_xticks([0.5,1.0,1.5,3.0,5.0])
    #axs[3].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[3].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[3].axhline(y=0, color='k', linestyle='-', linewidth=0.5)


    axs[4].errorbar(durations,ave4,std4, linestyle='None', marker='o', linewidth=1.5, markeredgewidth=2.0, capsize=10)
    axs[4].set_title("Coverage: {:.2f} [%]\nSTD: {:.2f}" .format(np.nanmean(mean_coverage.ravel()),np.nanstd(mean_coverage.ravel())), fontsize=fontsize)

    axs[4].set_xticks([0.5,1.0,1.5,3.0,5.0])
    #axs[4].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[4].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[4].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    plt.show()
    
    print("mAP: {}\nSTD: {}" .format(np.nanmean(mean_precision.ravel()),np.nanstd(mean_precision.ravel())))
    print("Average duration bias: {}\nSTD: {}" .format(np.nanmean(mean_duration.ravel()),np.nanstd(mean_duration.ravel())))
    print("Average start time bias: {}\nSTD: {}" .format(np.nanmean(mean_start_time.ravel()),np.nanstd(mean_start_time.ravel())))
    print("Average end time bias: {}\nSTD: {}" .format(np.nanmean(mean_end_time.ravel()),np.nanstd(mean_end_time.ravel())))
    print("Average coverage: {}\nSTD: {}" .format(np.nanmean(mean_coverage.ravel()),np.nanstd(mean_coverage.ravel())))

























def run_model(args, arguments):
    plt.rcParams.update({'font.size': 14})

    # switch to evaluate mode
    arguments['model'].eval()

    arguments['TADL'].reset_avail_winds(arguments['epoch'])

    # bring a new batch
    times, noisy_signals, clean_signals, targets, labels = arguments['TADL'].get_batch()
    
    mean = torch.mean(noisy_signals, 1, True)
    noisy_signals = noisy_signals-mean

    with torch.no_grad():
        noisy_signals = noisy_signals.unsqueeze(1)
        #external = torch.reshape(labels[:,0],[arguments['VADL'].batch_size,1])
        outputs = arguments['model'](noisy_signals)
        noisy_signals = noisy_signals.squeeze(1)


    times = times.cpu()
    noisy_signals = noisy_signals.cpu()
    clean_signals = clean_signals.cpu()
    targets = targets.cpu()
    labels = labels.cpu()

    if arguments['TADL'].batch_size < 21:
        if args.run_plot_window < 1.0:
            width=int(args.run_plot_window*times[0].shape[0])
            start=random.randrange(0,times[0].shape[0])
            end=min(start+width,times[0].shape[0]-1)
        else:
            start=0
            end=times[0].shape[0]-1

        fig, axs = plt.subplots(arguments['TADL'].batch_size, 1, figsize=(10,arguments['TADL'].batch_size*3))
        fig.tight_layout(pad=4.0)
        for i, batch_element in enumerate(range(arguments['TADL'].batch_size)):
            # indices to be eliminated from the output (i.e. non-segments)
            idxs = torch.where(outputs['pred_logits'][batch_element, :, :].argmax(-1) != 1)[0]
            segments=outputs['pred_segments'][batch_element,idxs,:].detach()

            if arguments['TADL'].batch_size > 1:
                axs[i].plot(times[batch_element][start:end],noisy_signals[batch_element][start:end])
                #axs[i].plot(times[batch_element],clean_signals[batch_element])
            else:
                axs.plot(times[batch_element][start:end],noisy_signals[batch_element][start:end])
                #axs.plot(times[batch_element],clean_signals[batch_element])

            x_points = (segments[:,0] * arguments['TADL'].window + times[batch_element,0]).cpu().detach().numpy()
            to_delete = []
            for x_point in x_points:
                if not (times[batch_element][start] <= x_point and x_point <= times[batch_element][end]):
                    to_delete.append(np.where(x_points==x_point)[0][0])
                
            x_points = np.delete(x_points, to_delete)
            y_points = np.repeat(0.5, len(x_points))
      
            if arguments['TADL'].batch_size > 1:
                axs[i].plot(x_points, y_points, 'r*')
            else:
                axs.plot(x_points, y_points, 'r*')

            #x_points = (outputs['pred_segments'][:,batch_element,0] * arguments['TADL'].window + times[batch_element,0]).cpu().detach().numpy()
            #y_points = np.repeat(3.8, len(x_points))
      
            #if arguments['TADL'].batch_size > 1:
            #  axs[i].plot(x_points, y_points, 'ro')
            #else:
            #  axs.plot(x_points, y_points, 'ro')
      
            x_points = ((segments[:,1] + segments[:,0]) * arguments['TADL'].window + times[batch_element,0]).cpu().detach().numpy()
            #x_points = (segments[:,batch_element,1] * arguments['TADL'].window + times1[batch_element,0]).cpu().detach().numpy()
            to_delete = []
            for x_point in x_points:
                if not (times[batch_element][start] <= x_point and x_point <= times[batch_element][end]):
                    to_delete.append(np.where(x_points==x_point)[0][0])
                
            x_points = np.delete(x_points, to_delete)
            y_points = np.repeat(0.5, len(x_points))
      
            if arguments['TADL'].batch_size > 1:
                axs[i].plot(x_points, y_points, 'g*')
            else:
                axs.plot(x_points, y_points, 'g*')


            #x_points = ((outputs['pred_segments'][:,batch_element,1] + outputs['pred_segments'][:,batch_element,0]) * DL.window + times[batch_element,0]).cpu().detach().numpy()
            #x_points = (outputs['pred_segments'][:,batch_element,1] * DL.window + times[batch_element,0]).cpu().detach().numpy()
            #y_points = np.repeat(3.8, len(x_points))

            #if arguments['TADL'].batch_size > 1:
            #  axs[i].plot(x_points, y_points, 'gx')
            #else:
            #  axs.plot(x_points, y_points, 'gx')


            # indices to be eliminated from the targets (i.e. non-segments)
            idxs = torch.where(targets[batch_element, 3] != 1)[0]
            segments=targets[batch_element, :, idxs].detach()

            x_points = (segments[0,:] * arguments['TADL'].window + times[batch_element,0]).cpu().detach().numpy()
            #x_points = (segments[:,batch_element,1] * arguments['TADL'].window + times1[batch_element,0]).cpu().detach().numpy()
            to_delete = []
            for x_point in x_points:
                if not (times[batch_element][start] <= x_point and x_point <= times[batch_element][end]):
                    to_delete.append(np.where(x_points==x_point)[0][0])
                
            x_points = np.delete(x_points, to_delete)
            y_points = np.repeat(0.25, len(x_points))
      
            if arguments['TADL'].batch_size > 1:
                axs[i].plot(x_points, y_points, 'ro')
            else:
                axs.plot(x_points, y_points, 'ro')

            x_points = ((segments[0,:] + segments[1,:]) * arguments['TADL'].window + times[batch_element,0]).cpu().detach().numpy()
            #x_points = (segments[:,batch_element,1] * arguments['TADL'].window + times1[batch_element,0]).cpu().detach().numpy()
            to_delete = []
            for x_point in x_points:
                if not (times[batch_element][start] <= x_point and x_point <= times[batch_element][end]):
                    to_delete.append(np.where(x_points==x_point)[0][0])
                
            x_points = np.delete(x_points, to_delete)
            y_points = np.repeat(0.25, len(x_points))
      
            if arguments['TADL'].batch_size > 1:
                axs[i].plot(x_points, y_points, 'go')
            else:
                axs.plot(x_points, y_points, 'go')





            
            if i < arguments['TADL'].batch_size-1:
                axs[i].set_xticklabels([])

            
            if i == arguments['TADL'].batch_size-1:
                axs[i].set_xlabel("Time [s]", fontsize=22)
                axs[i].set_xticklabels((times[batch_element]-times[batch_element][0]).tolist(), fontsize=18)
                axs[i].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            
            if i == int(arguments['TADL'].batch_size/2):
                axs[i].set_ylabel("Current [nA]", fontsize=22)
                
            axs[i].set_yticklabels(noisy_signals[batch_element].tolist(), fontsize=18)
            axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    else:
        print('This will not show more than 20 plots')

    plt.show()











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
