import os
import torch
import torch.nn as nn
import shutil
import math
import torch.optim as optim
import matplotlib.pyplot as plt

def learning_rate_schedule(args, arguments):
    """Build learning rate schedule."""
    num_examples = arguments['TADL'].shard_size

    optimizer_params = arguments['optimizer'].state[arguments['optimizer'].param_groups[0]["params"][-1]]
    if 'step' in optimizer_params:
        global_step = optimizer_params['step']
    else:
        global_step = 1

    # warmup_steps = warmup_epochs * num_examples / batch_size
    warmup_steps = int(round(args.warmup_epochs * num_examples // args.batch_size))

    global_batch_size = args.world_size * args.batch_size
    if args.lrs == 'linear':
        # scaled_lr = learning_rate * global_batch_size / 256
        scaled_lr = args.lr * global_batch_size / 256.
    elif args.lrs == 'sqrt':
        # scaled_lr = learning_rate * sqrt(global_batch_size)
        scaled_lr = args.lr * math.sqrt(global_batch_size)
    else:
        raise ValueError('Unknown learning rate scaling {}'.format(args.lrs))

    learning_rate = (float(global_step) / int(warmup_steps) * scaled_lr if warmup_steps else scaled_lr)

    # Cosine decay learning rate schedule
    total_steps = _get_train_steps(num_examples, args.epochs, args.batch_size)
    learning_rate = (learning_rate if global_step < warmup_steps else _cosine_decay(scaled_lr,
                                                                                    global_step - warmup_steps,
                                                                                    total_steps - warmup_steps))

    for param_group in arguments['optimizer'].param_groups:
        param_group['lr'] = learning_rate










def _cosine_decay(learning_rate, global_step, decay_steps, alpha=0.0):
    global_step = min(global_step, decay_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * global_step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return learning_rate * decayed



def _get_train_steps(num_examples, train_epochs, train_batch_size):
    """Determine the number of training steps."""
    return num_examples * train_epochs // train_batch_size + 1







def get_optimizer(model, args):
    """Returns an optimizer."""
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              args.lr,
                              momentum=args.momentum)

    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               args.lr)

    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(),
                                args.lr)

    else:
        raise ValueError('Unknown optimizer {}'.format(args.optimizer))

    return optimizer






def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar'):
    directory = os.path.join(state['arch'])
    if not os.path.exists(directory):
        os.mkdir(directory)

    torch.save(state, os.path.join(directory, filename))
    if is_best:
        if 'best_error' in state.keys():
            print('Saving a new best model with error {}'.format(state['best_error']))
            shutil.copyfile(os.path.join(directory, filename), os.path.join(directory, best_filename))
        else:
            print('Saving a new best model with precision {}'.format(state['best_precision']))
            shutil.copyfile(os.path.join(directory, filename), os.path.join(directory, best_filename))




def top_k_accuracy(preds, target, k):
    a=torch.transpose(torch.topk(preds,k=k,dim=1)[1],0,1)
    if len(target.shape) == 1:
        b=target
    else:
        b=torch.argmax(target,dim=1)

    c=a==b
    d=torch.any(c,dim=0)
    return torch.sum(d)/(d.shape[0] + 0.0)








class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x





def compute_relative_error(predicted, ground_truth):
    return ((predicted - ground_truth) / ground_truth)*100




def plot_detector_stats(losses, precisions):
    fig, (loss, precision) = plt.subplots(2, 1, sharex=True, figsize=(10,10))
    fig.suptitle('Training process history', fontweight="bold", size=20)

    loss.plot(losses)
    loss.set(ylabel='Loss')

    precision.plot(precisions, 'tab:green')
    precision.set(ylabel='Precision')

    plt.show()





def plot_features_stats(losses, duration_errors, amplitude_errors):
    plt.rcParams.update({'font.size': 20})
    fig, (loss, duration, amplitude) = plt.subplots(3, 1, sharex=True, figsize=(10,10))
    fig.suptitle('Training process history', fontweight="bold", size=20)

    loss.plot(losses)
    loss.set(ylabel='Loss')

    duration.plot(duration_errors, 'tab:green')
    duration.set(ylabel='Duration error')

    amplitude.plot(amplitude_errors, 'tab:orange')
    amplitude.set(ylabel='Amplitude error', xlabel='Epochs')

    plt.show()



def plot_counter_stats(losses, counter_errors):
    plt.rcParams.update({'font.size': 20})
    fig, (loss, counter) = plt.subplots(2, 1, sharex=True, figsize=(10,10))
    fig.suptitle('Training process history', fontweight="bold", size=20)

    loss.plot(losses)
    loss.set(ylabel='Loss')

    counter.plot(counter_errors, 'tab:green')
    counter.set(ylabel='Counter error')
    counter.set(xlabel='Epochs')

    plt.show()





def save_stats(stats, path, filename='stats.pth.tar'):
    directory = os.path.join(path)
    if not os.path.exists(directory):
        os.mkdir(directory)

    directory = os.path.join(path, stats['Arch'])
    if not os.path.exists(directory):
        os.mkdir(directory)

    print('Saving backbone model stats for {}'.format(stats['Arch']))
    torch.save(stats, os.path.join(directory, filename))

