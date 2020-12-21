import torch
import torch.distributed as dist





class AverageMeter(object):
    """Computes and stores the average current value"""

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





def reduce_tensor(tensor, world_size):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= world_size
        return rt


# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
        if hasattr(t, 'item'):
                return t.item()
        else:
                return t[0]
