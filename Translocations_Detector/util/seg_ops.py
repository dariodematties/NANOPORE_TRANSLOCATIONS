"""
Utilities for bounding segment manipulation and GIoU.
"""

import torch

def seg_cxw_to_x0x1(x):
    x_c, w = x.unbind(-1)
    b = [(x_c - 0.5 * w), (x_c + 0.5 * w)]
    return torch.stack(b, dim=-1)
    
def seg_x0x1_to_cxw(x):
    x0, x1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (x1 - x0)]
    return torch.stack(b, dim=-1)

def seg_bxw_to_cxw(x):
    x_b, w = x.unbind(-1)
    b = [(x_b + w / 2), w]
    return torch.stack(b, dim=-1)

def seg_length(segments):
    """
    Computes the lengths of a set of bounding segments, which are specified by its
    (x1, x2) coordinates.

    Arguments:
        segments (Tensor[N, 2]): segments for which the length will be computed. They
            are expected to be in (x1 x2) format

    Returns:
        length (Tensor[N]): length for each segment
    """
    return segments[:, 1] - segments[:, 0]
    

# segments1 contains N segments while
# segments2 contains M segments
def seg_iou(segments1, segments2):
    len1 = seg_length(segments1)
    len2 = seg_length(segments2)
    
    left = torch.max(segments1[:, None, :1], segments2[:, :1])  # [N, M, 1]
    right = torch.min(segments1[:, None, 1:], segments2[:, 1:]) # [N, M, 1]
    
    inter = (right - left).clamp(min=0) # [N, M, 1]
    inter = inter[:, :, 0] # [N, M]
    
    union = len1[:, None] + len2 - inter
    
    iou = inter / union
    return iou, union
    

def generalized_seg_iou(segments1, segments2):
    """
    Generalized IoU from https://giou.stanford.edu/
    
    The segments should be in [x0, x1] format
    
    Returns a [N, M] pairwise matrix, where N = len(segments1)
    and M = len(segments2)
    """
    # degenerate segments gives inf / nan results
    # so do an early check
    assert (segments1[:, 1:] >= segments1[:, :1]).all()
    assert (segments2[:, 1:] >= segments2[:, :1]).all()
    iou, union = seg_iou(segments1, segments2)

    left = torch.min(segments1[:, None, :1], segments2[:, :1])
    right = torch.max(segments1[:, None, 1:], segments2[:, 1:])

    inter = (right - left).clamp(min=0)  # [N,M,1]
    inter = inter[:, :, 0] # [N, M]

    return (iou - (inter - union) / inter)


