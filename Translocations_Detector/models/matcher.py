# This is basically a copy-paste from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""

import sys
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

sys.path.append('./util')
from seg_ops import seg_bxw_to_cxw, seg_cxw_to_x0x1, generalized_seg_iou

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    
    def __init__(self, cost_class: float = 1, cost_bsegment: float = 1, cost_giou: float = 1):
        """Creates the matcher
        
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bseg: This is the relative weight of the L1 error of the bounding segment coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding segment in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bsegment = cost_bsegment
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bseg != 0 or cost_giou != 0, "all costs cant be 0"
        
    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_segments": Tensor of dim [batch_size, num_queries, 2] with the predicted segment coordinates
                 
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_segments] (where num_target_segments is the number of ground-truth
                           pulses in the target) containing the class labels
                 "segments": Tensor of dim [num_target_segments, 2] containing the target segments coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_segments)
        """
        
        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        
        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bsegment = outputs["pred_segments"].flatten(0, 1)        # [batch_size * num_queries, 2]

        # Also concat the target labels and segments
        tgt_ids = torch.cat([v["labels"] for v in targets]).long()
        # [num_target_segments_(1) + num_target_segments_(2) + ... + num_target_segments_(batch_size)]
        tgt_bsegment = torch.cat([v["segments"] for v in targets])
        # [num_target_segments_(1) + num_target_segments_(2) + ... + num_target_segments_(batch_size), 2]
        
        # Compute the classification cost. Contrary to the loss, we don't use the Negative Log-Likelihood (NLL),
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between segments
        cost_bsegment = torch.cdist(out_bsegment, tgt_bsegment, p=1)

        # Compute the giou cost betwen segments
        cost_giou = -generalized_seg_iou(seg_cxw_to_x0x1(seg_bxw_to_cxw(out_bsegment)), seg_cxw_to_x0x1(seg_bxw_to_cxw(tgt_bsegment)))

        # Final cost matrix
        C = self.cost_bsegment * cost_bsegment + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v['segments']) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bsegment=args.set_cost_bsegment, cost_giou=args.set_cost_giou)
