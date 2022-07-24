"""
DETR model and criterion classes.
"""
import sys
import torch
import torch.nn.functional as F
from torch import nn

from misc import (accuracy, is_dist_avail_and_initialized, get_world_size)

sys.path.append('./util')
import seg_ops

from backbone import build_backbone


class DETR(nn.Module):
    """ This is the DETR module that performs translocation detection """
    def __init__(self, backbone, transformer, num_classes, num_queries):
        """ Initializes the model.

        Parameters:
            backbone:           torch module of the backbone to be used. See backbone.py
            transformer:        torch module of the transformer architecture. See transformer.py
            num_classes:        number of translocation classes
            num_queries:        number of translocation queries, ie detection slot. This is the maximal number of translocations
                                DETR can detect in a single signal window.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        # prediction heads, one extra class for predicting non-empty slots
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bsegment_embed = MLP(hidden_dim, hidden_dim, 2, 3)

        self.input_proj = nn.Conv1d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

        # output positional encodings (translocation queries)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)


        # spatial positional encodings
        self.pos_embed = nn.Parameter(torch.rand(10*num_queries, hidden_dim))

    def forward(self, inputs):
        """ The forward expects samples, which is a tensor and consists of:

               - samples: batched signal windows, of shape [batch_size x 1 x W]

            It returns a dict with the following elements:

               - "pred_logits":   the classification logits (including no-translocation) for all queries.
                                  Shape= [batch_size x num_queries x (num_classes + 1)]

               - "pred_segments": The normalized segments coordinates for all queries, represented as
                                  (center_x, width). These values are normalized in [0, 1],
                                  relative to the size of each individual window
        """
        # propagate inputs through the convolutional backbone
        x = self.backbone(inputs)   # x [batch_size, backbone number of channels * channel_dim]
                                    # where the backbone number of channels is 512 if backbone feature_predictor is resnet10,resnet18 or resnet34
                                    # and it is 2048 if feature_predictor is resnet50 or resnet101 or resnet152

        # Addapt the backbone output so it satisfies the required format
        # x = x[:,0:-1]
        x = x.view(inputs.shape[0], self.backbone.num_channels, -1)    # x [batch_size, procesed_window_size, channel_dim]

        # convert from backbone number of channels to hidden_dim feature vectors for the transformer
        h = self.input_proj(x)      # h [batch_size, hidden_dim, channel_dim]

        # construct positional encodings (W stands for window Width)
        W = h.shape[-1]
        pos = self.pos_embed[:W].unsqueeze(1).repeat(1, h.shape[0], 1)  # pos [channel_dim, batch_size, hidden_dim]

        # propagate through the transformer
        # h [num_queries, batch_size, hidden_dim]
        hs = self.transformer(h, self.query_embed.weight, pos.permute(1, 2, 0))[0]
        
        # finally project transformer outputs to class labels and bounding boxes
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bsegment_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1],        # [batch_size, num_queries, num_classes]
               'pred_segments': outputs_coord[-1]}      # [batch_size, num_queries, 2]

        # outputs_class shape is [num_decoder_layers, batch_size, num_queries, num_classes]
        # so outputs_class[-1] is the output from the last transformer decoder layer
        # outputs_coord shape is [num_decoder_layers, batch_size, num_queries, 2]
        # so outputs_coord[-1] is the output from the last transformer decoder layer
        return out










class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.

    The process happens in two steps:
        1) we compute hungarian assignment between ground truth segments and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and segment)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
      """ Create the criterion.

        Parameters:
            num_classes: number of translocation categories, omitting the special no-translocation category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-translocation category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
      """
      super().__init__()
      self.num_classes = num_classes
      self.matcher = matcher
      self.weight_dict = weight_dict
      self.eos_coef = eos_coef
      self.losses = losses
      empty_weight = torch.ones(self.num_classes + 1)
      empty_weight[-1] = self.eos_coef
      self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_segments, log=True):
      """Classification loss (NLL)
      targets dicts must contain the key "labels" containing a tensor of dim [nb_target_segments]
      """
      assert 'pred_logits' in outputs
      src_logits = outputs['pred_logits']

      idx = self._get_src_permutation_idx(indices)
      target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)]).long()
      target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                  dtype=torch.int64, device=src_logits.device)
      target_classes[idx] = target_classes_o

      loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
      losses = {'loss_ce': loss_ce}

      if log:
        # TODO this should probably be a separate loss, not hacked in this one here
        losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        
      return losses



    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_segments):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty segments
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-translocation" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses


    def loss_segments(self, outputs, targets, indices, num_segments):
        """Compute the losses related to the bounding segments, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center_x, w), normalized by the image size.
        """
        assert 'pred_segments' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_segments = outputs['pred_segments'][idx]
        target_segments = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bsegment = F.l1_loss(src_segments, target_segments, reduction='none')

        losses = {}
        losses['loss_bsegment'] = loss_bsegment.sum() / num_segments

        loss_giou = 1 - torch.diag(seg_ops.generalized_seg_iou(
            seg_ops.seg_cxw_to_x0x1(seg_ops.seg_bxw_to_cxw(src_segments)),
            seg_ops.seg_cxw_to_x0x1(seg_ops.seg_bxw_to_cxw(target_segments))))
        losses['loss_giou'] = loss_giou.sum() / num_segments
        return losses



    def _get_src_permutation_idx(self, indices):
      # permute predictions following indices
      batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
      src_idx = torch.cat([src for (src, _) in indices])
      return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


    def get_loss(self, loss, outputs, targets, indices, num_segments, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'segments': self.loss_segments
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_segments, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.

        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target segments accross all nodes, for normalization purposes
        num_segments = sum(len(t["labels"]) for t in targets)
        num_segments = torch.as_tensor([num_segments], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_segments)
        num_segments = torch.clamp(num_segments / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_segments))


        return losses
    


















class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

