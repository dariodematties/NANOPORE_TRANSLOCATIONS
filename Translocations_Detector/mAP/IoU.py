# Intersection over Union (IoU) computation
# Source from this amazing tutorial by Aladdin Persson
# https://www.youtube.com/watch?v=XXYG5ZWtjj0
import torch

def intersection_over_union(segments_preds, segments_labels, segment_format="mix"):
    """
    Calculates intersection over union

    Parameters:
        segments_preds (tensor): Predictions of Bounding Segments (BATCH_SIZE, 2)
        segments_labels (tensor): Correct labels of Bounding Segments (BATCH_SIZE, 2)
        segment_format (str): mix/midpoint/extremes, if segments (left,w) or (center,w) or (left,right) respectively

    Returns:
        tensor: Intersection over union for all examples
    """

    # segments mix format is [left, width]
    if segment_format == "mix":
        seg1_1 = segments_preds[..., 0:1]                               # (BATCH_SIZE, 1)
        seg1_2 = segments_preds[..., 0:1] + segments_preds[..., 1:2]    # (BATCH_SIZE, 1)
        seg2_1 = segments_labels[..., 0:1]                              # (BATCH_SIZE, 1)
        seg2_2 = segments_labels[..., 0:1] + segments_labels[..., 1:2]  # (BATCH_SIZE, 1)
    # segments midpoint format is [middle, width]
    elif segment_format == "midpoint":
        seg1_1 = segments_preds[..., 0:1] - segments_preds[..., 1:2] / 2    # (BATCH_SIZE, 1)
        seg1_2 = segments_preds[..., 0:1] + segments_preds[..., 1:2] / 2    # (BATCH_SIZE, 1)
        seg2_1 = segments_labels[..., 0:1] - segments_labels[..., 1:2] / 2  # (BATCH_SIZE, 1)
        seg2_2 = segments_labels[..., 0:1] + segments_labels[..., 1:2] / 2  # (BATCH_SIZE, 1)
    # segments extremes format is [left, right]
    elif segment_format == "extremes":
        seg1_1 = segments_preds[..., 0:1]   # (BATCH_SIZE, 1)
        seg1_2 = segments_preds[..., 1:2]   # (BATCH_SIZE, 1)
        seg2_1 = segments_labels[..., 0:1]  # (BATCH_SIZE, 1)
        seg2_2 = segments_labels[..., 1:2]  # (BATCH_SIZE, 1)
    else:
        raise Exception("error: Unrecognized segment format {}" .format(segment_format))

    assert (seg1_2[..., 0] >= seg1_1[..., 0]).all()
    assert (seg2_2[..., 0] >= seg2_1[..., 0]).all()

    left = torch.max(seg1_1, seg2_1)
    right = torch.min(seg1_2, seg2_2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (right - left).clamp(min=0)

    seg1_length = abs(seg1_2 - seg1_1)
    seg2_length = abs(seg2_2 - seg2_1)

    union = seg1_length + seg2_length - intersection

    # 1e-6 is for numerical stability
    return intersection / (union + 1e-6)
