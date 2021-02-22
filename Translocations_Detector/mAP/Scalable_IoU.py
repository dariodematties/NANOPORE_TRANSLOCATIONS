# Intersection over Union (IoU) computation
# Source from this amazing tutorial by Aladdin Persson
# https://www.youtube.com/watch?v=XXYG5ZWtjj0
# But I modified it to make it scalable
import torch

def intersection_over_union(segments_preds, segments_labels, segment_format="mix"):
    """
    Calculates intersection over union

    Parameters:
        segments_preds (tensor): Predictions of Bounding Segments (N, 2)
        segments_labels (tensor): Correct labels of Bounding Segments (M, 2)
        segment_format (str): mix/midpoint/extremes, if segments (left,w) or (center,w) or (left,right) respectively

    Returns:
        tensor: Intersection over union for all examples (N, M)
    """

    # segments mix format is [left, width]
    if segment_format == "mix":
        seg1_1 = segments_preds[..., 0:1]                               # (N, 1)
        seg1_2 = segments_preds[..., 0:1] + segments_preds[..., 1:2]    # (N, 1)
        seg2_1 = segments_labels[..., 0:1]                              # (M, 1)
        seg2_2 = segments_labels[..., 0:1] + segments_labels[..., 1:2]  # (M, 1)
    # segments midpoint format is [middle, width]
    elif segment_format == "midpoint":
        seg1_1 = segments_preds[..., 0:1] - segments_preds[..., 1:2] / 2    # (N, 1)
        seg1_2 = segments_preds[..., 0:1] + segments_preds[..., 1:2] / 2    # (N, 1)
        seg2_1 = segments_labels[..., 0:1] - segments_labels[..., 1:2] / 2  # (M, 1)
        seg2_2 = segments_labels[..., 0:1] + segments_labels[..., 1:2] / 2  # (M, 1)
    # segments extremes format is [left, right]
    elif segment_format == "extremes":
        seg1_1 = segments_preds[..., 0:1]   # (N, 1)
        seg1_2 = segments_preds[..., 1:2]   # (N, 1)
        seg2_1 = segments_labels[..., 0:1]  # (M, 1)
        seg2_2 = segments_labels[..., 1:2]  # (M, 1)
    else:
        raise Exception("error: Unrecognized segment format {}" .format(segment_format))

    assert (seg1_2[..., 0] >= seg1_1[..., 0]).all()
    assert (seg2_2[..., 0] >= seg2_1[..., 0]).all()

    left = torch.max(seg1_1[:, None, 0], seg2_1[:,0])
    right = torch.min(seg1_2[:, None, 0], seg2_2[:, 0])

    # .clamp(0) is for the case when they do not intersect
    intersection = (right - left).clamp(min=0)

    seg1_length = abs(seg1_2 - seg1_1)
    seg2_length = abs(seg2_2 - seg2_1)

    union = seg1_length[:, None, 0] + seg2_length[:, 0] - intersection

    # 1e-6 is for numerical stability
    return intersection / (union + 1e-6)
