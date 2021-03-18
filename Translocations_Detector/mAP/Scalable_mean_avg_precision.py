# Mean average precision (mAP) computation
# Source from this amazing tutorial by Aladdin Persson
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/metrics/mean_avg_precision.py
# But I modified it to make it scalable
import sys
import torch
from collections import Counter

sys.path.append('./mAP')
from Scalable_IoU import intersection_over_union

def mean_average_precision(
        device, pred_segments, true_segments, iou_threshold=0.5, seg_format="mix", num_classes=1
):
    """
    Calculates mean average precision 

    Parameters:
        device (str): the device used in the run
        pred_segments (list): list of lists containing all bsegments with each bsegments
        specified as [train_idx, class_prediction, prob_score, x1, x2]
        true_segments (list): Similar as pred_segments except all the correct ones 
        iou_threshold (float): threshold where predicted bsegments is correct
        segment_format (str): mix/midpoint/extremes, if (left,w) or (center,w) or (left,right) are respectively used to specify bsegments
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_segments:
            if detection[1] == c:
                detections.append(detection)

        for true_segment in true_segments:
            if true_segment[1] == c:
                ground_truths.append(true_segment)


        # find the amount of bsegments for each training example
        # Counter here finds how many ground truth bsegments we get
        # for each training example, so let's say wind 0 has 3,
        # wind 1 has 5 then we will obtain a dictionary with:
        # amount_bsegments = {0:3, 1:5}
        amount_bsegments = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bsegments = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bsegments.items():
            amount_bsegments[key] = torch.zeros(val)

        # sort by segment probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bsegments = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bsegments == 0:
            continue

        ground_truths = torch.tensor(ground_truths)
        for detection_idx, detection in enumerate(detections):
            detection = torch.tensor(detection)
            # Only take out the ground_truths that have the same
            # training idx as detection
            loc = detection[0] == ground_truths[:,0]
            ground_truth_wind = ground_truths[loc]

            num_gts = ground_truth_wind.shape[0]
            best_iou = 0

            if num_gts > 0:
                iou = intersection_over_union(
                    torch.unsqueeze(detection[3:],0).to(device),
                    ground_truth_wind[:,3:].to(device),
                    segment_format=seg_format,
                )
                best_iou = torch.max(iou)
                best_gt_idx = torch.argmax(iou)

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bsegments[int(detection[0])][best_gt_idx] == 0:
                    # true positive and add this bounding segment to seen
                    TP[detection_idx] = 1
                    amount_bsegments[int(detection[0])][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bsegments + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]).type(torch.float32), precisions))
        recalls = torch.cat((torch.tensor([0]).type(torch.float32), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)
