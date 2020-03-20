import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE

    # Compute intersection
    x1 = max(prediction_box[0], gt_box[0])
    y1 = max(prediction_box[1], gt_box[1])
    x2 = min(prediction_box[2], gt_box[2])
    y2 = min(prediction_box[3], gt_box[3])
    
    if ((x1 >= x2) or (y1 >= y2)):
        intersection = 0
    else:
        intersection = abs((x2-x1) * (y2-y1))
    
    # Compute union
    pred_box_area = (prediction_box[2] - prediction_box[0]) * (prediction_box[3] - prediction_box[1])
    gt_box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    union = pred_box_area + gt_box_area - intersection

    iou = intersection/union
    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if num_tp + num_fp == 0:
        return 1
    else:
        return num_tp/(num_tp + num_fp)


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if num_tp + num_fn == 0:
        return 0
    else:
        return num_tp/(num_tp + num_fn)


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold
    """
    Finn all mulige matches. Så for hver pred box finn all ground truth matches og legg til i liste
    """
    box_matches = []
    for pred_box in prediction_boxes:
        for gt_box in gt_boxes:
            if calculate_iou(pred_box, gt_box) >= iou_threshold:
                box_matches.append([calculate_iou(pred_box, gt_box), pred_box, gt_box])
    
    
    # Sort all matches on IoU in descending order
    box_matches.sort(key=lambda match: match[0], reverse=True)

    # Find all matches with the highest IoU threshold
    if len(box_matches):
        match = box_matches[0]
        matched_pred_boxes = np.array([match[1]])
        matched_gt_boxes = np.array([match[2]])
        
        for match in box_matches:
            if not ((match[1] == matched_pred_boxes).all(1).any() or (match[2] == matched_gt_boxes).all(1).any()):
                matched_pred_boxes = np.append(matched_pred_boxes, [match[1]], 0)
                matched_gt_boxes = np.append(matched_gt_boxes, [match[2]], 0)
    else:
        matched_pred_boxes = np.array([])
        matched_gt_boxes = np.array([])

    return matched_pred_boxes, matched_gt_boxes


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """

    matched_pred_boxes, matched_gt_boxes = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)

    results = {}
    results["true_pos"] = matched_pred_boxes.shape[0]

    results["false_pos"] = prediction_boxes.shape[0] - matched_pred_boxes.shape[0]

    results["false_neg"] = gt_boxes.shape[0] - matched_gt_boxes.shape[0]
    return results


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """

    tp = 0
    fp = 0
    fn = 0
    for pred_box, gt_box in zip(all_prediction_boxes, all_gt_boxes):
        r = (calculate_individual_image_result(pred_box, gt_box, iou_threshold))
        tp += r["true_pos"]
        fp += r["false_pos"]
        fn += r["false_neg"]

    return (calculate_precision(tp,fp,fn), calculate_recall(tp,fp,fn))


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)

    # YOUR CODE HERE

    precisions = [] 
    recalls = []


    for threshold in confidence_thresholds:
        img_preds = []

        for img, boxes_p in enumerate(all_prediction_boxes):
            preds = [box_p for i, box_p in enumerate(boxes_p) if confidence_scores[img][i] >= threshold]
            preds = np.array(preds)
            img_preds.append(preds)

        img_preds = np.array(img_preds)
        p, r = calculate_precision_recall_all_images(img_preds, all_gt_boxes, iou_threshold)
        precisions.append(p)
        recalls.append(r)

    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)

    # YOUR CODE HERE
    #for each level in recall levels we have to take the biggest value of precision whose recall value is greater or equal to that
    precision_levels = np.zeros(11)

    for i in range(len(recall_levels)):
        # calculate precission for this recall levell.
        j = 0
        while j < len(recalls) and recalls[j] >= recall_levels[i]:
            precision_levels[i] = precisions[j]
            j += 1


    average_precision = sum(precision_levels)/len(precision_levels)
    return average_precision

def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]
        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)

