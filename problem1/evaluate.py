import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from utils import compute_iou
import numpy as np
from utils import match_anchors_to_targets


# Ensure results directory exists
os.makedirs("results/visualizations", exist_ok=True)

def compute_ap(predictions, ground_truths, iou_threshold=0.5):
    """
    Compute Average Precision for a single class.
    
    Args:
        predictions: list of [boxes, scores] for one class
        ground_truths: list of ground-truth boxes for the same class
        iou_threshold: IoU threshold to consider a detection as True Positive
        
    Returns:
        ap: average precision (float)
    """
    if len(predictions) == 0:
        return 0.0

    # Sort predictions by score descending
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    matched = []

    for i, (box, score) in enumerate(predictions):
        if len(ground_truths) == 0:
            fp[i] = 1
            continue

        ious = compute_iou(torch.tensor([box]), torch.tensor(ground_truths))[0].numpy()
        max_iou_idx = np.argmax(ious)
        max_iou = ious[max_iou_idx]

        if max_iou >= iou_threshold and max_iou_idx not in matched:
            tp[i] = 1
            matched.append(max_iou_idx)
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    precision = cum_tp / (cum_tp + cum_fp + 1e-6)
    recall = cum_tp / (len(ground_truths) + 1e-6)

    # Integrate precision-recall curve using 11-point interpolation
    precisions = []
    for t in np.linspace(0, 1, 11):
        p = precision[recall >= t].max() if np.any(recall >= t) else 0
        precisions.append(p)

    ap = np.mean(precisions)
    return ap


def visualize_detections(image, predictions, ground_truths, save_path):
    """
    Visualize predictions and ground truth boxes.
    Handles DataLoader tuples and tensor images.
    Supports both (x1, y1, x2, y2) and (x, y, w, h) formats.
    """
    import torch

    # If image is a tuple from DataLoader, take the first element
    if isinstance(image, (tuple, list)):
        image = image[0]

    # Convert tensor to numpy
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()

    # If image is in [0,1], scale to [0,255]
    if image.max() <= 1.0:
        image = (image * 255).astype("uint8")

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    def draw_box(box, color="r", score=None):
        """Helper to handle both formats and draw a rectangle."""
        if len(box) == 4:
            x1, y1, x2, y2 = box
            if x2 > x1 and y2 > y1:  # (x1, y1, x2, y2) format
                w, h = x2 - x1, y2 - y1
            else:  # assume (x, y, w, h) format
                x1, y1, w, h = box
        else:
            return  # skip invalid box

        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        if score is not None:
            ax.text(x1, y1 - 2, f"{score:.2f}", color=color, fontsize=8)

    # Draw ground-truth boxes in green
    for box in ground_truths['boxes']:
        draw_box(box, color="g")

    # Draw predicted boxes in red.
    for box, score in predictions:
        draw_box(box, color="r", score=score)

    plt.axis("off")
    plt.savefig(save_path)
    plt.close()


def analyze_scale_performance(model, dataloader, anchors, device="mps"):
    """
    Analyze which scales detect which object sizes and save plots.
    
    Args:
        model: trained MultiScaleDetector
        dataloader: DataLoader for validation dataset
        anchors: list of anchor tensors per scale
        device: "cpu" or "cuda"
        
    Returns:
        scale_stats: dict with counts of small, medium, large objects per scale
    """
    # Define size thresholds (in pixels)

    os.makedirs("results/visualizations", exist_ok=True)

    scale_stats = {"small": [0]*len(anchors),
                   "medium": [0]*len(anchors),
                   "large": [0]*len(anchors)}

    anchor_coverage = [0] * len(anchors)

    size_thresholds = {"small": 32*32, "medium": 96*96}  # area in pixels

    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            images = torch.stack(images).to(device)
            outputs = model(images)

            for scale_idx, output in enumerate(outputs):
                # Decode boxes from predictions
                # output: [batch, num_anchors*(5+num_classes), H, W]
                batch_size, _, H, W = output.shape
                num_anchors = output.shape[1] // (5 + 3)
                output = output.view(batch_size, num_anchors, 5 + 3, H, W)
                # Take objectness scores
                obj_scores = torch.sigmoid(output[:, :, 4, :, :])
                # Take predicted boxes (tx, ty, tw, th)
                # For simplicity, we take the anchor centers + predicted offsets
                # In practice, decode according to your anchor scheme

                for b in range(batch_size):
                    gt_boxes = targets[b]['boxes']
                    for box in gt_boxes:
                        area = (box[2]-box[0])*(box[3]-box[1])
                        if area <= size_thresholds['small']:
                            scale_stats['small'][scale_idx] += 1
                        elif area <= size_thresholds['medium']:
                            scale_stats['medium'][scale_idx] += 1
                        else:
                            scale_stats['large'][scale_idx] += 1

                    # For anchor coverage, match anchors to GT boxes using provided utils
                    scale_anchors = anchors[scale_idx].to(device)
                    matched_labels, matched_boxes, pos_mask, neg_mask = match_anchors_to_targets(
                        scale_anchors, gt_boxes.to(device), targets[b]['labels'].to(device)
                    )
                    # Count positive anchors (coverage)
                    anchor_coverage[scale_idx] += pos_mask.sum().item()

    # Save bar plots per size
    for size, counts in scale_stats.items():
        plt.bar(range(len(counts)), counts)
        plt.xlabel("Scale index")
        plt.ylabel("Num detections")
        plt.title(f"{size.capitalize()} objects per scale")
        plt.savefig(f"results/visualizations/{size}_scale_coverage.png")
        print(f"Saved {size} scale coverage plot")
        plt.close()

    # Save anchor coverage plot
    plt.bar(range(len(anchor_coverage)), anchor_coverage)
    plt.xlabel("Scale index")
    plt.ylabel("Number of positive anchors")
    plt.title("Anchor coverage (positive anchors) per scale")
    plt.savefig("results/visualizations/anchor_coverage_per_scale.png")
    plt.close()

    return scale_stats
