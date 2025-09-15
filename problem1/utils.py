import torch
import numpy as np

def generate_anchors(feature_map_sizes, anchor_scales, image_size=224):
    all_anchors = []
    for (fm_h, fm_w), scales in zip(feature_map_sizes, anchor_scales):
        stride_y = image_size / fm_h
        stride_x = image_size / fm_w
        grid_y = np.arange(fm_h) * stride_y + stride_y / 2
        grid_x = np.arange(fm_w) * stride_x + stride_x / 2
        grid_x, grid_y = np.meshgrid(grid_x, grid_y) # [H, W]

        anchors_per_cell = []
        for scale in scales:
            w = h = scale
            x1 = grid_x - w / 2
            y1 = grid_y - h / 2
            x2 = grid_x + w / 2
            y2 = grid_y + h / 2
            anchor = np.stack([x1, y1, x2, y2], axis=-1)  # [H, W, 4]
            anchors_per_cell.append(anchor)
        # Stack to [H, W, num_anchors, 4], then reshape to [H * W * num_anchors, 4]
        anchors_per_cell = np.stack(anchors_per_cell, axis=2)  # [H, W, num_anchors, 4]
        anchors_per_cell = anchors_per_cell.reshape(-1, 4)     # [H * W * num_anchors, 4]
        anchors_per_cell = torch.tensor(anchors_per_cell, dtype=torch.float32)
        all_anchors.append(anchors_per_cell)
    return all_anchors



def compute_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes in a fully vectorized way.
    
    Args:
        boxes1: Tensor of shape [N, 4] (x1, y1, x2, y2)
        boxes2: Tensor of shape [M, 4] (x1, y1, x2, y2)
        
    Returns:
        iou: Tensor of shape [N, M]
    """
    # Ensure same device
    device = boxes1.device
    boxes2 = boxes2.to(device)

    # Ensure float
    boxes1 = boxes1.float()
    boxes2 = boxes2.float()

    N = boxes1.shape[0]
    M = boxes2.shape[0]

    boxes1_exp = boxes1[:, None, :]  # [N, 1, 4]
    boxes2_exp = boxes2[None, :, :]  # [1, M, 4]

    inter_x1 = torch.max(boxes1_exp[..., 0], boxes2_exp[..., 0])
    inter_y1 = torch.max(boxes1_exp[..., 1], boxes2_exp[..., 1])
    inter_x2 = torch.min(boxes1_exp[..., 2], boxes2_exp[..., 2])
    inter_y2 = torch.min(boxes1_exp[..., 3], boxes2_exp[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    union_area = area1[:, None] + area2[None, :] - inter_area
    union_area = union_area.clamp(min=1e-6)  # avoid division by zero

    iou = inter_area / union_area
    return iou


def match_anchors_to_targets(anchors, target_boxes, target_labels, 
                            pos_threshold=0.5, neg_threshold=0.3):
    """
    Match anchors to ground truth boxes.
    
    Args:
        anchors: Tensor of shape [num_anchors, 4]
        target_boxes: Tensor of shape [num_targets, 4]
        target_labels: Tensor of shape [num_targets]
        pos_threshold: IoU threshold for positive anchors
        neg_threshold: IoU threshold for negative anchors
        
    Returns:
        matched_labels: Tensor of shape [num_anchors]
                       (0: background, 1-N: classes)
        matched_boxes: Tensor of shape [num_anchors, 4]
        pos_mask: Boolean tensor indicating positive anchors
        neg_mask: Boolean tensor indicating negative anchors
    """
    device = anchors.device  # ensure same device
    target_boxes = target_boxes.to(device)
    target_labels = target_labels.to(device)

    iou_matrix = compute_iou(anchors, target_boxes)

    max_ious, max_indices = torch.max(iou_matrix, dim=1)

    matched_labels = torch.zeros(len(anchors), dtype=torch.long, device=device)
    matched_boxes = torch.zeros((len(anchors), 4), device=device)

    pos_mask = max_ious >= pos_threshold
    matched_labels[pos_mask] = target_labels[max_indices[pos_mask]]
    matched_boxes[pos_mask] = target_boxes[max_indices[pos_mask]]

    neg_mask = max_ious < neg_threshold

    return matched_labels, matched_boxes, pos_mask, neg_mask





    