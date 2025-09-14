import torch
import numpy as np

def generate_anchors(feature_map_sizes, anchor_scales, image_size=224):
    """
    Generate anchors for multiple feature maps.
    
    Args:
        feature_map_sizes: List of (H, W) tuples for each feature map
        anchor_scales: List of lists, scales for each feature map
        image_size: Input image size
        
    Returns:
        anchors: List of tensors, each of shape [num_anchors, 4]
                 in [x1, y1, x2, y2] format
    """
    # For each feature map:
    # 1. Create grid of anchor centers
    # 2. Generate anchors with specified scales and ratios
    # 3. Convert to absolute coordinates
    
    all_anchors = []

    for (fm_h, fm_w), scales in zip(feature_map_sizes, anchor_scales):
        stride_y = image_size / fm_h
        stride_x = image_size / fm_w

        grid_y = np.arrange(fm_h) * stride_y + stride_y / 2
        grid_x = np.arange(fm_w) * stride_x + stride_x / 2
        grid_x, grid_y = np.meshgrid(grid_x, grid_y) #shape [H, W]

        centers = np.stack([grid_x, grid_y], axis=1) # [H, W, 2]

        anchors = []
        for scale in scales:
            w = h = scale # since aspect ratio is 1:1
            # Format to convert corner [x1,y1,x2,y2]
            x1 = centers[..., 0] - w / 2
            y1 = centers[..., 1] - h / 2
            x2 = centers[..., 0] + w / 2
            y2 = centers[..., 1] + h / 2

            anchor_boxes = np.stack([x1,y1,x2,y2], axis=-1) # [H, W, 4]
            anchors.append(anchor_boxes)
        
        anchors = np.concatenate(anchors, axis=-2) #[H, W, num_scales, 4]
        anchors = anchors.reshape(-1, 4) #[num_anchors, 4]
        anchors = torch.tensor(anchors, dtype=torch.float32)

        all_anchors.append(anchors)

        return all_anchors



def compute_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: Tensor of shape [N, 4]
        boxes2: Tensor of shape [M, 4]
        
    Returns:
        iou: Tensor of shape [N, M]
    """
    N, M = len(boxes1), len(boxes2)
    ious = torch.zeros((N, M))

    for i in range(N):
        for j in range(M):
            box1_x1, box1_y1, box1_x2, box1_y2 = boxes1[i]
            box2_x1, box2_y1, box2_x2, box2_y2 = boxes2[j]

            inter_x1 = max(box1_x1, box2_x1)
            inter_y1 = max(box1_y1, box2_y1)
            inter_x2 = min(box1_x2, box2_x2)
            inter_y2 = min(box1_y2, box2_y2)

            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h

            area1 = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
            area2 = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
            union_area = area1 + area2 - inter_area

            ious[i, j] = inter_area / union_area if union_area > 0 else 0

    return ious



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
    iou_matrix = compute_iou(anchors, target_boxes)

    max_ious, max_indices = torch.max(iou_matrix, dim=1)

    matched_labels = torch.zeros(len(anchors), dtype=torch.long)
    matched_boxes = torch.zeros((len(anchors), 4))

    pos_mask = torch.zeros(len(anchors), dtype=torch.bool)
    neg_mask = torch.zeros(len(anchors), dtype=torch.bool)

    pos_mask = max_ious >= pos_threshold
    matched_labels[pos_mask] = target_labels[max_indices[pos_mask]]
    matched_boxes[pos_mask] = target_boxes[max_indices[pos_mask]]

    neg_mask = max_ious < neg_threshold

    return matched_labels, matched_boxes, pos_mask, neg_mask





    