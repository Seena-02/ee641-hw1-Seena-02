import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def extract_keypoints_from_heatmaps(heatmaps):
    """
    Extract (x, y) coordinates from heatmaps.
    Args:
        heatmaps: Tensor of shape [batch, num_keypoints, H, W]
    Returns:
        coords: Tensor of shape [batch, num_keypoints, 2]
    """
    batch_size, num_keypoints, H, W = heatmaps.shape
    coords = torch.zeros((batch_size, num_keypoints, 2), device=heatmaps.device)
    for b in range(batch_size):
        for k in range(num_keypoints):
            heatmap = heatmaps[b, k]
            max_val, max_idx = torch.max(heatmap.view(-1), 0)
            y = max_idx // W
            x = max_idx % W
            coords[b, k, 0] = x
            coords[b, k, 1] = y
    return coords

def compute_pck(predictions, ground_truths, thresholds, normalize_by='bbox'):
    """
    Compute PCK at various thresholds.
    Args:
        predictions: Tensor of shape [N, num_keypoints, 2]
        ground_truths: Tensor of shape [N, num_keypoints, 2]
        thresholds: List of threshold values (as fraction of normalization)
        normalize_by: 'bbox' for bounding box diagonal, 'torso' for torso length
    Returns:
        pck_values: Dict mapping threshold to accuracy
    """
    N, num_keypoints, _ = predictions.shape

    # Compute normalization distances for each sample
    norm_dists = torch.zeros(N)
    for i in range(N):
        gt = ground_truths[i]
        if normalize_by == 'bbox':
            x_vals = gt[:, 0]
            y_vals = gt[:, 1]
            width = x_vals.max() - x_vals.min()
            height = y_vals.max() - y_vals.min()
            norm_dists[i] = torch.sqrt(width**2 + height**2)
        elif normalize_by == 'torso':
            # torso length from keypoints 0 and 1
            p1 = gt[0]
            p2 = gt[1]
            norm_dists[i] = torch.norm(p1 - p2)
        else:
            raise ValueError(f"Unsupported normalize_by: {normalize_by}")

    pck_values = {}
    # Compute distances normalized and accuracy at each threshold
    dists = torch.norm(predictions - ground_truths, dim=2)  # [N, num_keypoints]
    for t in thresholds:
        correct = 0
        total = 0
        for i in range(N):
            threshold_dist = t * norm_dists[i]
            correct += (dists[i] <= threshold_dist).sum().item()
            total += num_keypoints
        pck = correct / total
        pck_values[t] = pck
    return pck_values


def plot_pck_curves(pck_heatmap, pck_regression, save_path=None):
    """
    Plot PCK curves comparing both methods.
    Save plot to results/visualizations folder if save_path is not provided.
    """
    if save_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        vis_dir = os.path.join(base_dir, "results", "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        save_path = os.path.join(vis_dir, "pck_comparison.png")
    else:
        # Ensure directory exists even when save_path is given explicitly
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    heatmap_threshs = sorted(pck_heatmap.keys())
    heatmap_vals = [pck_heatmap[t] for t in heatmap_threshs]
    reg_threshs = sorted(pck_regression.keys())
    reg_vals = [pck_regression[t] for t in reg_threshs]

    plt.figure(figsize=(8,6))
    plt.plot(heatmap_threshs, heatmap_vals, label='HeatmapNet')
    plt.plot(reg_threshs, reg_vals, label='RegressionNet')
    plt.xlabel('Threshold')
    plt.ylabel('PCK Accuracy')
    plt.title('PCK Curve Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Saving PCK Curves to {save_path}")
    plt.close()

def visualize_predictions(image, pred_keypoints, gt_keypoints, save_path=None):
    """
    Visualize predicted and ground truth keypoints on image.
    Save image to results/visualizations/sample_predictions if save_path not provided.
    """
    if save_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        vis_dir = os.path.join(base_dir, "results", "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        # Generate filename example or raise error to require explicit filename
        save_path = os.path.join(vis_dir, "sample_prediction.png")
    else:
        # Ensure directory exists even when save_path is given explicitly
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure()
    if torch.is_tensor(image):
        image = image.squeeze().cpu().numpy()
    plt.imshow(image, cmap='gray')

    pred = pred_keypoints.cpu().numpy()
    gt = gt_keypoints.cpu().numpy()

    # Plot GT keypoints
    for x, y in gt:
        plt.scatter(x, y, c='g', marker='o', label='GT')
    # Plot Predicted keypoints
    for x, y in pred:
        plt.scatter(x, y, c='r', marker='x', label='Pred')

    # Ensure legend only shows once for each label
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title('Predicted vs Ground Truth Keypoints')
    plt.savefig(save_path)
    print(f"Saving visualization to {save_path}")
    plt.close()

    
