import torch
import os
import json
from train import train_heatmap_model
from dataset import KeypointDataset

def ablation_study(dataset_path, model_class):
    """
    Conduct ablation studies on key hyperparameters.
    Experiments to run:
    1. Effect of heatmap resolution (32x32 vs 64x64 vs 128x128)
    2. Effect of Gaussian sigma (1.0, 2.0, 3.0, 4.0)
    3. Effect of skip connections (with vs without)
    """
    results = {}
    heatmap_resolutions = [32, 64, 128]
    sigmas = [1.0, 2.0, 3.0, 4.0]
    skip_connections_options = [True, False]
    base_dir = dataset_path

    # Effect of heatmap resolution
    for res in heatmap_resolutions:
        print(f'Running ablation for heatmap resolution: {res}')
        train_dataset = KeypointDataset(os.path.join(base_dir, 'train'),
                                       os.path.join(base_dir, 'train_annotations.json'),
                                       output_type='heatmap',
                                       heatmap_size=res,
                                       sigma=2.0)
        val_dataset = KeypointDataset(os.path.join(base_dir, 'val'),
                                     os.path.join(base_dir, 'val_annotations.json'),
                                     output_type='heatmap',
                                     heatmap_size=res,
                                     sigma=2.0)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

        model = model_class(num_keypoints=5)
        history = train_heatmap_model(model, train_loader, val_loader, num_epochs=30)
        results[f'heatmap_res_{res}'] = history

    # Effect of Gaussian sigma
    for sigma in sigmas:
        print(f'Running ablation for Gaussian sigma: {sigma}')
        train_dataset = KeypointDataset(os.path.join(base_dir, 'train'),
                                       os.path.join(base_dir, 'train_annotations.json'),
                                       output_type='heatmap',
                                       heatmap_size=64,
                                       sigma=sigma)
        val_dataset = KeypointDataset(os.path.join(base_dir, 'val'),
                                     os.path.join(base_dir, 'val_annotations.json'),
                                     output_type='heatmap',
                                     heatmap_size=64,
                                     sigma=sigma)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

        model = model_class(num_keypoints=5)
        history = train_heatmap_model(model, train_loader, val_loader, num_epochs=30)
        results[f'gaussian_sigma_{sigma}'] = history

    # Effect of skip connections
    # Requires model_class to accept a 'use_skip_connections' parameter or similar
    try:
        for use_skip in skip_connections_options:
            print(f'Running ablation for skip connections: {use_skip}')
            train_dataset = KeypointDataset(os.path.join(base_dir, 'train'),
                                           os.path.join(base_dir, 'train_annotations.json'),
                                           output_type='heatmap',
                                           heatmap_size=64,
                                           sigma=2.0)
            val_dataset = KeypointDataset(os.path.join(base_dir, 'val'),
                                         os.path.join(base_dir, 'val_annotations.json'),
                                         output_type='heatmap',
                                         heatmap_size=64,
                                         sigma=2.0)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

            model = model_class(num_keypoints=5, use_skip_connections=use_skip)
            history = train_heatmap_model(model, train_loader, val_loader, num_epochs=30)
            results[f'skip_connections_{use_skip}'] = history
    except TypeError:
        print('Model class does not support skip connections flag, skipping ablation for skip connections.')

    # Save results to JSON file
    with open('ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results

def analyze_failure_cases(models, test_loader, threshold=0.05, device='cuda'):
    """
    Identify and visualize failure cases among heatmap and regression models.
    
    models: dict with keys 'heatmap' and 'regression' containing the respective trained models.
    test_loader: DataLoader for test data.
    threshold: PCK threshold for success.
    """
    heatmap_model = models['heatmap'].to(device).eval()
    regression_model = models['regression'].to(device).eval()

    failure_cases = {
        'heatmap_success_regression_fail': [],
        'regression_success_heatmap_fail': [],
        'both_fail': []
    }

    from evaluate import extract_keypoints_from_heatmaps, compute_pck, visualize_predictions

    with torch.no_grad():
        for images, gt_targets in test_loader:
            images = images.to(device)
            gt_targets = gt_targets.to(device)

            # Heatmap prediction
            heatmap_outputs = heatmap_model(images)
            heatmap_preds = extract_keypoints_from_heatmaps(heatmap_outputs)  # [batch, num_kp, 2]

            # Regression prediction
            reg_outputs = regression_model(images)  # [batch, num_kp*2]
            reg_preds = reg_outputs.view(reg_outputs.size(0), -1, 2)

            # Ground truths for PCK: shape [batch, num_kp, 2]
            # For regression targets: normalized (x,y), for heatmap targets: convert heatmaps to coords if needed
            # Assuming test_loader yields regression type targets normalized to [0,1]
            gt_coords = gt_targets.view(gt_targets.size(0), -1, 2)

            # Normalize predictions similarly (heatmap coordinates are in heatmap pixels)
            # Convert heatmap_preds from heatmap coordinates to normalized [0,1]
            hmap_size = heatmap_outputs.shape[-1]
            heatmap_preds_norm = heatmap_preds.float() / hmap_size

            # Compute PCK per example for both
            batch_size = images.size(0)
            for i in range(batch_size):
                pck_heatmap = ((torch.norm(heatmap_preds_norm[i] - gt_coords[i], dim=1) < threshold).float().mean().item())
                pck_regression = ((torch.norm(reg_preds[i] - gt_coords[i], dim=1) < threshold).float().mean().item())

                heatmap_success = pck_heatmap == 1.0
                regression_success = pck_regression == 1.0

                if heatmap_success and not regression_success:
                    failure_cases['heatmap_success_regression_fail'].append((images[i].cpu(), heatmap_preds[i].cpu(), gt_coords[i].cpu()))
                elif regression_success and not heatmap_success:
                    failure_cases['regression_success_heatmap_fail'].append((images[i].cpu(), reg_preds[i].cpu(), gt_coords[i].cpu()))
                elif not heatmap_success and not regression_success:
                    failure_cases['both_fail']
