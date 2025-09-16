import os
import json
import torch
from torch.utils.data import DataLoader
from evaluate import extract_keypoints_from_heatmaps
import matplotlib.pyplot as plt

def ablation_study(dataset_path, model_class):
    """
    Conduct ablation studies on key hyperparameters.
    
    Experiments to run:
    1. Effect of heatmap resolution (32x32 vs 64x64 vs 128x128)
    2. Effect of Gaussian sigma (1.0, 2.0, 3.0, 4.0)
    3. Effect of skip connections (with vs without)
    """
    from train import train_heatmap_model
    from dataset import KeypointDataset

    heatmap_resolutions = [32, 64, 128]
    sigmas = [1.0, 2.0, 3.0, 4.0]
    skip_options = [True, False]

    results = {}

    base_dir = dataset_path
    os.makedirs("results", exist_ok=True)
    vis_dir = os.path.join("results", "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    print("Starting ablation study...")

    # 1. Heatmap resolution
    for res in heatmap_resolutions:
        print(f"Training with heatmap resolution {res}")
        train_ds = KeypointDataset(
            os.path.join(base_dir, "train"),
            os.path.join(base_dir, "train_annotations.json"),
            output_type='heatmap',
            heatmap_size=res,
            sigma=2.0)
        val_ds = KeypointDataset(
            os.path.join(base_dir, "val"),
            os.path.join(base_dir, "val_annotations.json"),
            output_type='heatmap',
            heatmap_size=res,
            sigma=2.0)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)
        model = model_class(num_keypoints=5)
        history = train_heatmap_model(model, train_loader, val_loader, num_epochs=30)
        results[f'heatmap_resolution_{res}'] = history
        print(f"Resolution {res} final val loss: {history['val_loss'][-1]:.4f}")

    # 2. Gaussian sigma
    for sigma in sigmas:
        print(f"Training with Gaussian sigma {sigma}")
        train_ds = KeypointDataset(
            os.path.join(base_dir, "train"),
            os.path.join(base_dir, "train_annotations.json"),
            output_type='heatmap',
            heatmap_size=64,
            sigma=sigma)
        val_ds = KeypointDataset(
            os.path.join(base_dir, "val"),
            os.path.join(base_dir, "val_annotations.json"),
            output_type='heatmap',
            heatmap_size=64,
            sigma=sigma)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)
        model = model_class(num_keypoints=5)
        history = train_heatmap_model(model, train_loader, val_loader, num_epochs=30)
        results[f'gaussian_sigma_{sigma}'] = history
        print(f"Sigma {sigma} final val loss: {history['val_loss'][-1]:.4f}")

    # 3. Skip connections
    for use_skip in skip_options:
        try:
            print(f"Training with skip connections: {use_skip}")
            train_ds = KeypointDataset(
                os.path.join(base_dir, "train"),
                os.path.join(base_dir, "train_annotations.json"),
                output_type='heatmap',
                heatmap_size=64,
                sigma=2.0)
            val_ds = KeypointDataset(
                os.path.join(base_dir, "val"),
                os.path.join(base_dir, "val_annotations.json"),
                output_type='heatmap',
                heatmap_size=64,
                sigma=2.0)
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=32)
            model = model_class(num_keypoints=5, use_skip_connections=use_skip)
            history = train_heatmap_model(model, train_loader, val_loader, num_epochs=30)
            results[f'skip_connections_{use_skip}'] = history
            print(f"Skip connections {use_skip} final val loss: {history['val_loss'][-1]:.4f}")
        except TypeError:
            print("Model does not support skip_connections parameter, skipping...")

    # Save results as JSON
    with open("results/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Ablation study complete. Results saved to results/ablation_results.json")
    return results


def analyze_failure_cases(models, test_loader, threshold=0.05, device='cuda'):
    """
    Identify and visualize failure cases.
    
    Find examples where:
    1. Heatmap succeeds but regression fails
    2. Regression succeeds but heatmap fails
    3. Both methods fail
    """
    heatmap_model = models['heatmap'].to(device).eval()
    regression_model = models['regression'].to(device).eval()

    failure_cases = {
        'heatmap_success_regression_fail': [],
        'regression_success_heatmap_fail': [],
        'both_fail': []
    }

    os.makedirs("results/visualizations/failure_cases", exist_ok=True)

    with torch.no_grad():
        for images, gt_targets in test_loader:
            images = images.to(device)
            gt_targets = gt_targets.to(device)

            heatmap_outputs = heatmap_model(images)
            heatmap_preds = extract_keypoints_from_heatmaps(heatmap_outputs)  # [B, num_kp, 2]
            reg_outputs = regression_model(images)  # [B, num_kp*2]
            reg_preds = reg_outputs.view(reg_outputs.size(0), -1, 2)  # reshape [B, num_kp, 2]

            gt_coords = gt_targets.view(gt_targets.size(0), -1, 2)

            hmap_size = heatmap_outputs.shape[-1]
            heatmap_preds_norm = heatmap_preds.float() / hmap_size

            for i in range(images.size(0)):
                pck_heatmap = (torch.norm(heatmap_preds_norm[i] - gt_coords[i], dim=1) < threshold).float().mean().item()
                pck_regression = (torch.norm(reg_preds[i] - gt_coords[i], dim=1) < threshold).float().mean().item()

                heatmap_success = (pck_heatmap == 1.0)
                regression_success = (pck_regression == 1.0)

                if heatmap_success and not regression_success:
                    failure_cases['heatmap_success_regression_fail'].append((images[i].cpu(), heatmap_preds[i].cpu(), gt_coords[i].cpu()))
                elif regression_success and not heatmap_success:
                    failure_cases['regression_success_heatmap_fail'].append((images[i].cpu(), reg_preds[i].cpu(), gt_coords[i].cpu()))
                elif not heatmap_success and not regression_success:
                    failure_cases['both_fail'].append((images[i].cpu(), heatmap_preds[i].cpu(), gt_coords[i].cpu()))

    # Save visualizations inline:
    import matplotlib.pyplot as plt

    def single_viz(img, pred_kp, gt_kp, path):
        plt.figure()
        img_np = img.squeeze().numpy()
        plt.imshow(img_np, cmap='gray')
        pred_np = pred_kp.numpy()
        gt_np = gt_kp.numpy()
        for x, y in gt_np:
            plt.scatter(x, y, c='g', marker='o', label='GT')
        for x, y in pred_np:
            plt.scatter(x, y, c='r', marker='x', label='Pred')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.title(os.path.basename(path))
        plt.savefig(path)
        plt.close()

    for category, cases in failure_cases.items():
        for idx, (img, pred, gt) in enumerate(cases):
            if category == 'regression_success_heatmap_fail':
                pred_vis = pred * 128
            else:
                pred_vis = pred * (128 / hmap_size)
            gt_vis = gt * 128
            save_path = os.path.join("results", "visualizations", "failure_cases", f"{category}_{idx+1}.png")
            single_viz(img, pred_vis, gt_vis, save_path)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    vis_dir = os.path.join(base_dir, "results", "visualizations", "failure_cases")
    os.makedirs(vis_dir, exist_ok=True)

    for fail_type, cases in failure_cases.items():
        for idx, (img, pred, gt) in enumerate(cases):
            if fail_type != 'regression_success_heatmap_fail':
                pred_vis = pred * (128 / hmap_size)
            else:
                pred_vis = pred * 128
            gt_vis = gt * 128
            plt.figure()
            img_np = img.squeeze().numpy()
            plt.imshow(img_np, cmap='gray')
            for x, y in gt_vis.numpy():
                plt.scatter(x, y, c='g', marker='o')
            for x, y in pred_vis.numpy():
                plt.scatter(x, y, c='r', marker='x')
            plt.title(f"{fail_type}_{idx+1}")
            plt.axis('off')
            plt.savefig(os.path.join(vis_dir, f"{fail_type}_{idx+1}.png"))
            plt.close()


    print(f"Failure case visualizations saved to results/visualizations/failure_cases/")
