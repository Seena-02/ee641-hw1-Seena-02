import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import os

from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet
from evaluate import extract_keypoints_from_heatmaps, compute_pck, plot_pck_curves, visualize_predictions


def train_heatmap_model(model, train_loader, val_loader, num_epochs=30, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Train the heatmap-based model.
    
    Uses MSE loss between predicted and target heatmaps.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            if outputs.shape[2:] != targets.shape[2:]:
                outputs = nn.functional.interpolate(outputs, size=targets.shape[2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)

                outputs = model(imgs)
                if outputs.shape[2:] != targets.shape[2:]:
                    outputs = nn.functional.interpolate(outputs, size=targets.shape[2:], mode='bilinear', align_corners=False)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * imgs.size(0)

        val_loss /= len(val_loader.dataset)

        # Log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        import matplotlib.pyplot as plt
        base_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(base_dir, "results", "visualizations")
        def save_heatmaps(heatmaps, epoch, results_dir):
            os.makedirs(os.path.join(results_dir, f"heatmaps_epoch_{epoch}"), exist_ok=True)
            heatmaps = heatmaps.cpu()
            batch_size, num_kp, H, W = heatmaps.shape
            for b in range(min(3, batch_size)):  # save max 3 samples
                for k in range(num_kp):
                    plt.imshow(heatmaps[b, k], cmap='hot')
                    plt.axis('off')
                    fname = os.path.join(results_dir, f"heatmaps_epoch_{epoch}", f"sample{b+1}_kp{k+1}.png")
                    plt.savefig(fname)
                    plt.close()

        # Inside train_heatmap_model's epoch loop, after validation loss calc:
        with torch.no_grad():
            sample_imgs, _ = next(iter(val_loader))
            sample_imgs = sample_imgs.to(device)
            pred_heatmaps = model(sample_imgs)
            save_heatmaps(pred_heatmaps, epoch+1, results_dir)


        print(f"[HeatmapNet] Epoch [{epoch+1}/{num_epochs}] | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    # Save final model
    #torch.save(model.state_dict(), "heatmap_model.pth")
    return history


def train_regression_model(model, train_loader, val_loader, num_epochs=30, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Train the direct regression model.
    
    Uses MSE loss between predicted and target coordinates.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * imgs.size(0)

        val_loss /= len(val_loader.dataset)

        # Log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"[RegressionNet] Epoch [{epoch+1}/{num_epochs}] | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    # Save final model
    #torch.save(model.state_dict(), "regression_model.pth")
    return history


def main():
    """
    Main training pipeline.
    Loads real dataset and trains both Heatmap and Regression models.
    Saves models and logs under ./results/
    """
    from baseline import analyze_failure_cases, ablation_study  # import here instead of top level
    # Get path to the directory containing this script
    base_dir = os.getcwd()
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Heatmap datasets
    train_dataset_heatmap = KeypointDataset(
        os.path.join(base_dir, "..", "datasets/keypoints/train"),
        os.path.join(base_dir, "..", "datasets/keypoints/train_annotations.json"),
        output_type="heatmap",
        heatmap_size=64,
        sigma=2.0
    )
    val_dataset_heatmap = KeypointDataset(
        os.path.join(base_dir, "..", "datasets/keypoints/val"),
        os.path.join(base_dir, "..", "datasets/keypoints/val_annotations.json"),
        output_type="heatmap",
        heatmap_size=64,
        sigma=2.0
    )

    # Regression datasets
    train_dataset_reg = KeypointDataset(
        os.path.join(base_dir, "..", "datasets/keypoints/train"),
        os.path.join(base_dir, "..", "datasets/keypoints/train_annotations.json"),
        output_type="regression"
    )
    val_dataset_reg = KeypointDataset(
        os.path.join(base_dir, "..", "datasets/keypoints/val"),
        os.path.join(base_dir, "..", "datasets/keypoints/val_annotations.json"),
        output_type="regression"
    )



    # Dataloaders
    train_loader_heatmap = DataLoader(train_dataset_heatmap, batch_size=32, shuffle=True)
    val_loader_heatmap = DataLoader(val_dataset_heatmap, batch_size=32)
    train_loader_reg = DataLoader(train_dataset_reg, batch_size=32, shuffle=True)
    val_loader_reg = DataLoader(val_dataset_reg, batch_size=32)

    # Models
    heatmap_model = HeatmapNet(num_keypoints=5)
    regression_model = RegressionNet(num_keypoints=5)

    # Training
    heatmap_log = train_heatmap_model(
        heatmap_model, train_loader_heatmap, val_loader_heatmap, num_epochs=1
    )
    regression_log = train_regression_model(
        regression_model, train_loader_reg, val_loader_reg, num_epochs=1
    )

    # Saving models
    torch.save(heatmap_model.state_dict(), os.path.join(results_dir, "heatmap_model.pth"))
    torch.save(regression_model.state_dict(), os.path.join(results_dir, "regression_model.pth"))

    # Save Logs
    all_logs = {
        "heatmap": heatmap_log,
        "regression": regression_log
    }
    with open(os.path.join(results_dir, "training_log.json"), "w") as f:
        import json
        json.dump(all_logs, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # Evaluation and Visualization

    heatmap_model.to(device).eval()
    regression_model.to(device).eval()

    # Collect predictions and ground truths for PCK calculation
    heatmap_preds_all = []
    regression_preds_all = []
    gt_all = []
    images_all = []

    with torch.no_grad():
        for (imgs_hm, targets_hm), (imgs_reg, targets_reg) in zip(val_loader_heatmap, val_loader_reg):
            imgs_hm, targets_hm = imgs_hm.to(device), targets_hm.to(device)
            imgs_reg, targets_reg = imgs_reg.to(device), targets_reg.to(device)

            # Heatmap model predictions
            heatmap_outputs = heatmap_model(imgs_hm)
            heatmap_preds = extract_keypoints_from_heatmaps(heatmap_outputs)  # [batch, num_kp, 2]
            hmap_size = heatmap_outputs.shape[-1]
            heatmap_preds_norm = heatmap_preds.float() / hmap_size  # Normalize coordinates to [0,1]

            # Regression model predictions
            reg_outputs = regression_model(imgs_reg)
            reg_preds = reg_outputs.view(reg_outputs.size(0), -1, 2)  # [batch, num_kp, 2]

            # Ground truth keypoints (normalized) from regression targets
            gt_coords = targets_reg.view(targets_reg.size(0), -1, 2)

            heatmap_preds_all.append(heatmap_preds_norm.cpu())
            regression_preds_all.append(reg_preds.cpu())
            gt_all.append(gt_coords.cpu())
            images_all.append(imgs_hm.cpu())  # use heatmap images for visualization

    heatmap_preds_all = torch.cat(heatmap_preds_all, dim=0)
    regression_preds_all = torch.cat(regression_preds_all, dim=0)
    gt_all = torch.cat(gt_all, dim=0)
    images_all = torch.cat(images_all, dim=0)

    # Compute PCK values at thresholds
    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1]
    pck_heatmap = compute_pck(heatmap_preds_all, gt_all, thresholds, normalize_by='bbox')
    pck_regression = compute_pck(regression_preds_all, gt_all, thresholds, normalize_by='bbox')

    os.makedirs(os.path.join(results_dir, "visualizations"), exist_ok=True)

    # Plot and save PCK curve comparison
    plot_pck_curves(pck_heatmap, pck_regression, save_path=os.path.join(results_dir, "visualizations", "pck_comparison.png"))

    # Visualize some sample predictions
    num_samples_to_visualize = min(5, images_all.size(0))
    for i in range(num_samples_to_visualize):
        image = images_all[i]
        pred_keypoints = heatmap_preds_all[i] * 128  # scale from [0,1] to image pixel coords (128x128)
        gt_keypoints = gt_all[i] * 128
        visualize_predictions(
            image,
            pred_keypoints,
            gt_keypoints,
            save_path=os.path.join(results_dir, "visualizations", f"sample_prediction_{i+1}.png")
        )

    models = {'heatmap': heatmap_model, 'regression': regression_model}
    analyze_failure_cases(models, val_loader_reg, threshold=0.05, device=device)
    ablation_results = ablation_study(
    dataset_path=os.path.join(base_dir, "..", "datasets/keypoints"),
    model_class=HeatmapNet
    )

if __name__ == '__main__':
    main()
