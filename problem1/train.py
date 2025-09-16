import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import random

from dataset import ShapeDetectionDataset
from model import MultiScaleDetector
from loss import DetectionLoss
from torchvision import transforms
from utils import generate_anchors
from evaluate import visualize_detections


def train_epoch(model, dataloader, criterion, optimizer, device, anchors):
    """Train for one epoch with debug prints."""
    model.train()
    running_loss = 0.0

    for batch_idx, (images, targets) in enumerate(dataloader):
        #print(f"\nBatch {batch_idx+1}/{len(dataloader)}")
        #print(f"Number of images in batch: {len(images)}")

        # Convert PIL images to tensor batch and move to device
        images = torch.stack([img for img in images]).to(device)
        #print(f"Images tensor shape: {images.shape}")

        # Move target boxes and labels to device
        for i, t in enumerate(targets):
            t['boxes'] = t['boxes'].to(device)
            t['labels'] = t['labels'].to(device)
            #print(f"Image {i}: {t['boxes'].shape[0]} boxes, {t['labels'].shape[0]} labels")

        # Forward pass
        outputs = model(images)
        #for s, out in enumerate(outputs):
            #print(f"Scale {s+1} output shape: {out.shape}")

        # Compute multi-task loss
        loss_dict = criterion(outputs, targets, anchors)
        loss = loss_dict['loss_total']
        #print(f"Loss (total): {loss.item():.4f}")
        #print(f"Loss breakdown: obj={loss_dict['loss_obj'].item():.4f}, "
              #f"cls={loss_dict['loss_cls'].item():.4f}, "
              #f"loc={loss_dict['loss_loc'].item():.4f}")

        # Backward + optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print("Optimizer step done.")

        running_loss += loss.item() * images.size(0)
        #print(f"Cumulative running loss: {running_loss:.4f}")

    epoch_loss = running_loss / len(dataloader.dataset)
    #print(f"Epoch loss: {epoch_loss:.4f}")
    return epoch_loss



def validate(model, dataloader, criterion, device, anchors):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, targets in dataloader:
            images = torch.stack(images).to(device)
            for t in targets:
                t['boxes'] = t['boxes'].to(device)
                t['labels'] = t['labels'].to(device)
            outputs = model(images)
            loss_dict = criterion(outputs, targets, anchors)  # pass anchors here
            loss = loss_dict["loss_total"]
            running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def main():
    # Configuration
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 50
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Initialize dataset, model, loss, optimizer
    # Training loop with logging
    # Save best model and training log

    # Get path to the directory containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))


    transform = transforms.Compose([
        transforms.ToTensor(),          # Converts PIL Image to [C,H,W] tensor in [0,1]
    ])

    train_dataset = ShapeDetectionDataset(
        os.path.join(base_dir, "../datasets/detection/train"),
        os.path.join(base_dir, "../datasets/detection/train_annotations.json"),
        transform=transform
    )

    val_dataset = ShapeDetectionDataset(
        os.path.join(base_dir, "../datasets/detection/val"),
        os.path.join(base_dir, "../datasets/detection/val_annotations.json"),
        transform=transform
    )


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    model = MultiScaleDetector(num_classes=3).to(device)
    criterion = DetectionLoss(num_classes=3)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Training
    best_val_loss = float("inf")
    results = {"train_loss": [], "val_loss": []}

    os.makedirs("results", exist_ok=True)
    # Feature map sizes from your model
    feature_map_sizes = [(56, 56), (28, 28), (14, 14)]
    anchor_scales = [
        [16, 24, 32],    # correct for assignment
        [48, 64, 96],    # correct for assignment
        [96, 128, 192],  # correct for assignment
    ]
    anchors = generate_anchors(feature_map_sizes, anchor_scales, image_size=224)

    print("Starting Training...")
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, anchors)
        val_loss   = validate(model, val_loader, criterion, device, anchors)

        # Run detection visualization on some validation images (take first batch only)
        val_iter = iter(val_loader)
        images, targets = next(val_iter)  # get first batch
        images = torch.stack(images).to(device)
        
        # Forward pass
        outputs = model(images)


        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "results/best_model.pth")

    #import random

    model.eval()
    val_iter = iter(val_loader)
    all_val_images = []
    all_val_targets = []

    # Collect 10 random validation samples (images and targets)
    while len(all_val_images) < 10:
        try:
            images_batch, targets_batch = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            images_batch, targets_batch = next(val_iter)
        for img, tgt in zip(images_batch, targets_batch):
            if len(all_val_images) < 10:
                all_val_images.append(img)
                all_val_targets.append(tgt)
            else:
                break

    with torch.no_grad():
        for i, (image, targets) in enumerate(zip(all_val_images, all_val_targets)):
            image_tensor = image.unsqueeze(0).to(device)  # add batch dim and move to device
            outputs = model(image_tensor)

            predictions = []  # Replace with actual decoded predictions

            visualize_detections(
                image,
                predictions=predictions,
                ground_truths=targets,
                save_path=f"results/visualizations/validation_image_{i+1}.png"
            )


    # Save training log
    with open("results/training_log.json", "w") as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()