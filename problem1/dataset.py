import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os

class ShapeDetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        """
        Initialize the dataset.
        
        Args:
            image_dir: Path to directory containing images
            annotation_file: Path to COCO-style JSON annotations
            transform: Optional transform to apply to images
        """
        self.image_dir = image_dir
        self.transform = transform
        # Load and parse annotations

        with open(annotation_file, "r") as f:
            coco = json.load(f)
        
        self.images = {img["id"]: img["file_name"] for img in coco["images"]}

        # Store image paths and corresponding annotations
        self.annotations = {img_id: [] for img_id in self.images}

        for annotation in coco["annotations"]:
            self.annotations[annotation["image_id"]].append(annotation)

        self.image_ids = list(self.images.keys())



        pass
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.image_ids)
        pass
    
    def __getitem__(self, idx):
        """
        Return a sample from the dataset.
        
        Returns:
            image: Tensor of shape [3, H, W]
            targets: Dict containing:
                - boxes: Tensor of shape [N, 4] in [x1, y1, x2, y2] format
                - labels: Tensor of shape [N] with class indices (0, 1, 2)
        """

        # Get image id from the specified index.
        img_id = self.image_ids[idx]

        img_path = os.path.join(self.image_dir, self.image_ids[img_id])
        image = Image.open(img_path).convert("RGB")

        # Load the annoations
        anns = self.annotations[img_id]

        # Extract bounding boxes and labels
        boxes = []
        labels = []

        for ann in anns:
            # Format for COCO is [x,y,w,h]
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h]) # Convert to x1, y1, x2, y2
            labels.append(ann["category_id"])
        
        # Convert to Tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        targets = {"boxes": boxes, "labels": labels}

        # Apply the transform if given one
        if self.transform:
            image = self.transform(image)
        
        return image, targets


        pass