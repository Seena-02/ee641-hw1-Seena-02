import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json
import os

class KeypointDataset(Dataset):
    def __init__(self, image_dir, annotation_file, output_type='heatmap', 
                 heatmap_size=64, sigma=2.0):
        """
        Initialize the keypoint dataset.
        
        Args:
            image_dir: Path to directory containing images
            annotation_file: Path to JSON annotations
            output_type: 'heatmap' or 'regression'
            heatmap_size: Size of output heatmaps (for heatmap mode)
            sigma: Gaussian sigma for heatmap generation
        """
        self.image_dir = image_dir
        self.output_type = output_type
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        # Load annotations
        with open(annotation_file, "r") as f:
            data = json.load(f)
            self.data = data["images"]  # Use the list of image entries


    def __len__(self):
        return len(self.data)

    
    def generate_heatmap(self, keypoints, height, width):
        """
        Generate gaussian heatmaps for keypoints.
        
        Args:
            keypoints: Array of shape [num_keypoints, 2] in (x, y) format
            height, width: Dimensions of the heatmap
            
        Returns:
            heatmaps: Tensor of shape [num_keypoints, height, width]
        """
        # For each keypoint:
        # 1. Create 2D gaussian centered at keypoint location
        # 2. Handle boundary cases
        num_keypoints = keypoints.shape[0]
        heatmaps = torch.zeros((num_keypoints, height, width), dtype=torch.float32)

        y = torch.arange(0, height, dtype=torch.float32).view(height, 1).expand(height, width)
        x = torch.arange(0, width, dtype=torch.float32).view(1, width).expand(height, width)
        for i, (kx, ky) in enumerate(keypoints):
            if 0 <= kx < width and 0 <= ky < height:
                # Compute squared distance to keypoint
                d2 = (x - kx) ** 2 + (y - ky) ** 2
                exponent = -d2 / (2 * self.sigma ** 2)
                heatmaps[i] = torch.exp(exponent)
                heatmaps[i] /= heatmaps[i].max()
        
        return heatmaps
            

    
    def __getitem__(self, idx):
        """
        Return a sample from the dataset.
        
        Returns:
            image: Tensor of shape [1, 128, 128] (grayscale)
            If output_type == 'heatmap':
                targets: Tensor of shape [5, 64, 64] (5 heatmaps)
            If output_type == 'regression':
                targets: Tensor of shape [10] (x,y for 5 keypoints, normalized to [0,1])
        """
        sample = self.data[idx]

        img_path = os.path.join(self.image_dir, sample["file_name"])
        image = Image.open(img_path).convert("L")
        image = image.resize((128,128))
        image = np.array(image, dtype=np.float32) / 255.0 # normalize
        image = torch.from_numpy(image).unsqueeze(0) # convert to pytorch tensor

        # Load keypoints here
        keypoints = torch.tensor(sample["keypoints"], dtype=torch.float32)
        num_keypoints = keypoints.shape[0]

        if self.output_type == "heatmap":
            # Scale keypoints to heatmap size
            scale_x = self.heatmap_size / image.shape[2]
            scale_y = self.heatmap_size / image.shape[1]
            keypoints_scaled = keypoints.clone()
            keypoints_scaled[:, 0] *= scale_x
            keypoints_scaled[:, 1] *= scale_y

            targets = self.generate_heatmap(keypoints_scaled, self.heatmap_size, self.heatmap_size)

        elif self.output_type == "regression":
            # Normalize keypoints to [0,1]
            h, w = image.shape[1], image.shape[2]
            keypoints_norm = keypoints.clone()
            keypoints_norm[:, 0] /= w
            keypoints_norm[:, 1] /= h

            targets = keypoints_norm.view(-1)  # flatten [10]

        else:
            raise ValueError(f"Invalid output_type: {self.output_type}")

        return image, targets


