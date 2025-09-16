import torch
import torch.nn as nn
import torch.nn.functional as F

class HeatmapNet(nn.Module):
    def __init__(self, num_keypoints=5):
        """
        Initialize the heatmap regression network.
        
        Args:
            num_keypoints: Number of keypoints to detect
        """
        super().__init__()
        self.num_keypoints = num_keypoints
        
        # Encoder (downsampling path)
        # Input: [batch, 1, 128, 128]
        # Progressively downsample to extract features
        
        # Decoder (upsampling path)
        # Progressively upsample back to heatmap resolution
        # Output: [batch, num_keypoints, 64, 64]
        
        # Skip connections between encoder and decoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(32, num_keypoints, kernel_size=1)

    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, 1, 128, 128]
            
        Returns:
            heatmaps: Tensor of shape [batch, num_keypoints, 64, 64]
        """
    # --- Encoder ---
        x1 = self.conv1(x)  # [B, 32, 64, 64]
        x2 = self.conv2(x1) # [B, 64, 32, 32]
        x3 = self.conv3(x2) # [B, 128, 16, 16]
        x4 = self.conv4(x3) # [B, 256, 8, 8]

        # --- Decoder ---
        x = self.deconv4(x4)           # 8 → 16
        x = torch.cat([x, x3], dim=1)  # skip connection (128+128=256)

        x = self.deconv3(x)            # 16 → 32
        x = torch.cat([x, x2], dim=1)  # skip connection (64+64=128)

        x = self.deconv2(x)            # 32 → 64
        # optionally: concat x1 here if you want another skip (not required)

        heatmaps = self.final_conv(x)  # [B, num_keypoints, 64, 64]
        return heatmaps


class RegressionNet(nn.Module):
    def __init__(self, num_keypoints=5):
        """
        Initialize the direct regression network.
        
        Args:
            num_keypoints: Number of keypoints to detect
        """
        super().__init__()
        self.num_keypoints = num_keypoints
        
        # Use same encoder architecture as HeatmapNet
        # But add global pooling and fully connected layers
        # Output: [batch, num_keypoints * 2]
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )

        # --- Regression head ---
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # [B, 256, 8, 8] -> [B, 256, 1, 1]
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, num_keypoints * 2)  # predict x,y for each keypoint
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, 1, 128, 128]
            
        Returns:
            coords: Tensor of shape [batch, num_keypoints * 2]
                   Values in range [0, 1] (normalized coordinates)
        """
        # Encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)  # [B, 256, 8, 8]

        # Global pooling
        x = self.global_pool(x)        # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)      # flatten to [B, 256]

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        coords = torch.sigmoid(self.fc3(x))  # normalized [0,1] coordinates

        return coords  # [B, num_keypoints*2]