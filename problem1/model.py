import torch
import torch.nn as nn

class MultiScaleDetector(nn.Module):
    def __init__(self, num_classes=3, num_anchors=3):
        """
        Initialize the multi-scale detector.
        
        Args:
            num_classes: Number of object classes (not including background)
            num_anchors: Number of anchors per spatial location
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Feature extraction backbone
        # Extract features at 3 different scales
        
        # Detection heads for each scale
        # Each head outputs: [batch, num_anchors * (4 + 1 + num_classes), H, W]

        # Had GPT generate instructions
        # ------------------------------------------------------------
        # BACKBONE (feature extractor) - 4 convolutional blocks
        # We will build 4 blocks that progressively reduce spatial size:
        #   Input: [B, 3, 224, 224]
        #   After Stem: [B, 64, 112, 112]     (stem contains two convs)
        #   Block2 (Scale1): [B, 128, 56, 56]
        #   Block3 (Scale2): [B, 256, 28, 28]
        #   Block4 (Scale3): [B, 512, 14, 14]
        # Each block uses Conv -> BatchNorm -> ReLU.
        # Use padding=1 with 3x3 convs to preserve "odd/even" alignment and keep halves exact.
        # ------------------------------------------------------------

        # The stem contains two convs
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.RelU(inplace=True)
        )


        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        head_out_channels = self.num_anchors * (5 + self.num_classes)

        self.head_scale1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inlace=True),
            nn.Conv2d(128, head_out_channels, kernel_size=1, stride=1, padding=0)
        )

        self.head_scale2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inlace=True),
            nn.Conv2d(256, head_out_channels, kernel_size=1, stride=1, padding=0)
        )

        self.head_scale3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inlace=True),
            nn.Conv2d(512, head_out_channels, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, 3, 224, 224]
            
        Returns:
            List of 3 tensors (one per scale), each containing predictions
            Shape: [batch, num_anchors * (5 + num_classes), H, W]
            where 5 = 4 bbox coords + 1 objectness score
        """
        x.self.stem(x)
        feat_s1 = self.block(x)
        feat_s2 = self.block2(feat_s1)
        feat_s3 = self.block3(feat_s2)

        pred_s1 = self.head_scale1(feat_s1)
        pred_s2 = self.head_scale2(feat_s2)
        pred_s3 = self.head_scale3(feat_s3)

        return [pred_s1, pred_s2, pred_s3]