"""
PyTorch Neural Network Architecture Definition for DR Severity Classifier.

Implements a 5-class PyTorch convolutional neural network for grading fundus images
according to the International Clinical Diabetic Retinopathy (ICDR) scale.
"""

import os
from typing import Optional, Dict, Any
import torch
import torch.nn as nn


class DRClassifier(nn.Module):
    """
    PyTorch 5-class Diabetic Retinopathy Severity Classifier.
    """

    def __init__(self, num_classes: int = 5, backbone: str = "custom_convnet", dropout_rate: float = 0.3):
        super(DRClassifier, self).__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone

        # Convolutional Feature Extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Classification Linear Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning unnormalized class logits.

        Args:
            x: Input image tensor of shape (N, 3, H, W)

        Returns:
            Logits tensor of shape (N, 5)
        """
        feats = self.features(x)
        feats = torch.flatten(feats, 1)
        logits = self.classifier(feats)
        return logits


def build_dr_classifier(
    weights_path: Optional[str] = None, num_classes: int = 5, device: str = "cpu"
) -> DRClassifier:
    """
    Construct DR classifier model instance and load optional local PyTorch weights (.pt/.pth).

    Args:
        weights_path: Optional local path to saved PyTorch state dict.
        num_classes: Number of target DR grades (default 5).
        device: 'cpu' or 'cuda'.

    Returns:
        Instantiated PyTorch DRClassifier model in eval mode.
    """
    model = DRClassifier(num_classes=num_classes)

    if weights_path is not None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Specified model weights file not found: {weights_path}")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model
