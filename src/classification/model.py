"""
PyTorch Neural Network Architecture Definition for DR Severity Classifier.
"""

from typing import Optional
import torch
import torch.nn as nn


class DRClassifier(nn.Module):
    """
    PyTorch 5-class Diabetic Retinopathy Severity Classifier.
    """

    def __init__(self, num_classes: int = 5, backbone: str = "custom_convnet"):
        super(DRClassifier, self).__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone

        # Lightweight convolutional feature extractor for baseline / architecture initialization
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw class logits.

        Args:
            x: Input tensor of shape (N, C, H, W)

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
    Construct DR classifier model instance and load optional local weights.

    Args:
        weights_path: Optional local path to saved PyTorch state dict (.pt / .pth).
        num_classes: Number of DR grades (default 5).
        device: 'cpu' or 'cuda'.

    Returns:
        Instantiated PyTorch DRClassifier model in eval mode.
    """
    model = DRClassifier(num_classes=num_classes)
    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model
