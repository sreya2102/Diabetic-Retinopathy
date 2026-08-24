"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Module.

Computes visual heatmaps highlighting retinal feature regions that drive the PyTorch
DR classifier's grade prediction.
"""

from typing import Dict, Any, Optional
import cv2
import numpy as np
import torch
import torch.nn as nn

from src.classification.model import DRClassifier, build_dr_classifier
from src.preprocessing.normalization import normalize_fundus


class PyTorchGradCAM:
    """
    Grad-CAM implementation for PyTorch convolutional classification models.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        # Default target layer to last conv layer in features sequence if unspecified
        if target_layer is None:
            if hasattr(model, "features"):
                # Find last Conv2d layer in features module
                for module in reversed(list(model.features.modules())):
                    if isinstance(module, nn.Conv2d):
                        target_layer = module
                        break
            if target_layer is None:
                raise ValueError("Could not automatically locate a target Conv2d layer in model.")

        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self, input_tensor: torch.Tensor, target_category: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate normalized Grad-CAM heatmap array (0.0 - 1.0).

        Args:
            input_tensor: Image tensor of shape (1, 3, H, W).
            target_category: Target class index (0-4). If None, uses top predicted class.

        Returns:
            2D numpy array heatmap of shape (H, W) with float values in [0.0, 1.0].
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_category is None:
            target_category = int(torch.argmax(output, dim=1).item())

        score = output[0, target_category]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients or activations during Grad-CAM backward pass.")

        # Compute mean gradient weight per channel
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # Shape (1, C, 1, 1)

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # Shape (1, 1, h, w)
        cam = torch.relu(cam)  # Apply ReLU to isolate positive contributions

        # Normalize CAM map
        cam = cam.squeeze().cpu().numpy()
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        else:
            cam = np.zeros_like(cam)

        return cam


def generate_gradcam(
    model: Optional[Any],
    image: np.ndarray,
    target_layer: Optional[Any] = None,
    target_category: Optional[int] = None,
    alpha: float = 0.5,
) -> Dict[str, Any]:
    """
    Generate Grad-CAM heatmap and visual color overlay for a fundus image array.

    Args:
        model: PyTorch DRClassifier instance or None (builds default if None).
        image: Preprocessed fundus image numpy array (RGB).
        target_layer: Target PyTorch Conv2d layer module.
        target_category: Target class ID (0-4). If None, uses model top prediction.
        alpha: Blending weight for color overlay (0.0 to 1.0).

    Returns:
        Dictionary containing:
        - "original": Original RGB image array
        - "heatmap": 2D float heatmap array (0.0 - 1.0)
        - "overlay": RGB blended overlay image array
        - "status": Implementation status message
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image array provided for Grad-CAM generation.")

    if model is None:
        model = build_dr_classifier(weights_path=None)

    h, w = image.shape[:2]

    # Preprocess image to tensor (1, 3, 512, 512)
    norm_img = normalize_fundus(image, target_size=(512, 512), standard_scaling=True)
    tensor_img = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float()
    tensor_img.requires_grad = True

    try:
        gradcam = PyTorchGradCAM(model, target_layer=target_layer)
        cam_low_res = gradcam.generate(tensor_img, target_category=target_category)

        # Resize heatmap back to original image dimensions (h, w)
        heatmap = cv2.resize(cam_low_res, (w, h), interpolation=cv2.INTER_CUBIC)
        heatmap = np.clip(heatmap, 0.0, 1.0)

        # Generate JET colormap overlay
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        color_cam = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        color_cam_rgb = cv2.cvtColor(color_cam, cv2.COLOR_BGR2RGB)

        # Blend original RGB image with colormap
        overlay = cv2.addWeighted(image, 1.0 - alpha, color_cam_rgb, alpha, 0)

        status = "Grad-CAM Heatmap & Visual Overlay Successfully Generated"

    except Exception as err:
        heatmap = np.zeros((h, w), dtype=np.float32)
        overlay = image.copy()
        status = f"Grad-CAM Generation Notice: {str(err)}"

    return {
        "original": image,
        "heatmap": heatmap,
        "overlay": overlay,
        "status": status,
    }
