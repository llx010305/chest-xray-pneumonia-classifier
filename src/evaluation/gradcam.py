"""Grad-CAM on the last ResNet-18 residual block."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.activations = None
        self.gradients = None
        block = model.layer4[-1]
        block.register_forward_hook(self._save_activation)
        block.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inputs, output) -> None:
        self.activations = output

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0]

    def __call__(self, image: torch.Tensor, class_index: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image.unsqueeze(0))
        logits[0, class_index].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1))
        cam = cam[0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam[None, None], size=image.shape[-2:], mode="bilinear", align_corners=False)
        return cam.squeeze().detach().cpu().numpy()
