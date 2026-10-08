"""ResNet-18 dataset, augmentation, and model construction."""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import functional as TF

from src.common import ROOT
from src.preprocessing.ops import read_gray

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def configure_torch_home() -> None:
    cache = ROOT / ".cache" / "torch"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(cache))


def build_model() -> torch.nn.Module:
    configure_torch_home()
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    return model


def _augment(image: torch.Tensor) -> torch.Tensor:
    if torch.rand(1).item() < 0.5:
        image = torch.flip(image, dims=[2])
    angle = float(torch.empty(1).uniform_(-15.0, 15.0))
    image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
    factor = float(torch.empty(1).uniform_(0.8, 1.2))
    center = image.mean()
    return torch.clamp((image - center) * factor + center, 0.0, 1.0)


class XrayDataset(Dataset):
    """Preprocessed grayscale PNG files. Augmentation is applied to the training split only."""

    def __init__(self, rows: list[dict], train: bool) -> None:
        self.rows = rows
        self.train = train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = read_gray(ROOT / row["processed_path"])
        tensor = torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0) / 255.0
        if self.train:
            tensor = _augment(tensor)
        tensor = tensor.repeat(3, 1, 1)
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        label = 1 if row["label"] == "PNEUMONIA" else 0
        return tensor, label
