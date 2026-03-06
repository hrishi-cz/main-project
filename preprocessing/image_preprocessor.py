"""Image preprocessing – callable torchvision transform pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
import torchvision.transforms as transforms


class ImagePreprocessor:
    """
    ImageNet-normalised torchvision transform pipeline.

    The instance is callable (implements ``__call__``) so it can be passed
    directly as the ``transform`` argument to any ``torch.utils.data.Dataset``
    or ``torchvision.datasets.*`` class.

    Usage
    -----
    >>> ip = ImagePreprocessor()
    >>> dataset = MyDataset(transform=ip)   # pass as callable transform
    >>> tensor = ip(pil_image)              # direct call → torch.Tensor [3,224,224]
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224)) -> None:
        self.target_size = target_size
        self.transforms = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        # Pre-build augmentation pipeline once (avoid re-creating on every call)
        self._augment_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Apply the full transform pipeline to a PIL Image."""
        return self.transforms(image)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        Make the preprocessor callable – required for use as a PyTorch
        ``transform`` argument (``DataLoader`` calls ``transform(sample)``).
        """
        return self.preprocess(image)

    def batch_preprocess(self, images: List[Image.Image]) -> torch.Tensor:
        """Process a list of PIL images and stack into a batch tensor ``[N, 3, H, W]``."""
        return torch.stack([self.preprocess(img) for img in images])

    def augment(self, image: Image.Image) -> Image.Image:
        """Apply data augmentation (training only – returns PIL Image)."""
        return self._augment_transforms(image)

    # ------------------------------------------------------------------
    # Config helper (used by /preprocess API endpoint)
    # ------------------------------------------------------------------

    def get_default_config(self) -> Dict[str, Any]:
        h, w = self.target_size
        return {
            "target_size": list(self.target_size),
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "output_shape": f"[3, {h}, {w}]",
        }
