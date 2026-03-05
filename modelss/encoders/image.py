"""
modelss/encoders/image.py

ResNet-50 image encoder with a fixed 2048 → 512 projection head.

Architecture
------------
  torchvision.models.resnet50(pretrained=True)
      ↓  (fc layer replaced with nn.Identity())
  [N, 2048]  global-average-pooled feature map
      ↓
  nn.Linear(2048, 512) → nn.ReLU()
      ↓
  [N, 512]  projected image features

The ``fc`` layer is replaced by ``nn.Identity()`` at construction time so
the backbone's forward pass outputs the raw 2048-dim GAP vector.  The
``projection`` head then maps this to the fixed 512-dim output required by
the fusion layer.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torchvision.models as tv_models

logger = logging.getLogger(__name__)

# Architectural constants (must not be changed – fusion layer depends on these)
IMAGE_BACKBONE_DIM: int = 2048
IMAGE_OUTPUT_DIM: int   = 512


class ImageEncoder(nn.Module):
    """
    ResNet-50 backbone with a fixed 2048 → 512 projection head.

    Parameters
    ----------
    pretrained : bool
        Load ImageNet-1k weights via ``torchvision.models.ResNet50_Weights``.
        Default ``True``.
    freeze_backbone : bool
        Freeze all ResNet-50 convolutional parameters so only the projection
        head is trained.  Useful for small datasets.  Default ``False``.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        # ── ResNet-50 backbone ────────────────────────────────────────────
        # Use the new weights API (torchvision ≥ 0.13); fall back gracefully
        # for older verions that still accept the bool ``pretrained`` kwarg.
        try:
            weights = (
                tv_models.ResNet50_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )
            backbone: nn.Module = tv_models.resnet50(weights=weights)
        except TypeError:
            # torchvision < 0.13
            backbone = tv_models.resnet50(pretrained=pretrained)  # type: ignore[call-arg]

        # Strip the classification head: replace fc with Identity so the
        # backbone forward pass returns [N, 2048] (GAP output).
        backbone.fc = nn.Identity()
        self.backbone: nn.Module = backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ── Projection head: 2048 → 512 ──────────────────────────────────
        self.projection: nn.Sequential = nn.Sequential(
            nn.Linear(IMAGE_BACKBONE_DIM, IMAGE_OUTPUT_DIM),
            nn.ReLU(),
        )

        logger.info(
            "ImageEncoder: backbone=resnet50  pretrained=%s  "
            "freeze_backbone=%s  output_dim=%d",
            pretrained, freeze_backbone, IMAGE_OUTPUT_DIM,
        )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract and project image features.

        Parameters
        ----------
        x : torch.Tensor
            Float image batch of shape ``(N, 3, H, W)``.  Values should be
            ImageNet-normalised (mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]) and spatially resized to ≥ 32 × 32.

        Returns
        -------
        torch.Tensor
            Shape ``(N, 512)`` — projected, ReLU-activated feature vectors.
        """
        gap_features: torch.Tensor = self.backbone(x)   # (N, 2048)
        return self.projection(gap_features)             # (N, 512)

    def get_output_dim(self) -> int:
        """Return the fixed output dimensionality (512)."""
        return IMAGE_OUTPUT_DIM
