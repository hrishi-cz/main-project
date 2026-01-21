"""
models/encoders/image.py

Image encoders using timm pretrained models.
"""

from typing import Optional
import torch
import torch.nn as nn
import timm


class ImageEncoder(nn.Module):
    """Pre-trained image encoder using timm models."""
    
    def __init__(
        self,
        encoder_name: str = "resnet50",
        pretrained: bool = True,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.pretrained = pretrained
        self.output_dim = output_dim
        
        # Load pretrained model
        self.backbone = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )
        
        self.feature_dim = self.backbone.num_features
        
        # Optional projection layer
        if output_dim is not None:
            self.projection = nn.Linear(self.feature_dim, output_dim)
        else:
            self.projection = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x)
        if self.projection is not None:
            features = self.projection(features)
        return features
    
    def get_output_dim(self) -> int:
        """Get output feature dimension."""
        return self.output_dim if self.output_dim else self.feature_dim
