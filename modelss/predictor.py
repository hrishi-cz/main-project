"""
models/predictor.py

Multimodal predictor combining multiple encoders and fusion.
"""

from typing import List, Dict, Optional
import torch
import torch.nn as nn
from .encoders.image import ImageEncoder
from .encoders.tabular import TabularEncoder
from .encoders.text import TextEncoder
from .fusion import ConcatenationFusion, AttentionFusion


class MultimodalPredictor(nn.Module):
    """Multimodal predictor combining image, tabular, and text data."""
    
    def __init__(
        self,
        image_encoder: Optional[ImageEncoder] = None,
        tabular_encoder: Optional[TabularEncoder] = None,
        text_encoder: Optional[TextEncoder] = None,
        fusion_strategy: str = "concatenation",
        num_classes: int = 1,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.tabular_encoder = tabular_encoder
        self.text_encoder = text_encoder
        
        # Determine which modalities are available
        feature_dims = []
        if image_encoder is not None:
            feature_dims.append(image_encoder.get_output_dim())
        if tabular_encoder is not None:
            feature_dims.append(tabular_encoder.get_output_dim())
        if text_encoder is not None:
            feature_dims.append(text_encoder.get_output_dim())
        
        # Create fusion layer
        if fusion_strategy == "concatenation":
            self.fusion = ConcatenationFusion(feature_dims)
        elif fusion_strategy == "attention":
            self.fusion = AttentionFusion(feature_dims)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
        # Prediction head
        fusion_dim = self.fusion.get_output_dim()
        self.prediction_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        tabular: Optional[torch.Tensor] = None,
        text: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        features = []
        
        if image is not None and self.image_encoder is not None:
            img_features = self.image_encoder(image)
            features.append(img_features)
        
        if tabular is not None and self.tabular_encoder is not None:
            tab_features = self.tabular_encoder(tabular)
            features.append(tab_features)
        
        if text is not None and self.text_encoder is not None:
            text_features = self.text_encoder(text)
            features.append(text_features)
        
        # Fuse features
        fused = self.fusion(features)
        
        # Predict
        output = self.prediction_head(fused)
        return output
