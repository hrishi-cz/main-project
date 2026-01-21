"""
models/fusion.py

Fusion strategies for multimodal features.
"""

from typing import List
import torch
import torch.nn as nn


class ConcatenationFusion(nn.Module):
    """Simple concatenation fusion."""
    
    def __init__(self, feature_dims: List[int]):
        super().__init__()
        self.feature_dims = feature_dims
        self.output_dim = sum(feature_dims)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate features."""
        return torch.cat(features, dim=1)
    
    def get_output_dim(self) -> int:
        return self.output_dim


class AttentionFusion(nn.Module):
    """Attention-based fusion of multimodal features."""
    
    def __init__(self, feature_dims: List[int], output_dim: int = 256):
        super().__init__()
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        
        # Project all modalities to same dimension
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in feature_dims
        ])
        
        # Attention weights
        self.attention = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.Tanh(),
            nn.Linear(output_dim // 2, 1),
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Apply attention fusion."""
        # Project features
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]
        projected = torch.stack(projected, dim=1)  # [batch, n_modalities, output_dim]
        
        # Compute attention weights
        attention_weights = self.attention(projected)  # [batch, n_modalities, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention
        fused = (projected * attention_weights).sum(dim=1)
        return fused
    
    def get_output_dim(self) -> int:
        return self.output_dim
