"""
models/encoders/tabular.py

Tabular data encoders using MLP.
"""

import torch
import torch.nn as nn


class TabularEncoder(nn.Module):
    """MLP encoder for tabular data."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        output_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)
    
    def get_output_dim(self) -> int:
        """Get output feature dimension."""
        return self.output_dim
