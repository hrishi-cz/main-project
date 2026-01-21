"""Hyperparameter configuration for models."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class HyperparameterConfig:
    """Configuration for model hyperparameters."""
    
    # Image encoder
    image_encoder_name: str = "resnet50"
    image_output_dim: int = 256
    
    # Tabular encoder
    tabular_hidden_dims: List[int] = None
    tabular_output_dim: int = 128
    
    # Text encoder
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_output_dim: int = 256
    
    # Fusion
    fusion_strategy: str = "attention"
    fusion_output_dim: int = 256
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 10
    dropout: float = 0.2
    
    def __post_init__(self):
        if self.tabular_hidden_dims is None:
            self.tabular_hidden_dims = [256, 128]
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            "image_encoder_name": self.image_encoder_name,
            "image_output_dim": self.image_output_dim,
            "tabular_hidden_dims": self.tabular_hidden_dims,
            "tabular_output_dim": self.tabular_output_dim,
            "text_model_name": self.text_model_name,
            "text_output_dim": self.text_output_dim,
            "fusion_strategy": self.fusion_strategy,
            "fusion_output_dim": self.fusion_output_dim,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "dropout": self.dropout,
        }
