"""
models/encoders/text.py

Text encoders using transformers.
"""

from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class TextEncoder(nn.Module):
    """Text encoder using pretrained transformer models."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        
        # Load model and tokenizer
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.feature_dim = self.model.config.hidden_size
        
        # Optional projection layer
        if output_dim is not None:
            self.projection = nn.Linear(self.feature_dim, output_dim)
        else:
            self.projection = None
    
    def forward(self, texts: list, max_length: int = 512) -> torch.Tensor:
        """Forward pass - encode text."""
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**encoded)
            # Use [CLS] token embedding
            features = outputs.last_hidden_state[:, 0, :]
        
        if self.projection is not None:
            features = self.projection(features)
        
        return features
    
    def get_output_dim(self) -> int:
        """Get output feature dimension."""
        return self.output_dim if self.output_dim else self.feature_dim
