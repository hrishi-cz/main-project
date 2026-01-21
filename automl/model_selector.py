"""AutoML model selector for automatic model selection."""

import torch
import torch.nn as nn
from typing import List, Dict, Optional


class ModelSelector:
    """Automatically selects the best model based on task and data characteristics."""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
    
    def select_model(self, task: str, data_shape: Dict) -> str:
        """Select appropriate model based on task type."""
        if task == "classification":
            return "multimodal_classifier"
        elif task == "regression":
            return "multimodal_regressor"
        else:
            return "multimodal_predictor"
    
    def register_model(self, name: str, model: nn.Module):
        """Register a model in the selector."""
        self.models[name] = model
    
    def get_model(self, name: str) -> Optional[nn.Module]:
        """Retrieve registered model."""
        return self.models.get(name)
