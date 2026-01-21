"""Model registry for managing trained models."""

import os
import json
from typing import Dict, Optional, List
from datetime import datetime
import torch


class ModelRegistry:
    """Registry for managing trained models."""
    
    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = registry_path
        self.metadata = {}
        os.makedirs(registry_path, exist_ok=True)
    
    def register_model(
        self,
        model_name: str,
        model: torch.nn.Module,
        metadata: Dict,
    ) -> str:
        """Register a trained model."""
        timestamp = datetime.now().isoformat()
        model_id = f"{model_name}_{timestamp}"
        
        # Save model
        model_path = os.path.join(self.registry_path, f"{model_id}.pth")
        torch.save(model.state_dict(), model_path)
        
        # Save metadata
        meta = {
            "model_id": model_id,
            "model_name": model_name,
            "timestamp": timestamp,
            "metadata": metadata,
        }
        self.metadata[model_id] = meta
        
        return model_id
    
    def load_model(self, model_id: str) -> Optional[torch.nn.Module]:
        """Load model by ID."""
        model_path = os.path.join(self.registry_path, f"{model_id}.pth")
        if os.path.exists(model_path):
            return torch.load(model_path)
        return None
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self.metadata.keys())
