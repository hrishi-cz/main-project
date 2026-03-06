"""Model registry for managing trained models."""

import os
import json
from typing import Dict, Optional, List
from datetime import datetime
import torch


class ModelRegistry:
    """Registry for managing trained models."""
    
    _instance = None
    
    def __new__(cls, registry_path: str = "./model_registry"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, registry_path: str = "./model_registry"):
        if self._initialized:
            return
        self.registry_path = registry_path
        self.metadata = {}
        os.makedirs(registry_path, exist_ok=True)
        self._initialized = True
        self._load_metadata()
    
    def _load_metadata(self):
        """Load metadata from registry."""
        meta_file = os.path.join(self.registry_path, "metadata.json")
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                self.metadata = json.load(f)
    
    def _save_metadata(self):
        """Save metadata to registry."""
        meta_file = os.path.join(self.registry_path, "metadata.json")
        with open(meta_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def register_model(
        self,
        model_name: str,
        model: torch.nn.Module,
        metadata: Dict,
    ) -> str:
        """Register a trained model."""
        timestamp = datetime.now().isoformat()
        safe_timestamp = timestamp.replace(":", "-")
        model_id = f"{model_name}_{safe_timestamp}"
        
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
        self._save_metadata()
        
        return model_id
    
    def load_model(self, model_id: str) -> Optional[Dict]:
        """Load model state_dict by ID. Returns the state_dict (OrderedDict), not an nn.Module."""
        model_path = os.path.join(self.registry_path, f"{model_id}.pth")
        if os.path.exists(model_path):
            return torch.load(model_path, weights_only=True)
        return None
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self.metadata.keys())
    
    def get_model_info(self, model_id: str) -> Dict:
        """Get model metadata by ID."""
        if model_id not in self.metadata:
            raise FileNotFoundError(f"Model {model_id} not found in registry")
        return self.metadata[model_id]
    
    def unregister_model(self, model_id: str) -> bool:
        """Unregister (delete) a model."""
        if model_id not in self.metadata:
            return False
        
        model_path = os.path.join(self.registry_path, f"{model_id}.pth")
        if os.path.exists(model_path):
            os.remove(model_path)
        
        del self.metadata[model_id]
        self._save_metadata()
        return True
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        return cls()

