"""Hyperparameter configuration for models."""


import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
try:
    import yaml
except ImportError:
    yaml = None



@dataclass
class HyperparameterConfig:
    """
    Configuration for model hyperparameters. Supports dynamic loading, merging, and exporting.
    """
    image_encoder_name: str = "resnet50"
    image_output_dim: int = 256
    tabular_hidden_dims: Optional[List[int]] = None
    tabular_output_dim: int = 128
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_output_dim: int = 256
    fusion_strategy: str = "attention"
    fusion_output_dim: int = 256
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 10
    dropout: float = 0.2

    def __post_init__(self):
        if self.tabular_hidden_dims is None:
            self.tabular_hidden_dims = [256, 128]

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'HyperparameterConfig':
        """Create config from dict, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str) -> 'HyperparameterConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, path: str) -> 'HyperparameterConfig':
        if not yaml:
            raise ImportError("pyyaml is required for YAML config support.")
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def save_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_yaml(self, path: str):
        if not yaml:
            raise ImportError("pyyaml is required for YAML config support.")
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f)

    def merge(self, override: Dict[str, Any]):
        """Merge override dict into config."""
        for k, v in override.items():
            if hasattr(self, k):
                setattr(self, k, v)


def load_dynamic_config(config_path: Optional[str] = None) -> HyperparameterConfig:
    """
    Load config from JSON/YAML file if provided, else use defaults.
    """
    if config_path:
        ext = Path(config_path).suffix.lower()
        if ext in ['.yaml', '.yml']:
            return HyperparameterConfig.from_yaml(config_path)
        elif ext == '.json':
            return HyperparameterConfig.from_json(config_path)
        else:
            raise ValueError(f"Unsupported config file type: {ext}")
    return HyperparameterConfig()

# Global hyperparameters schema
HYPERPARAMETERS = {
    "image_encoder_name": {
        "type": "string",
        "default": "resnet50",
        "options": ["resnet50", "mobilenet_v3_small", "efficientnet_b0", "vit_base_patch16_224"]
    },
    "image_output_dim": {
        "type": "integer",
        "default": 256,
        "min": 64,
        "max": 2048
    },
    "tabular_output_dim": {
        "type": "integer",
        "default": 128,
        "min": 32,
        "max": 512
    },
    "text_model_name": {
        "type": "string",
        "default": "sentence-transformers/all-MiniLM-L6-v2",
        "options": ["distilbert-base-uncased", "roberta-base", "sentence-transformers/all-MiniLM-L6-v2"]
    },
    "text_output_dim": {
        "type": "integer",
        "default": 256,
        "min": 64,
        "max": 1024
    },
    "fusion_strategy": {
        "type": "string",
        "default": "attention",
        "options": ["concatenation", "attention", "weighted"]
    },
    "learning_rate": {
        "type": "float",
        "default": 1e-3,
        "min": 1e-6,
        "max": 1e-1
    },
    "batch_size": {
        "type": "integer",
        "default": 32,
        "options": [8, 16, 32, 64, 128]
    },
    "num_epochs": {
        "type": "integer",
        "default": 10,
        "min": 1,
        "max": 100
    },
    "dropout": {
        "type": "float",
        "default": 0.2,
        "min": 0.0,
        "max": 0.9
    }
}

def get_presets() -> Dict[str, Dict]:
    """Return all available preset configurations."""
    return PRESETS.copy()

# Preset configurations
PRESETS = {
    "small": {
        "image_encoder_name": "mobilenet_v3_small",
        "image_output_dim": 128,
        "tabular_output_dim": 64,
        "text_model_name": "distilbert-base-uncased",
        "text_output_dim": 128,
        "fusion_strategy": "concatenation",
        "learning_rate": 5e-4,
        "batch_size": 16,
        "num_epochs": 5,
        "dropout": 0.1,
    },
    "medium": {
        "image_encoder_name": "resnet50",
        "image_output_dim": 256,
        "tabular_output_dim": 128,
        "text_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "text_output_dim": 256,
        "fusion_strategy": "attention",
        "learning_rate": 1e-3,
        "batch_size": 32,
        "num_epochs": 10,
        "dropout": 0.2,
    },
    "large": {
        "image_encoder_name": "resnet50",
        "image_output_dim": 512,
        "tabular_output_dim": 256,
        "text_model_name": "roberta-base",
        "text_output_dim": 512,
        "fusion_strategy": "attention",
        "learning_rate": 1e-3,
        "batch_size": 64,
        "num_epochs": 15,
        "dropout": 0.3,
    },
    "fast": {
        "image_encoder_name": "mobilenet_v3_small",
        "image_output_dim": 128,
        "tabular_output_dim": 64,
        "text_model_name": "distilbert-base-uncased",
        "text_output_dim": 128,
        "fusion_strategy": "concatenation",
        "learning_rate": 1e-2,
        "batch_size": 128,
        "num_epochs": 3,
        "dropout": 0.1,
    }
}



def validate_hyperparameters(params: Dict) -> Dict:
    """
    Validate and normalize hyperparameters against schema.
    Unknown keys are passed through. Out-of-range values are clipped.
    """
    validated = {}
    for key, value in params.items():
        if key not in HYPERPARAMETERS:
            validated[key] = value
            continue
        schema = HYPERPARAMETERS[key]
        param_type = schema.get("type", "string")
        try:
            if param_type == "integer":
                validated[key] = int(value)
                if "min" in schema and validated[key] < schema["min"]:
                    validated[key] = schema["min"]
                if "max" in schema and validated[key] > schema["max"]:
                    validated[key] = schema["max"]
            elif param_type == "float":
                validated[key] = float(value)
                if "min" in schema and validated[key] < schema["min"]:
                    validated[key] = schema["min"]
                if "max" in schema and validated[key] > schema["max"]:
                    validated[key] = schema["max"]
            else:
                validated[key] = str(value)
                if "options" in schema and validated[key] not in schema["options"]:
                    validated[key] = schema.get("default", value)
        except Exception:
            validated[key] = schema.get("default", value)
    return validated



def get_preset(preset_name: str) -> Dict:
    """Get a preset configuration by name."""
    presets = get_presets()
    if preset_name not in presets:
        raise ValueError(f"Preset {preset_name} not found. Available: {list(presets.keys())}")
    return presets[preset_name].copy()



def get_default_config() -> HyperparameterConfig:
    """Get default configuration as HyperparameterConfig object."""
    return HyperparameterConfig()

