"""AutoML package for automatic model selection and training."""

from .model_selector import ModelSelector
from .trainer import Trainer

__all__ = ["ModelSelector", "Trainer"]
