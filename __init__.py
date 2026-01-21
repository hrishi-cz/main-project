"""Root package initialization for APEX framework."""

__version__ = "0.1.0"
__author__ = "Abhiram"

from .api.main_enhanced import create_app
from .modelss import MultimodalPredictor

__all__ = ["create_app", "MultimodalPredictor"]
