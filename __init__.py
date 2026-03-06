"""Root package initialization for APEX framework."""

__version__ = "0.1.0"
__author__ = "Abhiram"

from .modelss import MultimodalPredictor

def create_app():
    """Factory function to create FastAPI application."""
    from .run_api import app
    return app

__all__ = ["create_app", "MultimodalPredictor"]

