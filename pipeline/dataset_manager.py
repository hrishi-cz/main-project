"""Dataset manager for pipeline operations."""

import pandas as pd
from typing import Dict, List, Optional


class DatasetManager:
    """Manage datasets throughout the pipeline."""
    
    def __init__(self):
        self.datasets = {}
        self.metadata = {}
    
    def register_dataset(self, name: str, data: pd.DataFrame, metadata: Dict = None):
        """Register a dataset."""
        self.datasets[name] = data
        self.metadata[name] = metadata or {}
    
    def get_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """Get dataset by name."""
        return self.datasets.get(name)
    
    def split_dataset(self, name: str, train_ratio: float = 0.8):
        """Split dataset into train and test."""
        data = self.datasets.get(name)
        if data is None:
            return None
        
        split_idx = int(len(data) * train_ratio)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        return {
            "train": train_data,
            "test": test_data,
        }
    
    def list_datasets(self) -> List[str]:
        """List all registered datasets."""
        return list(self.datasets.keys())
