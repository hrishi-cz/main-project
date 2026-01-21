"""Data loader for loading various data formats."""

import pandas as pd
from typing import Dict, List, Optional, Union
import numpy as np


class DataLoader:
    """Universal data loader for multimodal datasets."""
    
    def __init__(self):
        self.data = None
        self.metadata = {}
    
    def load_csv(self, filepath: str, sep: str = ",") -> pd.DataFrame:
        """Load CSV file."""
        return pd.read_csv(filepath, sep=sep)
    
    def load_parquet(self, filepath: str) -> pd.DataFrame:
        """Load Parquet file."""
        return pd.read_parquet(filepath)
    
    def load_json(self, filepath: str) -> Dict:
        """Load JSON file."""
        import json
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def load_images(self, image_paths: List[str]) -> List:
        """Load images from file paths."""
        from PIL import Image
        images = []
        for path in image_paths:
            try:
                img = Image.open(path)
                images.append(img)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
        return images
    
    def merge_datasets(self, datasets: Dict[str, Union[pd.DataFrame, List]]) -> Dict:
        """Merge multiple datasets."""
        return datasets
