"""Tabular preprocessing for data preparation."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Optional


class TabularPreprocessor:
    """Preprocessor for tabular data."""
    
    def __init__(self, scaling: str = "standard"):
        self.scaling = scaling
        if scaling == "standard":
            self.scaler = StandardScaler()
        elif scaling == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess tabular data."""
        # Handle missing values
        data = data.fillna(data.mean(numeric_only=True))
        
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Scale features
        if self.scaler is not None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = self.scaler.fit_transform(data[numeric_cols])
        
        return data
    
    def handle_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle categorical variables."""
        categorical_cols = data.select_dtypes(include=['object']).columns
        data = pd.get_dummies(data, columns=categorical_cols)
        return data
    
    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = (data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std()
        return data
