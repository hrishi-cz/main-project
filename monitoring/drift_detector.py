"""Data drift detection for monitoring model degradation."""

from typing import Dict, List, Optional, Union
import numpy as np


class DriftDetector:
    """Detect data drift in model inputs/outputs."""
    
    def __init__(self, threshold: float = 0.1, baseline_data: Optional[Union[np.ndarray, any]] = None):
        self.threshold = threshold
        self.baseline_stats = {}
        self.drift_history = []
        
        if baseline_data is not None:
            self.set_baseline(baseline_data)
    
    def set_baseline(self, data: Union[np.ndarray, any], feature_names: List[str] = None):
        """Set baseline statistics from initial data."""
        # Handle DataFrame input
        if hasattr(data, 'values'):  # pandas DataFrame
            data_array = data.values
            if feature_names is None:
                feature_names = list(data.columns)
        else:
            data_array = np.array(data) if not isinstance(data, np.ndarray) else data
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(data_array.shape[1])]
        
        for i, name in enumerate(feature_names):
            if i < data_array.shape[1]:
                col_data = data_array[:, i] if len(data_array.shape) > 1 else data_array
                self.baseline_stats[name] = {
                    "mean": float(np.mean(col_data)),
                    "std": float(np.std(col_data)),
                }
    
    def detect_drift(self, data: Union[np.ndarray, any]) -> Dict:
        """Detect drift in new data."""
        if not self.baseline_stats:
            return {
                "overall_drift_detected": False,
                "drift_detected": False,
                "drifted_features": []
            }
        
        # Handle DataFrame input
        if hasattr(data, 'values'):  # pandas DataFrame
            data_array = data.values
        else:
            data_array = np.array(data) if not isinstance(data, np.ndarray) else data
        
        drift_detected = False
        drifted_features = []
        
        feature_names = list(self.baseline_stats.keys())
        for i, name in enumerate(feature_names):
            if i >= data_array.shape[1]:
                break
                
            baseline = self.baseline_stats[name]
            col_data = data_array[:, i] if len(data_array.shape) > 1 else data_array
            current_mean = float(np.mean(col_data))
            
            # Simple drift detection using z-score
            z_score = abs((current_mean - baseline["mean"]) / (baseline["std"] + 1e-8))
            if z_score > self.threshold:
                drift_detected = True
                drifted_features.append(name)
        
        result = {
            "drift_detected": drift_detected,
            "overall_drift_detected": drift_detected,
            "drifted_features": drifted_features,
        }
        self.drift_history.append(result)
        return result

