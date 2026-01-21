"""Data drift detection for monitoring model degradation."""

from typing import Dict, List, Optional
import numpy as np


class DriftDetector:
    """Detect data drift in model inputs/outputs."""
    
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.baseline_stats = {}
        self.drift_history = []
    
    def set_baseline(self, data: np.ndarray, feature_names: List[str] = None):
        """Set baseline statistics from initial data."""
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(data.shape[1])]
        
        for i, name in enumerate(feature_names):
            self.baseline_stats[name] = {
                "mean": float(np.mean(data[:, i])),
                "std": float(np.std(data[:, i])),
            }
    
    def detect_drift(self, data: np.ndarray) -> Dict:
        """Detect drift in new data."""
        if not self.baseline_stats:
            return {}
        
        drift_detected = False
        drifted_features = []
        
        feature_names = list(self.baseline_stats.keys())
        for i, name in enumerate(feature_names):
            baseline = self.baseline_stats[name]
            current_mean = float(np.mean(data[:, i]))
            
            # Simple drift detection using z-score
            z_score = abs((current_mean - baseline["mean"]) / (baseline["std"] + 1e-8))
            if z_score > self.threshold:
                drift_detected = True
                drifted_features.append(name)
        
        return {
            "drift_detected": drift_detected,
            "drifted_features": drifted_features,
        }
