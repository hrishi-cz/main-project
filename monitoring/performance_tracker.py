"""Performance tracking for model monitoring."""

from typing import Dict, List, Optional
from datetime import datetime
import numpy as np


class PerformanceTracker:
    """Track model performance over time."""
    
    def __init__(self):
        self.metrics = {}
        self.history = []
    
    def log_prediction(self, prediction: np.ndarray, actual: np.ndarray):
        """Log a prediction with actual value."""
        timestamp = datetime.now().isoformat()
        
        # Calculate metrics
        if len(actual.shape) == 1:  # Regression
            mse = np.mean((prediction - actual) ** 2)
            mae = np.mean(np.abs(prediction - actual))
            metrics = {"mse": float(mse), "mae": float(mae)}
        else:  # Classification
            accuracy = np.mean(prediction == actual)
            metrics = {"accuracy": float(accuracy)}
        
        entry = {
            "timestamp": timestamp,
            "metrics": metrics,
        }
        self.history.append(entry)
    
    def get_performance_summary(self) -> Dict:
        """Get summary of performance metrics."""
        if not self.history:
            return {}
        
        # Calculate averages
        all_metrics = [h["metrics"] for h in self.history]
        summary = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        return summary
