"""Performance tracking for model monitoring."""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np


class PerformanceTracker:
    """Track model performance over time."""
    
    _instances = {}
    
    def __new__(cls, model_id: str = "default"):
        if model_id not in cls._instances:
            cls._instances[model_id] = super().__new__(cls)
            cls._instances[model_id]._initialized = False
        return cls._instances[model_id]
    
    def __init__(self, model_id: str = "default"):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self.model_id = model_id
        self.metrics = {}
        self.history = []
        self._initialized = True
    
    def log_prediction(self, prediction: np.ndarray, actual: np.ndarray, timestamp: Optional[str] = None):
        """Log a prediction with actual value."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Calculate metrics
        if len(actual.shape) == 1:  # Regression or binary
            mse = np.mean((prediction - actual) ** 2)
            mae = np.mean(np.abs(prediction - actual))
            metrics = {"mse": float(mse), "mae": float(mae), "rmse": float(np.sqrt(mse))}
        else:  # Multi-class classification
            accuracy = np.mean(np.argmax(prediction, axis=1) == np.argmax(actual, axis=1))
            metrics = {"accuracy": float(accuracy)}
        
        entry = {
            "timestamp": timestamp,
            "metrics": metrics,
        }
        self.history.append(entry)
    
    def get_recent_metrics(self, limit: int = 20) -> List[Dict]:
        """Get recent performance metrics."""
        return self.history[-limit:] if self.history else []
    
    def get_metric_trend(self, metric_name: str, hours: int = 24) -> List[Dict]:
        """Get trend for a specific metric over time."""
        if not self.history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        trend = []
        
        for entry in self.history:
            entry_time = datetime.fromisoformat(entry["timestamp"])
            if entry_time >= cutoff_time and metric_name in entry.get("metrics", {}):
                trend.append({
                    "timestamp": entry["timestamp"],
                    "value": entry["metrics"][metric_name]
                })
        
        return trend
    
    def get_performance_summary(self) -> Dict:
        """Get summary of performance metrics."""
        if not self.history:
            return {}
        
        # Calculate averages
        all_metrics = [h["metrics"] for h in self.history]
        summary = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            if values:
                summary[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
        return summary
    
    def clear_history(self):
        """Clear all history."""
        self.history = []
        self.metrics = {}

