"""
research/experiment_collector.py

Collects experiment metadata from trained models in the registry.
Aggregates metrics, schema, hyperparameters, and XAI artifacts for paper generation.
"""

import os
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class ExperimentCollector:
    """
    Scans model registry and collects all experiment metadata.
    
    Usage:
        collector = ExperimentCollector(registry_dir="models")
        experiments = collector.collect()
        # Returns list of experiment dicts with metrics, modalities, fusion_type, xai, etc.
    """

    def __init__(self, registry_dir: str = "models"):
        """
        Parameters
        ----------
        registry_dir : str
            Path to model registry directory (e.g., "models/"). 
            Should contain subdirectories for each model with metadata.json.
        """
        self.registry_dir = registry_dir

    def collect(self) -> List[Dict[str, Any]]:
        """
        Scan registry and collect all experiment metadata.
        
        Returns
        -------
        List[Dict] with entries:
            {
                "model_id": "apex_v1_...",
                "metrics": {"accuracy": 0.85, "f1": 0.82, ...},
                "modalities": ["tabular", "image"],
                "target": "disease",
                "target_type": "binary",
                "fusion_type": "uncertainty_graph",
                "latency_ms": {"mean": 45.2, "p95": 120.5},
                "memory_mb": 2048,
                "xai": {...},
                "preprocessing_plan": {...},
                "hyperparameters": {...}
            }
        """
        experiments = []

        if not os.path.exists(self.registry_dir):
            logger.warning(f"Registry directory not found: {self.registry_dir}")
            return experiments

        # Iterate through model directories
        for model_id in os.listdir(self.registry_dir):
            model_path = os.path.join(self.registry_dir, model_id)
            meta_path = os.path.join(model_path, "metadata.json")

            if not os.path.isdir(model_path) or not os.path.exists(meta_path):
                continue

            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)

                # Standardize latency (could be dict or scalar)
                latency_val = meta.get("latency_ms", {})
                if isinstance(latency_val, dict):
                    latency_ms = latency_val
                else:
                    latency_ms = {"mean": float(latency_val) if latency_val else None}

                experiment = {
                    "model_id": model_id,
                    "metrics": meta.get("metrics", {}),
                    "modalities": meta.get("modalities", []),
                    "target": meta.get("target"),
                    "target_type": meta.get("problem_type"),
                    "fusion_type": meta.get("fusion_type", "concatenation"),
                    "latency_ms": latency_ms,
                    "memory_mb": meta.get("memory_mb", None),
                    "xai": meta.get("xai", {}),
                    "preprocessing_plan": meta.get("preprocessing_plan", {}),
                    "hyperparameters": meta.get("hyperparameters", {}),
                    "schema": meta.get("schema", {}),
                }

                experiments.append(experiment)
                logger.info(f"  ✓ Collected {model_id}")

            except Exception as e:
                logger.warning(f"  Failed to read {model_id}: {e}")

        logger.info(f"Total experiments collected: {len(experiments)}")
        return experiments

    def get_best_experiment(self, experiments: List[Dict], metric: str = "accuracy") -> Dict[str, Any]:
        """
        Find the best-performing experiment by a given metric.
        
        Parameters
        ----------
        experiments : List[Dict]
            Output from collect().
        metric : str
            Metric key to optimize (default "accuracy").
        
        Returns
        -------
        Dict : Best experiment or empty dict if none found.
        """
        valid = [e for e in experiments if metric in e.get("metrics", {})]
        if not valid:
            return {}
        return max(valid, key=lambda e: e["metrics"][metric])

    def get_experiments_by_modality(
        self, experiments: List[Dict], modality: str
    ) -> List[Dict[str, Any]]:
        """
        Filter experiments that use a specific modality.
        """
        return [e for e in experiments if modality in e.get("modalities", [])]
