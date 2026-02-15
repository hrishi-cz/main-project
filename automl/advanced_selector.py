"""Advanced model selection based on GPU memory, dataset size, and task complexity."""

import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ModelSelectionResult:
    """Result of model selection."""
    image_encoder: Optional[str]
    text_encoder: Optional[str]
    tabular_encoder: Optional[str]
    fusion_strategy: str
    batch_size: int
    num_epochs: int
    learning_rate: float
    total_features_dim: int
    rationale: Dict[str, str]
    hardware_info: Dict


class AdvancedModelSelector:
    """Advanced model selection based on data and hardware characteristics."""
    
    def __init__(self):
        self.image_encoders = {
            "lightweight": {
                "name": "MobileNetV3",
                "output_dim": 1280,
                "params": "2.2M",
                "throughput": "high"
            },
            "balanced": {
                "name": "ResNet50",
                "output_dim": 2048,
                "params": "25M",
                "throughput": "medium"
            },
            "sota": {
                "name": "ViT-Base",
                "output_dim": 768,
                "params": "86M",
                "throughput": "low"
            }
        }
        
        self.text_encoders = {
            "fast": {
                "name": "DistilBERT",
                "output_dim": 768,
                "params": "66M",
                "speed": "fast"
            },
            "balanced": {
                "name": "BERT-base",
                "output_dim": 768,
                "params": "110M",
                "speed": "medium"
            },
            "sota": {
                "name": "RoBERTa-large",
                "output_dim": 1024,
                "params": "355M",
                "speed": "slow"
            }
        }
        
        self.tabular_encoders = {
            "simple": {
                "name": "MLP",
                "output_dim": 128,
                "params": "small",
                "interpretable": "yes"
            },
            "interpretable": {
                "name": "TabNet",
                "output_dim": 128,
                "params": "medium",
                "interpretable": "yes"
            },
            "sota": {
                "name": "FT-Transformer",
                "output_dim": 256,
                "params": "large",
                "interpretable": "no"
            }
        }
    
    def select_models(
        self,
        schema_result,
        dataset_size: int,
        progress_callback=None
    ) -> ModelSelectionResult:
        """
        Select appropriate models based on schema and dataset characteristics.
        
        Args:
            schema_result: SchemaDetectionResult from schema detector
            dataset_size: Number of samples in dataset
            progress_callback: Progress callback
        
        Returns:
            ModelSelectionResult with selected models and hyperparameters
        """
        if progress_callback:
            progress_callback(10, "Analyzing hardware capabilities...")
        
        # Analyze hardware
        gpu_memory = self._get_gpu_memory()
        hardware_info = {
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory_gb": gpu_memory,
            "device": "GPU" if torch.cuda.is_available() else "CPU",
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
        }
        
        if progress_callback:
            progress_callback(20, f"GPU Memory: {gpu_memory}GB | Dataset Size: {dataset_size}")
        
        # Select image encoder
        if progress_callback:
            progress_callback(30, "Selecting image encoder...")
        
        image_encoder = None
        image_rationale = ""
        if schema_result.image_cols:
            image_encoder, image_rationale = self._select_image_encoder(
                dataset_size, gpu_memory
            )
        
        # Select text encoder
        if progress_callback:
            progress_callback(40, "Selecting text encoder...")
        
        text_encoder = None
        text_rationale = ""
        if schema_result.text_cols:
            text_encoder, text_rationale = self._select_text_encoder(
                schema_result.problem_type, dataset_size, gpu_memory
            )
        
        # Select tabular encoder
        if progress_callback:
            progress_callback(50, "Selecting tabular encoder...")
        
        tabular_encoder = None
        tabular_rationale = ""
        if schema_result.tabular_cols:
            tabular_encoder, tabular_rationale = self._select_tabular_encoder(
                len(schema_result.tabular_cols), dataset_size, gpu_memory
            )
        
        # Select training hyperparameters
        if progress_callback:
            progress_callback(60, "Computing hyperparameters...")
        
        batch_size, epochs = self._compute_batch_size_and_epochs(
            dataset_size, gpu_memory, schema_result.modalities
        )
        
        learning_rate = self._compute_learning_rate(
            schema_result.problem_type, dataset_size
        )
        
        # Select fusion strategy
        if progress_callback:
            progress_callback(70, "Selecting fusion strategy...")
        
        fusion_strategy = self._select_fusion_strategy(
            schema_result.modalities, gpu_memory
        )
        
        # Calculate total feature dimension
        total_features_dim = 0
        if image_encoder:
            total_features_dim += self.image_encoders[image_encoder]["output_dim"]
        if text_encoder:
            total_features_dim += self.text_encoders[text_encoder]["output_dim"]
        if tabular_encoder:
            total_features_dim += self.tabular_encoders[tabular_encoder]["output_dim"]
        
        if progress_callback:
            progress_callback(100, "Model selection complete!")
        
        return ModelSelectionResult(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            tabular_encoder=tabular_encoder,
            fusion_strategy=fusion_strategy,
            batch_size=batch_size,
            num_epochs=epochs,
            learning_rate=learning_rate,
            total_features_dim=total_features_dim,
            rationale={
                "image": image_rationale,
                "text": text_rationale,
                "tabular": tabular_rationale,
                "batch_size": f"Dataset: {dataset_size} | GPU Memory: {gpu_memory}GB",
                "epochs": f"Larger dataset ({dataset_size}) requires fewer epochs",
                "learning_rate": f"Adaptive to problem type ({schema_result.problem_type}) and dataset size"
            },
            hardware_info=hardware_info
        )
    
    def _get_gpu_memory(self) -> float:
        """Get available GPU memory in GB."""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return round(total, 2)
        except:
            return 0.0
    
    def _select_image_encoder(self, dataset_size: int, gpu_memory: float) -> Tuple[str, str]:
        """Select image encoder based on dataset size and GPU memory."""
        if dataset_size < 1000:
            return "lightweight", f"Small dataset (<1k samples) requires lightweight model for efficient training"
        elif dataset_size < 10000:
            if gpu_memory >= 8:
                return "balanced", f"Medium dataset (1-10k) with sufficient GPU memory ({gpu_memory}GB)"
            else:
                return "lightweight", f"Medium dataset but limited GPU memory ({gpu_memory}GB)"
        else:
            if gpu_memory >= 12:
                return "sota", f"Large dataset (>10k) with excellent GPU memory ({gpu_memory}GB)"
            elif gpu_memory >= 8:
                return "balanced", f"Large dataset with moderate GPU memory ({gpu_memory}GB)"
            else:
                return "lightweight", f"Large dataset but limited GPU memory ({gpu_memory}GB)"
    
    def _select_text_encoder(
        self,
        problem_type: str,
        dataset_size: int,
        gpu_memory: float
    ) -> Tuple[str, str]:
        """Select text encoder based on task complexity."""
        if "binary" in problem_type or dataset_size < 5000:
            return "fast", "Binary/small task - DistilBERT for speed optimization"
        elif "multiclass" in problem_type:
            if gpu_memory >= 8 and dataset_size > 10000:
                return "sota", f"Multiclass task with sufficient resources (GPU: {gpu_memory}GB, Dataset: {dataset_size})"
            else:
                return "balanced", "Multiclass task - BERT for balance"
        else:
            return "balanced", "Default text encoder selection"
    
    def _select_tabular_encoder(
        self,
        num_tabular_cols: int,
        dataset_size: int,
        gpu_memory: float
    ) -> Tuple[str, str]:
        """Select tabular encoder based on feature complexity."""
        if num_tabular_cols < 10 and dataset_size < 5000:
            return "simple", f"Simple tabular ({num_tabular_cols} features) - MLP encoder"
        elif num_tabular_cols < 50:
            if gpu_memory >= 8:
                return "interpretable", f"Medium complexity ({num_tabular_cols} features) - TabNet for interpretability"
            else:
                return "simple", f"Limited GPU ({gpu_memory}GB) - MLP encoder"
        else:
            if gpu_memory >= 12:
                return "sota", f"Complex tabular ({num_tabular_cols} features) with good GPU - FT-Transformer"
            else:
                return "interpretable", f"Complex tabular but limited GPU - TabNet"
    
    def _compute_batch_size_and_epochs(
        self,
        dataset_size: int,
        gpu_memory: float,
        modalities: List[str]
    ) -> Tuple[int, int]:
        """Compute batch size and epochs based on dataset and hardware."""
        # Determine base batch size
        has_images = "image" in modalities
        has_text = "text" in modalities
        
        if has_images:  # Images require more memory
            if gpu_memory < 4:
                batch_size = 4
            elif gpu_memory < 8:
                batch_size = 8 if dataset_size < 50000 else 16
            elif gpu_memory < 12:
                batch_size = 16 if dataset_size < 50000 else 32
            else:
                batch_size = 32
        elif has_text:  # Text is more memory-intensive than tabular
            if gpu_memory < 8:
                batch_size = 16
            elif gpu_memory < 12:
                batch_size = 32 if dataset_size < 50000 else 64
            else:
                batch_size = 32
        else:  # Pure tabular
            batch_size = min(256, max(32, dataset_size // 100))
        
        # Determine epochs (inverse relationship with dataset size)
        if dataset_size < 5000:
            epochs = 45 if has_images else (40 if has_text else 30)
        elif dataset_size < 50000:
            epochs = 18 if has_images else (15 if has_text else 12)
        elif dataset_size < 500000:
            epochs = 12 if has_images else (10 if has_text else 8)
        else:
            epochs = 10 if has_images else (8 if has_text else 6)
        
        return batch_size, epochs
    
    def _compute_learning_rate(self, problem_type: str, dataset_size: int) -> float:
        """Compute adaptive learning rate."""
        base_lr = 1e-3
        
        # Adjust for problem type
        if "binary" in problem_type:
            base_lr = 5e-4
        elif "regression" in problem_type:
            base_lr = 1e-3
        else:
            base_lr = 1e-3
        
        # Adjust for dataset size (smaller datasets need larger learning rates for better convergence)
        if dataset_size < 1000:
            base_lr *= 2.0
        elif dataset_size < 10000:
            base_lr *= 1.5
        # No adjustment for large datasets
        
        return base_lr
    
    def _select_fusion_strategy(self, modalities: List[str], gpu_memory: float) -> str:
        """Select fusion strategy."""
        if len(modalities) > 1:
            if gpu_memory >= 8:
                return "attention"  # More powerful but requires more memory
            else:
                return "concatenation"  # Simpler, less memory
        return "concatenation"
