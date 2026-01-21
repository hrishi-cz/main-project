"""
Test script to verify all imports work
Run this BEFORE starting the API
"""

print("Testing APEX AutoVision imports...\n")

# Test 1: PyTorch
print("[1/10] Testing PyTorch...")
try:
    import torch
    print(f"  ✅ PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠️  GPU not available")
except ImportError as e:
    print(f"  ❌ PyTorch import failed: {e}")

# Test 2: FastAPI
print("\n[2/10] Testing FastAPI...")
try:
    import fastapi
    import uvicorn
    print(f"  ✅ FastAPI {fastapi.__version__}")
except ImportError as e:
    print(f"  ❌ FastAPI import failed: {e}")

# Test 3: pandas
print("\n[3/10] Testing pandas...")
try:
    import pandas as pd
    print(f"  ✅ pandas {pd.__version__}")
except ImportError as e:
    print(f"  ❌ pandas import failed: {e}")

# Test 4: scipy
print("\n[4/10] Testing scipy...")
try:
    import scipy
    print(f"  ✅ scipy {scipy.__version__}")
except ImportError as e:
    print(f"  ❌ scipy import failed: {e}")

# Test 5: Monitoring module
print("\n[5/10] Testing monitoring module...")
try:
    from monitoring.drift_detector import DriftDetector
    from monitoring.performance_tracker import PerformanceTracker
    print("  ✅ Drift detector imported")
    print("  ✅ Performance tracker imported")
except ImportError as e:
    print(f"  ❌ Monitoring import failed: {e}")

# Test 6: Config module
print("\n[6/10] Testing config module...")
try:
    from config.hyperparameters import HYPERPARAMETERS, PRESETS, validate_hyperparameters
    print("  ✅ Hyperparameters imported")
    print(f"  ✅ {len(HYPERPARAMETERS)} parameters defined")
    print(f"  ✅ {len(PRESETS)} presets available")
except ImportError as e:
    print(f"  ❌ Config import failed: {e}")

# Test 7: Retraining pipeline
print("\n[7/10] Testing retraining pipeline...")
try:
    from pipeline.retraining_pipeline import RetrainingPipeline
    print("  ✅ Retraining pipeline imported")
except ImportError as e:
    print(f"  ❌ Retraining import failed: {e}")

# Test 8: Orchestrator
print("\n[8/10] Testing orchestrator...")
try:
    from pipeline.orchestrator import PipelineOrchestrator
    print("  ✅ Orchestrator imported")
except ImportError as e:
    print(f"  ❌ Orchestrator import failed: {e}")

# Test 9: Model Registry
print("\n[9/10] Testing model registry...")
try:
    from registry.model_registry import ModelRegistry
    print("  ✅ Model registry imported")
except ImportError as e:
    print(f"  ❌ Model registry import failed: {e}")

# Test 10: Data loader
print("\n[10/10] Testing data loader...")
try:
    from data_ingestion.loader import DatasetLoader
    print("  ✅ Data loader imported")
except ImportError as e:
    print(f"  ❌ Data loader import failed: {e}")

print("\n" + "="*60)
print("Import test complete!")
print("="*60)
print("\nIf any imports failed, install missing packages:")
print("  pip install -r requirements.txt")
print("\nIf all passed, start the server:")
print("  python api/main_enhanced.py")
print("="*60)
