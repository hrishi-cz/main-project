"""
Enhanced API for APEX AutoVision - FIXED VERSION
No warnings, all imports corrected, server actually runs
"""
import sys
import uuid
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
# Try imports with error handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available")

# Import your existing modules
try:
    from pipeline.orchestrator import PipelineOrchestrator
    ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    ORCHESTRATOR_AVAILABLE = False
    print(f"Warning: Could not import PipelineOrchestrator: {e}")

try:
    from registry.model_registry import ModelRegistry
    REGISTRY_AVAILABLE = True
except ImportError as e:
    REGISTRY_AVAILABLE = False
    print(f"Warning: Could not import ModelRegistry: {e}")

try:
    from monitoring.drift_detector import DriftDetector
    DRIFT_AVAILABLE = True
except ImportError as e:
    DRIFT_AVAILABLE = False
    print(f"Warning: Could not import DriftDetector: {e}")

try:
    from monitoring.performance_tracker import PerformanceTracker
    PERF_TRACKER_AVAILABLE = True
except ImportError as e:
    PERF_TRACKER_AVAILABLE = False
    print(f"Warning: Could not import PerformanceTracker: {e}")

try:
    from pipeline.retraining_pipeline import RetrainingPipeline
    RETRAIN_AVAILABLE = True
except ImportError as e:
    RETRAIN_AVAILABLE = False
    print(f"Warning: Could not import RetrainingPipeline: {e}")

try:
    from config.hyperparameters import (
        HYPERPARAMETERS,
        PRESETS,
        validate_hyperparameters,
        get_preset
    )
    HYPERPARAMS_AVAILABLE = True
except ImportError as e:
    HYPERPARAMS_AVAILABLE = False
    print(f"Warning: Could not import hyperparameters: {e}")
    # Provide defaults
    HYPERPARAMETERS = {}
    PRESETS = {}
    
    def validate_hyperparameters(params):
        return params
    
    def get_preset(name):
        return {}

# ==================== FastAPI App ====================
app = FastAPI(
    title="APEX AutoVision API",
    version="2.0.0",
    description="Multimodal AutoML with Drift Detection & Retraining"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Job storage
JOB_STORE: Dict[str, Dict[str, Any]] = {}
DRIFT_DETECTORS: Dict[str, Any] = {}

# ==================== Request Models ====================
class TrainRequest(BaseModel):
    dataset_sources: List[str]
    hyperparameters: Optional[Dict[str, Any]] = None

class RetrainRequest(BaseModel):
    model_id: str
    new_data_sources: List[str]
    hyperparameters: Optional[Dict[str, Any]] = None

class DriftCheckRequest(BaseModel):
    model_id: str
    new_data_source: str

class HyperparameterValidation(BaseModel):
    hyperparameters: Dict[str, Any]

# ==================== Root Endpoint ====================
@app.get("/")
def root():
    """Health check and system info"""
    gpu_available = False
    if TORCH_AVAILABLE:
        gpu_available = torch.cuda.is_available()
    
    return {
        "message": "APEX AutoVision API - Enhanced",
        "version": "2.0.0",
        "status": "running",
        "features": {
            "training": ORCHESTRATOR_AVAILABLE,
            "drift_detection": DRIFT_AVAILABLE,
            "retraining": RETRAIN_AVAILABLE,
            "hyperparameters": HYPERPARAMS_AVAILABLE,
        },
        "gpu_available": gpu_available,
    }

# ==================== Training ====================
def _run_training_job(
    job_id: str,
    dataset_sources: List[str],
    hyperparameters: Optional[Dict[str, Any]]
):
    """Background training task"""
    try:
        if not ORCHESTRATOR_AVAILABLE:
            raise Exception("PipelineOrchestrator not available")
        
        JOB_STORE[job_id]["status"] = "running"
        orchestrator = PipelineOrchestrator()
        
        def progress_callback(epoch: int, batch: int, total_batches: int, loss: float):
            JOB_STORE[job_id]["current_epoch"] = epoch
            JOB_STORE[job_id]["loss"] = round(loss, 4)
            if "total_epochs" in JOB_STORE[job_id]:
                progress = int((epoch / JOB_STORE[job_id]["total_epochs"]) * 100)
                JOB_STORE[job_id]["progress"] = progress
        
        result = orchestrator.run(
            dataset_sources=dataset_sources,
            progress_callback=progress_callback
        )
        
        JOB_STORE[job_id]["status"] = "completed"
        JOB_STORE[job_id]["result"] = result
        JOB_STORE[job_id]["progress"] = 100
        
    except Exception as e:
        JOB_STORE[job_id]["status"] = "failed"
        JOB_STORE[job_id]["error"] = str(e)
        import traceback
        JOB_STORE[job_id]["traceback"] = traceback.format_exc()

@app.post("/train")
def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """Start model training"""
    if not ORCHESTRATOR_AVAILABLE:
        raise HTTPException(500, "Training not available - PipelineOrchestrator not loaded")
    
    # Validate hyperparameters if provided
    if request.hyperparameters and HYPERPARAMS_AVAILABLE:
        try:
            validated_params = validate_hyperparameters(request.hyperparameters)
        except ValueError as e:
            raise HTTPException(400, str(e))
    
    job_id = str(uuid.uuid4())
    total_epochs = 50
    if request.hyperparameters and "num_epochs" in request.hyperparameters:
        total_epochs = request.hyperparameters["num_epochs"]
    
    JOB_STORE[job_id] = {
        "status": "queued",
        "progress": 0,
        "current_epoch": 0,
        "total_epochs": total_epochs,
        "loss": None,
        "result": None,
        "error": None,
    }
    
    background_tasks.add_task(
        _run_training_job,
        job_id,
        request.dataset_sources,
        request.hyperparameters
    )
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Training started"
    }

@app.get("/status/{job_id}")
def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in JOB_STORE:
        raise HTTPException(404, "Job not found")
    return JOB_STORE[job_id]

# ==================== Model Management ====================
@app.get("/models")
def list_models():
    """List all models"""
    if not REGISTRY_AVAILABLE:
        raise HTTPException(500, "ModelRegistry not available")
    
    try:
        models = ModelRegistry.list_models()
        model_list = []
        
        for model_id in models:
            try:
                info = ModelRegistry.get_model_info(model_id)
                model_list.append({
                    "model_id": model_id,
                    "problem_type": info.get("problem_type"),
                    "num_classes": info.get("num_classes"),
                    "dataset_size": info.get("dataset_size"),
                    "modalities": info.get("modalities"),
                    "final_f1": info.get("final_f1"),
                    "final_r2": info.get("final_r2"),
                })
            except Exception:
                continue
        
        return {"models": model_list}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/models/{model_id}")
def get_model_info(model_id: str):
    """Get model info"""
    if not REGISTRY_AVAILABLE:
        raise HTTPException(500, "ModelRegistry not available")
    
    try:
        info = ModelRegistry.get_model_info(model_id)
        return info
    except FileNotFoundError:
        raise HTTPException(404, "Model not found")
    except Exception as e:
        raise HTTPException(500, str(e))

# ==================== Drift Detection ====================
@app.post("/drift/check")
def check_drift(request: DriftCheckRequest):
    """Check for drift"""
    if not DRIFT_AVAILABLE:
        raise HTTPException(500, "DriftDetector not available")
    
    if not PANDAS_AVAILABLE:
        raise HTTPException(500, "pandas not available")
    
    try:
        # Load reference data (in production, use actual training data)
        new_df = pd.read_csv(request.new_data_source)
        
        if request.model_id not in DRIFT_DETECTORS:
            DRIFT_DETECTORS[request.model_id] = DriftDetector(new_df)
        
        detector = DRIFT_DETECTORS[request.model_id]
        results = detector.detect_drift(new_df)
        
        return {
            "model_id": request.model_id,
            "drift_results": results,
            "recommendation": "retrain" if results["overall_drift_detected"] else "continue",
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/drift/history/{model_id}")
def get_drift_history(model_id: str, limit: int = 50):
    """Get drift history"""
    if not PERF_TRACKER_AVAILABLE:
        raise HTTPException(500, "PerformanceTracker not available")
    
    try:
        tracker = PerformanceTracker(model_id)
        history = tracker.get_recent_metrics(limit)
        return {"model_id": model_id, "history": history, "count": len(history)}
    except Exception as e:
        raise HTTPException(500, str(e))

# ==================== Retraining ====================
@app.post("/retrain")
def retrain_model(request: RetrainRequest, background_tasks: BackgroundTasks):
    """Trigger retraining"""
    if not RETRAIN_AVAILABLE:
        raise HTTPException(500, "RetrainingPipeline not available")
    
    try:
        if request.hyperparameters and HYPERPARAMS_AVAILABLE:
            validate_hyperparameters(request.hyperparameters)
        
        pipeline = RetrainingPipeline(request.model_id)
        
        job_id = str(uuid.uuid4())
        JOB_STORE[job_id] = {
            "status": "running",
            "type": "retraining",
            "original_model_id": request.model_id,
        }
        
        def run_retrain():
            try:
                result = pipeline.retrain(
                    request.new_data_sources,
                    request.hyperparameters
                )
                JOB_STORE[job_id]["status"] = "completed"
                JOB_STORE[job_id]["new_model_id"] = result.get("model_id")
                JOB_STORE[job_id]["result"] = result
            except Exception as e:
                JOB_STORE[job_id]["status"] = "failed"
                JOB_STORE[job_id]["error"] = str(e)
        
        background_tasks.add_task(run_retrain)
        
        return {
            "job_id": job_id,
            "message": "Retraining started",
            "original_model_id": request.model_id
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/retrain/should/{model_id}")
def should_retrain(model_id: str, new_data_source: str):
    """Check if should retrain"""
    if not RETRAIN_AVAILABLE:
        raise HTTPException(500, "RetrainingPipeline not available")
    
    if not PANDAS_AVAILABLE:
        raise HTTPException(500, "pandas not available")
    
    try:
        pipeline = RetrainingPipeline(model_id)
        new_df = pd.read_csv(new_data_source)
        
        tracker = PerformanceTracker(model_id)
        recent = tracker.get_recent_metrics(1)
        current_metrics = recent[0]["metrics"] if recent else {}
        
        decision = pipeline.should_retrain(new_df, current_metrics)
        return decision
    except Exception as e:
        raise HTTPException(500, str(e))

# ==================== Hyperparameters ====================
@app.get("/hyperparameters/schema")
def get_hyperparameter_schema():
    """Get hyperparameter schema"""
    if not HYPERPARAMS_AVAILABLE:
        raise HTTPException(500, "Hyperparameters module not available")
    return {"parameters": HYPERPARAMETERS}

@app.get("/hyperparameters/presets")
def get_hyperparameter_presets():
    """Get presets"""
    if not HYPERPARAMS_AVAILABLE:
        raise HTTPException(500, "Hyperparameters module not available")
    return {"presets": PRESETS}

@app.get("/hyperparameters/preset/{preset_name}")
def get_preset_by_name(preset_name: str):
    """Get specific preset"""
    if not HYPERPARAMS_AVAILABLE:
        raise HTTPException(500, "Hyperparameters module not available")
    
    try:
        preset = get_preset(preset_name)
        return {"preset_name": preset_name, "hyperparameters": preset}
    except ValueError as e:
        raise HTTPException(404, str(e))

@app.post("/hyperparameters/validate")
def validate_hyperparams(request: HyperparameterValidation):
    """Validate hyperparameters"""
    if not HYPERPARAMS_AVAILABLE:
        return {"valid": True, "validated_parameters": request.hyperparameters}
    
    try:
        validated = validate_hyperparameters(request.hyperparameters)
        return {"valid": True, "validated_parameters": validated}
    except ValueError as e:
        return {"valid": False, "error": str(e)}

# ==================== Performance Monitoring ====================
@app.get("/monitoring/performance/{model_id}")
def get_performance_metrics(model_id: str, limit: int = 20):
    """Get performance metrics"""
    if not PERF_TRACKER_AVAILABLE:
        raise HTTPException(500, "PerformanceTracker not available")
    
    try:
        tracker = PerformanceTracker(model_id)
        metrics = tracker.get_recent_metrics(limit)
        return {"model_id": model_id, "metrics": metrics, "count": len(metrics)}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/monitoring/trend/{model_id}/{metric_name}")
def get_metric_trend(model_id: str, metric_name: str):
    """Get metric trend"""
    if not PERF_TRACKER_AVAILABLE:
        raise HTTPException(500, "PerformanceTracker not available")
    
    try:
        tracker = PerformanceTracker(model_id)
        trend = tracker.get_metric_trend(metric_name)
        return {"model_id": model_id, "metric_name": metric_name, "trend": trend}
    except Exception as e:
        raise HTTPException(500, str(e))

# ==================== Startup ====================
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("APEX AutoVision API Server")
    print("="*60)
    print(f"PyTorch Available: {TORCH_AVAILABLE}")
    print(f"Training Available: {ORCHESTRATOR_AVAILABLE}")
    print(f"Drift Detection Available: {DRIFT_AVAILABLE}")
    print(f"Retraining Available: {RETRAIN_AVAILABLE}")
    print(f"Hyperparameters Available: {HYPERPARAMS_AVAILABLE}")
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("="*60)
    print("\nStarting server on http://0.0.0.0:8000")
    print("API Docs: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)