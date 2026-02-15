"""Run API with correct Python path - Enhanced version with GPU support"""
import sys
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch

# Add apex directory to path
apex_dir = Path(__file__).parent
sys.path.insert(0, str(apex_dir))

# Create enhanced API with PyTorch
app = FastAPI(title="APEX Framework API", version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GPU Detection
GPU_AVAILABLE = torch.cuda.is_available()
GPU_DEVICE = torch.cuda.get_device_name(0) if GPU_AVAILABLE else "CPU"

# Global progress tracking
ingestion_progress = {
    "status": "idle",
    "progress": 0,
    "message": "",
    "current_source": "",
    "datasets": [],
    "successful": 0,
    "failed": 0,
    "total": 0,
    "cache_location": ""
}

@app.get("/")
async def root():
    return {
        "message": "APEX Framework API",
        "status": "running",
        "gpu_available": GPU_AVAILABLE,
        "device": GPU_DEVICE
    }

@app.get("/health")
async def health():
    """Health check with GPU status"""
    return {
        "status": "healthy",
        "service": "APEX API",
        "gpu_available": GPU_AVAILABLE,
        "device": GPU_DEVICE,
        "cuda_version": torch.version.cuda
    }

@app.get("/config")
async def get_config():
    from config.hyperparameters import HyperparameterConfig
    config = HyperparameterConfig()
    return config.to_dict()

@app.post("/predict")
async def predict(data: dict):
    return {"prediction": "placeholder", "confidence": 0.95}

# Global ingestion tracking
last_ingestion_metadata = {
    "ingested_sources": [],      # URLs/paths that were actually ingested
    "ingested_hashes": [],       # Hash IDs of ingested datasets
    "sources_to_hashes": {},     # Mapping source -> hash for tracking
    "timestamp": None
}

@app.post("/ingest/datasets")
async def ingest_datasets(payload: dict):
    """Ingest datasets from multiple sources (Kaggle, HTTP, Local)"""
    global ingestion_progress
    
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from data_ingestion.ingestion_manager import DataIngestionManager
        
        sources = payload.get("sources", [])
        if not sources:
            return {"error": "No sources provided", "status": "failed"}
        
        # Initialize progress
        ingestion_progress.update({
            "status": "running",
            "progress": 0,
            "message": "Starting ingestion...",
            "datasets": [],
            "successful": 0,
            "failed": 0,
            "total": len(sources),
            "cache_location": str(Path("./data/dataset_cache").absolute())
        })
        
        manager = DataIngestionManager()
        ingested_hashes = []  # Track successfully ingested hashes
        
        for idx, source in enumerate(sources):
            try:
                ingestion_progress["current_source"] = source[:50]
                ingestion_progress["progress"] = int((idx / len(sources)) * 50)
                ingestion_progress["message"] = f"Ingesting: {source[:50]}..."
                print(f"📥 Ingesting: {source}")
                
                # Ingest with callback
                def progress_cb(progress, message):
                    # Map 0-100 progress to this source's portion
                    overall_progress = int(
                        (idx / len(sources)) * 50 +
                        (progress / 100) * (50 / len(sources))
                    )
                    ingestion_progress["progress"] = overall_progress
                    ingestion_progress["message"] = message
                    print(f"[{progress}%] {message}")
                
                loaded_data, metadata = manager.ingest_data(
                    source,
                    progress_callback=progress_cb
                )
                
                if loaded_data:
                    ingestion_progress["successful"] += 1
                    # Get dataset info
                    for hash_id, data in loaded_data.items():
                        if data is not None:
                            ingested_hashes.append(hash_id)  # Track hash
                            # Determine cache status
                            cache_status = metadata.get("cache_status", {}).get(source, "unknown")
                            if cache_status == "cached":
                                status_display = "Cached"
                            else:
                                status_display = "Success"
                            
                            ingestion_progress["datasets"].append({
                                "source": source,
                                "hash": hash_id,
                                "shape": list(data.shape),
                                "columns": list(data.columns[:5]),
                                "status": status_display
                            })
                else:
                    ingestion_progress["failed"] += 1
                    ingestion_progress["datasets"].append({
                        "source": source,
                        "status": "Error"
                    })
            
            except Exception as e:
                print(f"❌ Error ingesting {source}: {e}")
                ingestion_progress["failed"] += 1
                ingestion_progress["datasets"].append({
                    "source": source,
                    "status": "Error"
                })
        
        ingestion_progress["status"] = "completed"
        ingestion_progress["progress"] = 100
        ingestion_progress["message"] = "Ingestion completed!"
        
        # Store ingested metadata globally for schema detection
        global last_ingestion_metadata
        last_ingestion_metadata = {
            "ingested_sources": sources,
            "ingested_hashes": ingested_hashes,
            "timestamp": datetime.now().isoformat()
        }
        print(f"✅ Tracked {len(ingested_hashes)} ingested datasets: {ingested_hashes}")
        
        return ingestion_progress
    
    except Exception as e:
        ingestion_progress["status"] = "failed"
        ingestion_progress["message"] = str(e)
        return ingestion_progress


@app.get("/cache/stats")
async def cache_stats():
    """Get cache directory statistics"""
    try:
        import json
        from pathlib import Path
        cache_dir = Path("./data/dataset_cache")
        if not cache_dir.exists():
            return {
                "cache_location": str(cache_dir.absolute()),
                "total_items": 0,
                "total_size_mb": 0,
                "items": []
            }
        metadata_file = cache_dir / "cache_metadata.json"
        items = []
        total_size = 0.0
        item_list = []
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                items = list(metadata.keys())
        for item in items:
            size_mb = metadata[item].get("size_mb", 0)
            # If size_mb is very small, show at least 0.01 MB for visibility
            size_mb = max(round(size_mb, 2), 0.01 if size_mb > 0 else 0)
            item_list.append({"name": item, "size_mb": size_mb})
            total_size += size_mb
        return {
            "cache_location": str(cache_dir.absolute()),
            "total_items": len(items),
            "total_size_mb": round(total_size, 2),
            "items": item_list
        }
    except Exception as e:
        return {"error": str(e), "total_items": 0}

@app.get("/ingest/status")
async def ingest_status():
    """Get current ingestion progress status"""
    global ingestion_progress
    return ingestion_progress

@app.post("/cache/clear")
async def cache_clear():
    """Clear dataset cache"""
    try:
        import shutil
        import json
        from pathlib import Path
        
        cache_dir = Path("./data/dataset_cache")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        
        # Recreate empty cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = cache_dir / "cache_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({}, f)
        
        return {"message": "Cache cleared successfully", "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}

@app.get("/schema/statistics")
async def schema_statistics():
    """Get schema detection statistics from cached data."""
    try:
        import json
        from pathlib import Path
        import pandas as pd
        sys.path.insert(0, str(Path(__file__).parent))
        from data_ingestion.schema_detector import SchemaDetector
        cache_dir = Path("./data/dataset_cache")
        metadata_file = cache_dir / "cache_metadata.json"
        if not metadata_file.exists():
            return {"error": "No cached datasets found."}
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        # Use first cached dataset
        if not metadata:
            return {"error": "No cached datasets found."}
        first_hash = next(iter(metadata))
        data_file = cache_dir / first_hash / "data.parquet"
        if not data_file.exists():
            return {"error": "No cached data file found."}
        df = pd.read_parquet(data_file)
        detector = SchemaDetector()
        result = detector.detect_schema(df)
        return {
            "column_count": len(df.columns),
            "problem_type": result.problem_type,
            "sample_count": len(df),
            "modalities": {m: len(getattr(result, f"{m}_cols")) for m in result.modalities},
            "data_types": {dt: sum(1 for col in result.detected_columns if col['dtype'] == dt) for dt in set(col['dtype'] for col in result.detected_columns)},
            "detection_confidence": result.detection_confidence
        }
    except Exception as e:
        return {"error": str(e)}

# Schema detection endpoint (Phase 2)
@app.post("/detect-schema")
async def detect_schema(request: Request):
    """Detect schema for ONLY the ingested datasets from Phase 1."""
    try:
        import json
        from pathlib import Path
        import pandas as pd
        sys.path.insert(0, str(Path(__file__).parent))
        from data_ingestion.schema_detector import MultiDatasetSchemaDetector
        from dataclasses import asdict
        
        cache_dir = Path("./data/dataset_cache")
        metadata_file = cache_dir / "cache_metadata.json"
        
        if not metadata_file.exists():
            return JSONResponse({"error": "No cached datasets found."}, status_code=400)
        
        # Get only the ingested hashes from Phase 1, or fallback to all cached hashes if empty
        ingested_hashes = last_ingestion_metadata.get("ingested_hashes", [])
        ingested_sources = last_ingestion_metadata.get("ingested_sources", [])
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        if not ingested_hashes:
            # Fallback: use all cached hashes
            ingested_hashes = list(metadata.keys())
            ingested_sources = [metadata[h].get("source", h) for h in ingested_hashes]
            print(f"⚠️ No session ingested hashes, using all cached: {ingested_hashes}")
        if not ingested_hashes:
            return JSONResponse({"error": "No datasets were ingested or cached. Run Phase 1 first."}, status_code=400)
        print(f"📊 Schema Detection: Processing {len(ingested_hashes)} ingested/cached datasets")
        
        # Load ONLY ingested datasets
        datasets = {}
        for hash_id in ingested_hashes:
            if hash_id not in metadata:
                print(f"⚠️ Hash {hash_id} not in metadata")
                continue
            
            meta = metadata[hash_id]
            cache_path = cache_dir / hash_id
            df = None
            
            # Try Parquet
            for file_path in cache_path.glob("*.parquet"):
                try:
                    df = pd.read_parquet(file_path)
                    print(f"✅ Loaded Parquet: {hash_id} ({df.shape})")
                    break
                except Exception as e:
                    print(f"⚠️ Parquet read failed: {e}")
                    continue
            
            # Try CSV if Parquet failed
            if df is None:
                for file_path in cache_path.glob("*.csv"):
                    try:
                        df = pd.read_csv(file_path)
                        print(f"✅ Loaded CSV: {hash_id} ({df.shape})")
                        break
                    except Exception as e:
                        print(f"⚠️ CSV read failed: {e}")
                        continue
            
            # Try JSON if CSV failed
            if df is None:
                for file_path in cache_path.glob("*.json"):
                    try:
                        df = pd.read_json(file_path)
                        print(f"✅ Loaded JSON: {hash_id} ({df.shape})")
                        break
                    except Exception as e:
                        print(f"⚠️ JSON read failed: {e}")
                        continue
            
            # Try other formats
            if df is None:
                for file_path in cache_path.iterdir():
                    if file_path.is_file():
                        try:
                            if file_path.suffix in [".txt", ".data", ".tsv"]:
                                df = pd.read_csv(file_path, sep=r'\s+' if file_path.suffix == ".data" else None)
                                print(f"✅ Loaded {file_path.suffix}: {hash_id}")
                                break
                        except Exception as e:
                            print(f"⚠️ {file_path.suffix} read failed: {e}")
                            continue
            
            if df is not None:
                datasets[hash_id] = df
        
        if not datasets:
            print("❌ No valid ingested datasets found")
            return JSONResponse({"error": "Could not load any ingested datasets."}, status_code=400)
        
        # Detect schema for ingested datasets
        detector = MultiDatasetSchemaDetector()
        result = detector.detect_schema(datasets)
        
        return {
            "status": "success",
            "datasets_detected": len(datasets),
            "ingested_sources": ingested_sources,
            "schema_detection": result,
            "modalities_found": result.get("modalities", []),
            "target_column": result.get("target_column"),
            "problem_type": result.get("problem_type"),
            "confidence": result.get("detection_confidence", 0.0)
        }
        
    except Exception as e:
        import traceback
        print(f"❌ Schema detection error: {e}")
        print(traceback.format_exc())
        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc()},
            status_code=500
        )


# Phase 3: Preprocessing Endpoint
@app.post("/preprocess")
async def preprocess_data(request: Request):
    """Preprocess data based on detected modalities."""
    try:
        body = await request.json()
        modalities = body.get("modalities", [])
        
        from preprocessing.image_preprocessor import ImagePreprocessor
        from preprocessing.text_preprocessor import TextPreprocessor
        from preprocessing.tabular_preprocessor import TabularPreprocessor
        
        preprocessing_config = {
            "image": ImagePreprocessor().get_default_config() if "image" in modalities else None,
            "text": TextPreprocessor().get_default_config() if "text" in modalities else None,
            "tabular": TabularPreprocessor().get_default_config() if "tabular" in modalities else None
        }
        
        return {
            "status": "success",
            "preprocessing_config": preprocessing_config,
            "modalities_processed": modalities
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# Phase 4: Model Selection Endpoint
@app.post("/select-model")
async def select_model(request: Request):
    """Select best model based on problem type and data characteristics."""
    try:
        body = await request.json()
        problem_type = body.get("problem_type", "unsupervised")
        modalities = body.get("modalities", [])
        
        from automl.model_selector import ModelSelector
        
        selector = ModelSelector()
        recommendations = selector.recommend_models(problem_type, modalities)
        
        return {
            "status": "success",
            "problem_type": problem_type,
            "modalities": modalities,
            "recommended_models": recommendations,
            "best_model": recommendations[0] if recommendations else None
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# Phase 5: Training Endpoint
@app.post("/train-model")
async def train_model(request: Request):
    """Train selected model on preprocessed data."""
    try:
        body = await request.json()
        model_name = body.get("model_name", "xgboost")
        config = body.get("config", {})
        epochs = body.get("epochs", 10)
        
        from automl.trainer import ModelTrainer
        
        trainer = ModelTrainer()
        training_config = trainer.get_training_config(model_name, epochs, config)
        
        return {
            "status": "success",
            "model": model_name,
            "training_config": training_config,
            "estimated_time": "estimated based on data size",
            "gpu_enabled": GPU_AVAILABLE
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# Phase 6: Monitoring Endpoint
@app.post("/monitor-model")
async def monitor_model(request: Request):
    """Setup model monitoring and performance tracking."""
    try:
        body = await request.json()
        model_id = body.get("model_id", "model_001")
        
        from monitoring.performance_tracker import PerformanceTracker
        
        tracker = PerformanceTracker()
        monitor_config = tracker.get_monitoring_config()
        
        return {
            "status": "success",
            "model_id": model_id,
            "monitoring_enabled": True,
            "metrics_tracked": ["accuracy", "precision", "recall", "f1", "auc"],
            "drift_detection": True,
            "alert_thresholds": {
                "accuracy_drop": 0.05,
                "drift_score": 0.7
            }
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# Phase 7: Prediction Endpoint (NEW)
@app.post("/predict")
async def predict_multimodal(request: Request):
    """Make predictions with multimodal input (image, text, tabular)."""
    try:
        body = await request.json()
        
        # Extract multimodal inputs
        image_data = body.get("image_data", None)  # Base64 or path
        text_data = body.get("text_data", None)    # Text string
        tabular_data = body.get("tabular_data", None)  # Dict of features
        model_name = body.get("model_name", "xgboost")
        
        from modelss.predictor import MultimodalPredictor
        
        predictor = MultimodalPredictor()
        
        # Process multimodal inputs
        predictions = predictor.predict(
            image_data=image_data,
            text_data=text_data,
            tabular_data=tabular_data,
            model_name=model_name
        )
        
        return {
            "status": "success",
            "prediction": predictions.get("prediction", None),
            "confidence": predictions.get("confidence", 0.0),
            "modalities_used": [m for m in ["image", "text", "tabular"] if body.get(f"{m}_data") is not None],
            "model_used": model_name
        }
    except Exception as e:
        return JSONResponse({"error": str(e), "traceback": str(Exception.__traceback__)}, status_code=500)


if __name__ == "__main__":
    print("🚀 Starting APEX Framework API Server...")
    print(f"📍 API will be available at: http://localhost:8001")
    print(f"🎮 GPU: {'✅ ENABLED - ' + GPU_DEVICE if GPU_AVAILABLE else '❌ DISABLED - CPU Mode'}")
    print(f"📚 API Documentation: http://localhost:8001/docs")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")