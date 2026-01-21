"""Run API with correct Python path - Lightweight version"""
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# Add apex directory to path
apex_dir = Path(__file__).parent
sys.path.insert(0, str(apex_dir))

# Create lightweight API without torch imports
app = FastAPI(title="APEX Framework API", version="0.1.0")

@app.get("/")
async def root():
    return {"message": "APEX Framework API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "APEX API"}

@app.get("/config")
async def get_config():
    from config.hyperparameters import HyperparameterConfig
    config = HyperparameterConfig()
    return config.to_dict()

@app.post("/predict")
async def predict(data: dict):
    return {"prediction": "placeholder", "confidence": 0.95}

@app.get("/modules")
async def list_modules():
    return {
        "modules": [
            "Configuration",
            "Data Ingestion",
            "Preprocessing",
            "Monitoring",
            "Pipeline",
            "AutoML",
            "Model Registry"
        ]
    }

if __name__ == "__main__":
    print("🚀 Starting APEX Framework API Server...")
    print("📍 API will be available at: http://localhost:8001")
    print("📚 API Documentation: http://localhost:8001/docs")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")