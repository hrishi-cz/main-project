"""
DEPLOYMENT GUIDE: Running FIX-4 in Production

==================================================================
TABLE OF CONTENTS
==================================================================

1. Environment setup (Docker, dependencies)
2. Pre-training: Schema detection + modality validation
3. Model training: On embeddings
4. Serving: Flask/FastAPI inference server
5. Monitoring: Track modality scores, model drift
6. CI/CD: Automated testing, model updates
7. Troubleshooting: Common issues and fixes

==================================================================

1. # ENVIRONMENT SETUP

## 1.1. Docker Image (Recommended for production)

Create Dockerfile:

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies

RUN apt-get update && apt-get install -y \\
build-essential \\
libsm6 libxext6 \\
libxrender-dev \\
&& rm -rf /var/lib/apt/lists/\*

# Copy requirements

COPY requirements.txt .

# Install Python packages

RUN pip install --no-cache-dir -r requirements.txt

# Copy code

COPY . .

# Expose port

EXPOSE 5000

# Health check

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
CMD python -c "import requests; requests.get('http://localhost:5000/health')"

# Run server

CMD ["python", "api/run_server.py"]

Build and run:

docker build -t apex2-fix4:latest .
docker run -p 5000:5000 -e MODALITY_PIPELINE=true apex2-fix4:latest

## 1.2. Local Development

Create virtual environment:

python -m venv venv
source venv/bin/activate # On Windows: venv\\Scripts\\activate

pip install -r requirements.txt

## 1.3. AWS/Cloud Deployment

For AWS:

1. Build Docker image
2. Push to ECR: aws ecr push-image ...
3. Deploy to ECS/Sagemaker
4. Set up RDS for model registry
5. Enable CloudWatch monitoring

# ================================================================== 2. PRE-TRAINING: SCHEMA DETECTION & VALIDATION

Step 1: Analyze raw data

from data_ingestion.integrator import process_dataset
from pathlib import Path
import json

# Load data

data = load_raw_dataset("path/to/data.csv")

# Run modality pipeline

schema_results = process_dataset(
raw_data_dict={
"images": load_images("data/images"),
"descriptions": load_texts("data/descriptions.txt"),
"tabular": load_tabular("data/features.csv"),
},
y=load_targets("data/targets.csv"),
task_type="regression",
)

# Save schema

schema = {}
for field_name, metadata in schema_results.items():
schema[field_name] = {
"modality": metadata.modality_name,
"embedding_dim": metadata.embeddings.shape[1],
"encoder": metadata.encoder_name,
"score": metadata.final_score(),
"is_valid": metadata.is_valid,
}

with open("models/schema.json", "w") as f:
json.dump(schema, f, indent=2)

logger.info(f"Schema saved: {list(schema.keys())}")

Step 2: Save embeddings cache (optional, for faster retraining)

from pickle import dump

embedding_cache = {
field: meta.embeddings
for field, meta in schema_results.items()
}

with open("models/embeddings_cache.pkl", "wb") as f:
dump(embedding_cache, f)

logger.info("Embeddings cache saved")

Step 3: Report on modalities

print("Modality Analysis Report")
print("-" \* 50)
for field_name, metadata in schema_results.items():
print(f"\n{field_name}:")
print(f" Predictability: {metadata.predictability_score:.3f}")
print(f" Complementarity: {metadata.complementarity_score:.3f}")
print(f" Final Score: {metadata.final_score():.3f}")
print(f" Valid: {'✓' if metadata.is_valid else '✗'}")

Expected output:

Modality Analysis Report

---

images:
Predictability: 0.878
Complementarity: 0.652
Final Score: 0.821
Valid: ✓

descriptions:
Predictability: 0.891
Complementarity: 0.715
Final Score: 0.844
Valid: ✓

tabular:
Predictability: 0.856
Complementarity: 0.680
Final Score: 0.811
Valid: ✓

# ================================================================== 3. MODEL TRAINING

Step 1: Prepare training data

from data_ingestion.integrator import Integrator
from sklearn.model_selection import train_test_split
import joblib

integrator = Integrator(min_predictability=0.3)

# Load or generate training data

train_data = load_training_data("path/to/train")
X_train_raw, y_train = train_data["X"], train_data["y"]

# Split

X_train, X_val, y_train, y_val = train_test_split(
X_train_raw, y_train, test_size=0.2, random_state=42
)

Step 2: Process with modality pipeline

logger.info("Processing training data with modality pipeline...")

modality_results = integrator.process_multimodal(
raw_data_dict=X_train,
y=y_train,
task_type="regression",
)

# Combine embeddings

embeddings_list = [
meta.embeddings
for meta in modality_results.values()
if meta.is_valid
]
X_train_embeddings = np.concatenate(embeddings_list, axis=1)

Step 3: Train model

from sklearn.ensemble import RandomForestRegressor

logger.info("Training RandomForest on embeddings...")

model = RandomForestRegressor(
n_estimators=200,
max_depth=20,
random_state=42,
n_jobs=-1,
)

model.fit(X_train_embeddings, y_train)

# Evaluate

train_score = model.score(X_train_embeddings, y_train)
logger.info(f"Training R² score: {train_score:.3f}")

Step 4: Validate on held-out set

# Process validation data

meta_val = integrator.process_multimodal(
raw_data_dict=X_val,
y=y_val,
task_type="regression",
)

embeddings_val = np.concatenate([
meta.embeddings for meta in meta_val.values()
if meta.is_valid
], axis=1)

val_score = model.score(embeddings_val, y_val)
logger.info(f"Validation R² score: {val_score:.3f}")

Step 5: Save model & metadata

joblib.dump(model, "models/apex_v1_production.pkl")
joblib.dump(modality_results, "models/modality_metadata.pkl")

logger.info("Model and metadata saved")

# ================================================================== 4. SERVING: INFERENCE SERVER

Use Flask (lightweight) or FastAPI (modern):

from flask import Flask, request, jsonify
from data_ingestion.integrator import Integrator
import joblib
import numpy as np

app = Flask(**name**)

# Load model at startup

model = joblib.load("models/apex_v1_production.pkl")
integrator = Integrator()

@app.route("/health", methods=["GET"])
def health():
"""Health check endpoint."""
return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
"""Single prediction endpoint."""
try:
data = request.json

          # Assume data is raw (images, text, etc.)
          # Encode using integrator
          meta = integrator.process_single_modality(
              raw_data=data["features"],
              modality=data.get("modality"),
          )

          embeddings = meta.embeddings
          prediction = model.predict(embeddings)[0]

          return jsonify({
              "prediction": float(prediction),
              "confidence": float(model.predict_proba(embeddings)[0].max())
              if hasattr(model, "predict_proba") else None,
              "modality_score": meta.final_score(),
          }), 200

      except Exception as e:
          logger.error(f"Prediction failed: {e}")
          return jsonify({"error": str(e)}), 400

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
"""Batch prediction endpoint."""
try:
data = request.json

          # Process multiple samples
          results = []
          for sample in data["samples"]:
              meta = integrator.process_single_modality(
                  raw_data=sample["features"],
                  modality=sample.get("modality"),
              )
              embeddings = meta.embeddings
              pred = model.predict(embeddings)[0]
              results.append({
                  "prediction": float(pred),
                  "modality_score": meta.final_score(),
              })

          return jsonify({"results": results}), 200

      except Exception as e:
          logger.error(f"Batch prediction failed: {e}")
          return jsonify({"error": str(e)}), 400

if **name** == "**main**":
app.run(host="0.0.0.0", port=5000, debug=False)

Test the server:

curl -X GET http://localhost:5000/health

curl -X POST http://localhost:5000/predict \\
-H "Content-Type: application/json" \\
-d '{
"features": [[1.0, 2.0, 3.0]],
"modality": "tabular"
}'

# ================================================================== 5. MONITORING

Track key metrics:

1. Modality scores (over time)
2. Model predictions (drift detection)
3. Latency (inference time)
4. Errors (failed predictions)

Setup monitoring:

from monitoring.performance_tracker import PerformanceTracker
import json
from datetime import datetime

tracker = PerformanceTracker()

def track_prediction(features, prediction, modality_scores, latency_ms):
"""Log prediction for monitoring."""
tracker.log_prediction(
timestamp=datetime.now(),
features=features,
prediction=prediction,
modality_scores=modality_scores,
latency_ms=latency_ms,
)

# Save logs periodically

logs = tracker.get*logs()
with open(f"logs/predictions*{datetime.now().date()}.jsonl", "a") as f:
for log in logs:
f.write(json.dumps(log) + "\\n")

Example monitoring metrics (CloudWatch/Prometheus):

- apex_modality_scores{modality="text"} = 0.85
- apex_modality_scores{modality="image"} = 0.88
- apex_model_accuracy = 0.92
- apex_inference_latency_ms = 45.2
- apex_prediction_errors_total = 3

# ================================================================== 6. CI/CD PIPELINE

GitHub Actions example:

name: Train and Deploy

on:
push:
branches: [main, develop]

jobs:
test:
runs-on: ubuntu-latest
steps: - uses: actions/checkout@v2 - name: Set up Python
uses: actions/setup-python@v2
with:
python-version: '3.10' - name: Install dependencies
run: |
pip install -r requirements.txt
pip install pytest pytest-cov - name: Run tests
run: pytest tests/ -v --cov

    train:
      if: success()
      needs: test
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v2
        - name: Set up Python
          uses: actions/setup-python@v2
          with:
            python-version: '3.10'
        - name: Install dependencies
          run: pip install -r requirements.txt
        - name: Detect schema
          run: python scripts/detect_schema.py
        - name: Train model
          run: python scripts/train_model.py
        - name: Validate model
          run: python scripts/validate_model.py
        - name: Upload model
          run: python scripts/upload_model.py

    deploy:
      if: success() && github.ref == 'refs/heads/main'
      needs: train
      runs-on: ubuntu-latest
      steps:
        - name: Deploy to Docker Hub
          run: |
            docker build -t $DOCKER_REPO:latest .
            docker push $DOCKER_REPO:latest
        - name: Deploy to ECS
          run: python scripts/deploy_ecs.py

# ================================================================== 7. TROUBLESHOOTING

Issue: Modality scores too low (< 0.3)

- Check data quality (missing values, outliers)
- Verify encoder is appropriate
- Try different task_type (regression vs classification)
- Increase training data size

Issue: Slow inference (> 500ms per sample)

- Use embedding cache (precomputed)
- Reduce image resolution (64x64 → 32x32)
- Use smaller encoder (DistilBERT → SBERT)
- Deploy on GPU

Issue: High model error rate

- Check modality scores (are all modalities valid?)
- Try ensemble (multiple models)
- Retrain on latest data
- Investigate feature drift

Issue: Memory errors during training

- Reduce batch size
- Use gradient accumulation
- Cache embeddings to disk
- Process in mini-batches

Issue: Docker build fails

- Check Python version compatibility
- Verify all system dependencies installed
- Clean build cache: docker build --no-cache

==================================================================
PRODUCTION CHECKLIST
==================================================================

Before deploying:

☐ All unit tests passing
☐ Integration tests passing
☐ Modality scores documented
☐ Model performance benchmarked
☐ Error handling implemented
☐ Logging configured
☐ Monitoring setup
☐ CI/CD pipeline configured
☐ Rollback plan documented
☐ Health checks working
☐ Load tested
☐ Security reviewed

==================================================================
"""

# This is a markdown/guide file. Follow the steps above for production deployment.
