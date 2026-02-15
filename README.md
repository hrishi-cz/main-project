# APEX 2 - Multimodal AutoML Framework

**Advanced Predictive Ensemble with eXtendable modularity**

A comprehensive framework for multimodal machine learning that handles images, text, and tabular data with automated model selection, retraining, and drift detection.

## Status

✅ **Project Fixed and Ready to Use**

All critical issues have been identified and fixed:

- ✅ Fixed hallucinated imports and missing functions
- ✅ Fixed API endpoint mismatches
- ✅ Implemented missing methods in core classes
- ✅ Integrated frontend with API
- ✅ Added NLTK data support
- ✅ All modules verified and working

## Project Structure

```
apex2-worktree/
├── api/                           # FastAPI REST endpoints
│   └── main_enhanced.py          # Main API server (port 8000)
├── frontend/                      # Streamlit web interface
│   └── app_enhanced.py           # Frontend app (port 8501)
├── pipeline/                      # ML pipeline orchestration
│   ├── orchestrator.py           # Pipeline tasks manager
│   ├── dataset_manager.py        # Dataset handling
│   └── retraining_pipeline.py    # Automated retraining
├── modelss/                       # Neural network models
│   ├── predictor.py              # Multimodal predictor
│   ├── fusion.py                 # Feature fusion strategies
│   └── encoders/                 # Input encoders
│       ├── image.py              # Image encoder (ResNet, etc.)
│       ├── text.py               # Text encoder (BERT, etc.)
│       └── tabular.py            # Tabular encoder (MLP)
├── preprocessing/                 # Data preprocessing
│   ├── image_preprocessor.py     # Image preprocessing
│   ├── text_preprocessor.py      # Text/NLP preprocessing
│   └── tabular_preprocessor.py   # Numerical/categorical preprocessing
├── data_ingestion/               # Data loading and schema
│   ├── loader.py                 # Universal data loader
│   └── schema.py                 # Data schema definitions
├── automl/                        # Automatic ML components
│   ├── model_selector.py         # Model selection logic
│   └── trainer.py                # Training loop
├── monitoring/                    # Model monitoring
│   ├── drift_detector.py         # Data drift detection
│   └── performance_tracker.py    # Performance metrics tracking
├── model_registry_pkg/           # Model registry
│   └── model_registry.py         # Model versioning and storage
├── config/                        # Configuration
│   └── hyperparameters.py        # Hyperparameter presets
├── utils/                         # Utilities
│   ├── progress_display.py       # Progress bars
│   └── nltk_setup.py             # NLTK data downloader
└── requirements.txt              # Dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run API Server

Open a terminal and start the API:

```bash
python run_api.py
# OR for more detailed output:
python api/main_enhanced.py
```

The API will be available at:

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **API ReDoc**: http://localhost:8000/redoc

### 3. Run Frontend (in another terminal)

```bash
streamlit run frontend/app_enhanced.py
```

Frontend will open at: http://localhost:8501

### 4. Run Verification Tests

```bash
python run_project_demo.py
```

This tests all modules and shows their status.

## Key Features

### 🔄 Multimodal Fusion

- **Image Processing**: ResNet50, MobileNetV3, Vision Transformers
- **Text Processing**: DistilBERT, RoBERTa, Sentence Transformers
- **Tabular Processing**: MLP, TabNet, FT-Transformer
- **Fusion Strategies**: Concatenation, Attention-based, Weighted

### 🤖 Automatic ML

- Automatic model selection based on data characteristics
- Hyperparameter presets (small, medium, large, fast)
- Adaptive batch sizing and epoch calculation

### 📊 Monitoring & Drift Detection

- Real-time performance tracking
- Statistical drift detection (Kolmogorov-Smirnov test)
- Performance trend analysis
- Automated retraining triggers

### 🔌 REST API

- Complete REST endpoints for training, inference, and monitoring
- Swagger/OpenAPI documentation
- Job-based asynchronous training
- Model registry with versioning

### 🎨 Web Interface

- Interactive Streamlit dashboard
- Real-time model management
- Performance visualization
- Monitoring and drift detection views

## API Endpoints

### System Status

- `GET /` - Health check and system info
- `GET /health` - Quick health status
- `GET /config` - System configuration

### Model Management

- `GET /models` - List all registered models
- `GET /models/{model_id}` - Get model details
- `GET /modules` - List available modules

### Training

- `POST /train` - Start training job
- `GET /status/{job_id}` - Get training job status

### Monitoring

- `POST /drift/check` - Check for data drift
- `GET /drift/history/{model_id}` - Get drift history
- `GET /monitoring/performance/{model_id}` - Get performance metrics
- `GET /monitoring/trend/{model_id}/{metric_name}` - Get metric trends

### Retraining

- `POST /retrain` - Trigger model retraining
- `GET /retrain/should/{model_id}` - Check if retraining needed

### Hyperparameters

- `GET /hyperparameters/schema` - Get hyperparameter schema
- `GET /hyperparameters/presets` - Get available presets
- `GET /hyperparameters/preset/{preset_name}` - Get specific preset
- `POST /hyperparameters/validate` - Validate hyperparameters

## Configuration

### Hyperparameter Presets

Edit `config/hyperparameters.py` to customize:

```python
PRESETS = {
    "small": {...},   # Lightweight, fast training
    "medium": {...},  # Balanced
    "large": {...},   # High accuracy, slower
    "fast": {...},    # Minimal latency
}
```

### Model Selection

Modify `automl/model_selector.py` to change model selection logic:

```python
def select_model(self, task: str, data_shape: Dict) -> str:
    # Return model name based on task and data characteristics
```

## Known Limitations

1. **NLTK Data**: First run will download NLTK data automatically
2. **GPU Memory**: Batch sizes adapt to available GPU memory
3. **Model Registry**: Stores metadata in `./model_registry/`
4. **Performance**: Large datasets may consume significant disk space

## Troubleshooting

### API Connection Error

```
API Server: Disconnected
```

- Ensure API is running on http://localhost:8000
- Check if port 8000 is already in use: `netstat -ano | findstr :8000`

### NLTK Module Missing

```
ModuleNotFoundError: No module named 'nltk'
```

- Run: `pip install nltk`
- NLTK data will be downloaded on first use

### CUDA/GPU Issues

- API will automatically fall back to CPU if CUDA unavailable
- Check in API health check response: `"gpu_available": true/false`

### Memory Issues

- Reduce batch size in hyperparameters
- Use "small" preset for testing
- Process smaller datasets first

## Development

### Running Tests

```bash
python run_project_demo.py          # Verify all modules
pytest tests/                       # Run unit tests (if added)
```

### Adding New Models

1. Create new encoder in `modelss/encoders/`
2. Register in `automl/model_selector.py`
3. Update hyperparameter schema in `config/hyperparameters.py`

### Custom Pipeline Tasks

1. Create task function
2. Register with `Orchestrator.register_task()`
3. Execute with `Orchestrator.run_pipeline()`

## Architecture Flow

```
Data Input
    ↓
Schema Detection → Problem Type Inference
    ↓
Parallel Preprocessing:
├─→ Image: Resize → Normalize → Vectorize (224x224x3)
├─→ Text: Tokenize → Embed → Extract CLS token
└─→ Tabular: Impute → Scale/Encode → Vectorize

    ↓
Model Selection & Encoding:
├─→ ImageEncoder (ResNet/MobileNet/ViT)
├─→ TextEncoder (BERT/RoBERTa/DistilBERT)
└─→ TabularEncoder (MLP/TabNet)

    ↓
Feature Fusion:
├─→ Concatenation or
└─→ Attention-based Fusion

    ↓
Prediction Head:
Linear → ReLU → Dropout → Output

    ↓
Monitoring:
├─→ Performance Tracking
├─→ Drift Detection
└─→ Retraining Decision
```

## Performance Benchmarks

Expected processing times (CPU):

- Image encoding: ~0.1-0.5s per batch
- Text encoding: ~0.05-0.2s per batch
- Tabular encoding: ~0.01-0.05s per batch
- Fusion & prediction: ~0.01-0.05s per batch

With GPU acceleration (CUDA):

- 3-5x faster processing
- Larger batch sizes supported

## Citation

If you use APEX in your research, please cite:

```bibtex
@software{apex2024,
  title={APEX: Advanced Predictive Ensemble with eXtendable modularity},
  version={0.1.0},
  year={2024}
}
```

## License

This project is provided as-is for educational and research purposes.

## Contributors

- Abhiram - Initial framework design

## Support

For issues or questions:

1. Check the troubleshooting section
2. Review API documentation at http://localhost:8000/docs
3. Check system status endpoint for detailed diagnostics

---

**Last Updated**: February 10, 2026
**Status**: ✅ Fully Functional
