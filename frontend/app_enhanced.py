"""APEX AutoML Frontend - Comprehensive Multimodal ML Platform with Workflow Integration."""

import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


# Page configuration
st.set_page_config(
    page_title="APEX AutoML",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .phase-header {background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;}
    .phase-status {font-size: 14px; color: #666;}
    .success-badge {background-color: #d4edda; padding: 10px; border-radius: 5px; margin: 10px 0;}
    .info-badge {background-color: #d1ecf1; padding: 10px; border-radius: 5px; margin: 10px 0;}
    .warning-badge {background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0;}
    </style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8001"

def check_api_connection():
    """Check if API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_api_status():
    """Get full API status."""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=2)
        return response.json() if response.status_code == 200 else None
    except:
        return None

# Session state initialization
if 'workflow_stage' not in st.session_state:
    st.session_state.workflow_stage = 1
if 'dataset_uploaded' not in st.session_state:
    st.session_state.dataset_uploaded = False
if 'schema_detected' not in st.session_state:
    st.session_state.schema_detected = False
if 'detected_schema' not in st.session_state:
    st.session_state.detected_schema = None
if 'model_selected' not in st.session_state:
    st.session_state.model_selected = False
if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = {}
if 'ingested_row_count' not in st.session_state:
    st.session_state.ingested_row_count = None

def render_workflow_dashboard():
    """Render the main workflow dashboard."""
    st.title("🚀 APEX AutoML Framework")
    st.markdown("**Advanced Predictive Ensemble with eXtendable Modularity**\n---")
    
    # API Status Section
    col1, col2, col3 = st.columns(3)
    with col1:
        api_status = get_api_status()
        if api_status:
            st.success("✅ API Connected")
            st.caption(f"v{api_status.get('version', 'N/A')}")
        else:
            st.error("❌ API Disconnected")
            st.caption("Make sure API server is running on http://localhost:8001")
    
    with col2:
        st.info("📊 Workflow: Multi-Phase Training")
        st.caption(f"Current Stage: Phase {st.session_state.workflow_stage}")
    
    with col3:
        gpu_available = api_status.get('gpu_available', False) if api_status else False
        if gpu_available:
            st.success("✅ GPU Available")
        else:
            st.warning("⚠️ CPU Mode")
    
    st.divider()
    
    # Workflow Progress
    st.subheader("📋 Workflow Stages")
    
    stages = [
        ("Phase 1", "Data Ingestion", "Upload & cache datasets"),
        ("Phase 2", "Schema Detection", "Detect columns & problem type"),
        ("Phase 3", "Preprocessing", "Prepare data for training"),
        ("Phase 4", "Model Selection", "Auto-select models & params"),
        ("Phase 5", "Training", "Train with GPU acceleration"),
        ("Phase 6", "Monitoring", "Track performance & drift"),
        ("Phase 7", "Prediction", "Make multimodal predictions")
    ]
    
    cols = st.columns(7)
    for i, (col, (phase, name, desc)) in enumerate(zip(cols, stages)):
        with col:
            if st.session_state.workflow_stage == i + 1:
                st.markdown(f"### 🔵 {phase}\n**{name}**")
            elif st.session_state.workflow_stage > i + 1:
                st.markdown(f"### ✅ {phase}\n**{name}**")
            else:
                st.markdown(f"### ⭕ {phase}\n{name}")
            st.caption(desc)
    
    st.divider()
    
    # Phase Selection
    phase = st.radio(
        "Select Workflow Phase:",
        ["Phase 1: Data Ingestion", "Phase 2: Schema Detection", "Phase 3: Preprocessing",
         "Phase 4: Model Selection", "Phase 5: Training", "Phase 6: Monitoring", "Phase 7: Prediction"],
        index=st.session_state.workflow_stage - 1,
        horizontal=True
    )
    
    st.session_state.workflow_stage = int(phase.split()[1].rstrip(':'))
    
    # Render selected phase
    if st.session_state.workflow_stage == 1:
        render_phase_1_data_ingestion()
    elif st.session_state.workflow_stage == 2:
        render_phase_2_schema_detection()
    elif st.session_state.workflow_stage == 3:
        render_phase_3_preprocessing()
    elif st.session_state.workflow_stage == 4:
        render_phase_4_model_selection()
    elif st.session_state.workflow_stage == 5:
        render_phase_5_training()
    elif st.session_state.workflow_stage == 6:
        render_phase_6_monitoring()
    elif st.session_state.workflow_stage == 7:
        render_phase_7_prediction()


def render_phase_1_data_ingestion():
    """Phase 1: Data Ingestion with Caching."""
    st.header("Phase 1️⃣ - Data Ingestion & Caching")
    
    st.markdown("""
    **Workflow:**
    1. Provide dataset sources (Kaggle URLs, HTTP links, or local paths)
    2. System generates SHA-256 hash for each source
    3. Check cache for existing data
    4. Download and validate if not cached
    5. Store in local cache for future use
    """)
    
    st.markdown("### 📥 Dataset Sources")
    dataset_sources = st.text_area(
        "Enter one or more dataset sources (Kaggle URL, HTTP link, or local path), one per line:",
        placeholder="https://kaggle.com/datasets/...\nhttps://example.com/data.csv\n/path/to/dataset.csv",
        height=120
    )
    dataset_sources = [s.strip() for s in dataset_sources.splitlines() if s.strip()]
    st.markdown("### 💾 Cache Info")
    st.info("""
    - **Caching**: Automatic SHA-256 hashing
    - **Storage**: `data/dataset_cache/`
    - **Formats**: CSV, Parquet, JSON
    - **Hit Ratio**: Skip download if cached
    """)
    
    st.divider()
    
    # Upload/Download section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Load Datasets", use_container_width=True):
            if not check_api_connection():
                st.error("API not connected!")
            else:
                st.markdown("### 📥 Dataset Ingestion Progress")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                info_container = st.container()
                
                total_datasets = len(dataset_sources)
                
                # Call backend to ingest datasets
                st.write("**Starting real data ingestion...**")
                import time
                
                try:
                    # Start the ingestion request
                    ingestion_done = False
                    result = None
                    
                    # Fire off the ingestion request
                    response = requests.post(
                        f"{API_BASE_URL}/ingest/datasets",
                        json={"sources": dataset_sources},
                        timeout=600
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        ingestion_done = True
                    
                    # Poll for progress every second until done or timeout
                    if not ingestion_done:
                        last_progress = 0
                        for attempt in range(300):  # 5 minutes polling
                            try:
                                status_resp = requests.get(
                                    f"{API_BASE_URL}/ingest/status",
                                    timeout=5
                                )
                                
                                if status_resp.status_code == 200:
                                    status_data = status_resp.json()
                                    
                                    # Update progress bar
                                    current_progress = status_data.get("progress", 0) / 100
                                    if current_progress > last_progress:
                                        progress_bar.progress(min(current_progress, 0.99))
                                        last_progress = current_progress
                                    
                                    # Update status message
                                    message = status_data.get("message", "Processing...")
                                    current_source = status_data.get("current_source", "")
                                    status_text.write(f"📍 {current_source} - {message}")
                                    
                                    # Check if completed
                                    if status_data.get("status") == "completed":
                                        result = status_data
                                        ingestion_done = True
                                        break
                            
                            except:
                                pass  # Continue polling
                            
                            time.sleep(1)
                    
                    # Display results - simplified
                    if result:
                        progress_bar.progress(1.0)
                        st.markdown("---")
                        
                        # Simple status display
                        datasets_list = result.get("datasets", [])
                        
                        if datasets_list:
                            for idx, dataset_info in enumerate(datasets_list, 1):
                                source = dataset_info.get("source", "Unknown")
                                status = dataset_info.get("status", "Unknown")
                                
                                # Show very simple status
                                if "Cached" in status:
                                    st.success(f"✅ **Dataset {idx}** - Already Cached\n📍 {source[:60]}")
                                elif "Success" in status:
                                    st.success(f"✅ **Dataset {idx}** - Downloaded Successfully\n📍 {source[:60]}")
                                else:
                                    st.error(f"❌ **Dataset {idx}** - Failed to Load\n📍 {source[:60]}")
                        else:
                            st.warning("No datasets to display")
                        
                        # Update session state
                        st.session_state.dataset_uploaded = True
                        
                        # Extract dataset info for later phases
                        dataset_shapes = []
                        for ds_info in result.get("datasets", []):
                            if "shape" in ds_info and ("Cached" in ds_info.get("status", "") or "Success" in ds_info.get("status", "")):
                                dataset_shapes.append(ds_info["shape"])
                        
                        row_count = dataset_shapes[0][0] if dataset_shapes else 5000
                        
                        st.session_state.dataset_info = {
                            "sources": dataset_sources,
                            "count": result.get("successful", 0),
                            "shapes": dataset_shapes,
                            "row_count": row_count,
                            "timestamp": datetime.now().isoformat()
                        }
                        st.session_state.ingested_row_count = row_count
                    else:
                        st.error("Ingestion failed or timed out")
                
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out (>10 minutes)")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    
    with col2:
        if st.button("📊 View Cache", use_container_width=True):
            try:
                response = requests.get(f"{API_BASE_URL}/cache/stats", timeout=5)
                if response.status_code == 200:
                    cache_info = response.json()
                    st.success(f"✅ Cache Location: {cache_info['cache_location']}")
                    
                    col_c1, col_c2, col_c3 = st.columns(3)
                    col_c1.metric("Cached Items", cache_info['total_items'])
                    col_c2.metric("Total Size (MB)", cache_info['total_size_mb'])
                    col_c3.metric("Status", "Ready")
                    
                    if cache_info['items']:
                        with st.expander("📁 Cached Files"):
                            for item in cache_info['items']:
                                st.caption(f"• {item['name']} ({item['size_mb']} MB)")
                else:
                    st.info(f"Cache storage: {API_BASE_URL}/cache/stats")
            except Exception as e:
                st.info(f"Cache storage: ~/data/dataset_cache/")
    
    with col3:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            try:
                response = requests.post(f"{API_BASE_URL}/cache/clear", timeout=5)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ {result['message']}")
                else:
                    st.error(f"Failed to clear cache: {response.status_code}")
            except Exception as e:
                st.error(f"Cache clear error: {str(e)}")
    
    # Status display
    if st.session_state.dataset_uploaded:
        st.markdown("### ✅ Status")
        col1, col2, col3 = st.columns(3)
        col1.metric("Datasets Loaded", st.session_state.dataset_info.get("count", 0))
        col2.metric("Cache Location", "./data/dataset_cache/")
        col3.metric("Last Updated", "Just now")
        
        if st.button("➡️ Next: Schema Detection", use_container_width=True):
            st.session_state.workflow_stage = 2
            st.rerun()


def render_phase_2_schema_detection():
    """Phase 2: Schema Detection."""
    st.header("Phase 2️⃣ - Schema Detection & Problem Type Inference")
    
    st.markdown("""
    **Schema Detection for Ingested Datasets:**
    - Analyzes ONLY the datasets loaded in Phase 1
    - Detects 3 modalities: Image, Text, Tabular
    - Identifies target column automatically
    - Infers problem type: Classification / Regression / Unsupervised
    """)
    
    if not st.session_state.dataset_uploaded:
        st.warning("⚠️ Please load datasets in Phase 1 first")
        if st.button("← Go to Phase 1"):
            st.session_state.workflow_stage = 1
            st.rerun()
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔍 Detect Schema", use_container_width=True):
            with st.spinner("Analyzing ingested datasets..."):
                progress_bar = st.progress(0)
                status = st.empty()
                
                try:
                    status.write("📍 Detecting schema for ingested datasets...")
                    
                    # Backend now uses globally tracked ingested datasets
                    response = requests.post(
                        f"{API_BASE_URL}/detect-schema",
                        json={},  # No need to pass sources - backend uses last ingestion
                        timeout=120
                    )
                    progress_bar.progress(0.5)
                    
                    if response.status_code == 200:
                        detected = response.json()
                        st.session_state.detected_schema = detected
                        progress_bar.progress(1.0)
                        st.success("✅ Schema detection complete!")
                        st.session_state.schema_detected = True
                        st.rerun()
                    else:
                        error_data = response.json() if response.headers.get('content-type') == 'application/json' else {"error": response.text}
                        st.error(f"❌ Detection failed: {error_data.get('error', 'Unknown error')}")
                        with st.expander("🔍 Debug Info"):
                            st.json(error_data)
                except requests.exceptions.Timeout:
                    st.error("❌ Timeout after 120 seconds")
                except Exception as e:
                    st.error(f"❌ Detection error: {str(e)}")
                    with st.expander("🔍 Debug Info"):
                        st.code(str(e))
    
    with col2:
        if st.checkbox("Show Detection Details"):
            st.info("📊 Verbose mode: See detailed detection output")
    
    # Display results if detection succeeded
    if st.session_state.schema_detected and st.session_state.detected_schema:
        st.divider()
        st.markdown("### 📋 Detected Schema Results")
        schema_data = st.session_state.detected_schema.get("data", {})
        detected_columns = schema_data.get("detected_columns", {})
        modalities = schema_data.get("modalities", [])
        target_column = schema_data.get("target_column", "Unknown")
        problem_type = schema_data.get("problem_type", "Unknown")
        confidence = schema_data.get("detection_confidence", 0)
        # Show summary
        col1, col2, col3 = st.columns(3)
        col1.metric("Target Column", target_column)
        col2.metric("Problem Type", problem_type)
        col3.metric("Confidence", f"{confidence:.1%}")
        st.info(f"**Modalities Found**: {', '.join(modalities) if modalities else 'None'}")
        # Show detailed schema
        tab1, tab2 = st.tabs(["Summary", "Debug Info"])
        with tab1:
            st.markdown("#### 📊 Columns by Modality")
            for modality, cols in detected_columns.items():
                st.write(f"**{modality.title()} Columns:** {', '.join(cols) if cols else 'None'}")
            st.markdown(f"**All Modalities:** {', '.join(detected_columns.keys())}")
            st.markdown(f"**Target Column:** {target_column}")
        with tab2:
            st.markdown("#### 🔍 Raw Schema Detection Response")
            st.json(schema_data)
        st.divider()
        if st.button("➡️ Next: Preprocessing", use_container_width=True):
            st.session_state.workflow_stage = 3
            st.rerun()
    elif st.session_state.schema_detected:
        st.warning("⚠️ Schema detection ran but no results available")


def render_phase_3_preprocessing():
    """Phase 3: Preprocessing Pipeline."""
    st.header("Phase 3️⃣ - Data Preprocessing")
    
    if not st.session_state.schema_detected:
        st.warning("⚠️ Please detect schema in Phase 2 first")
        return
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("⚙️ Start Preprocessing", use_container_width=True):
            with st.spinner("Preprocessing data..."):
                # Call backend for real preprocessing
                schema_data = st.session_state.detected_schema.get("data", {})
                response = requests.post(
                    f"{API_BASE_URL}/preprocess",
                    json={
                        "data_path": schema_data.get("data_path"),
                        "image_columns": schema_data.get("detected_columns", {}).get("image", []),
                        "text_columns": schema_data.get("detected_columns", {}).get("text", []),
                        "tabular_columns": schema_data.get("detected_columns", {}).get("tabular", [])
                    },
                    timeout=300
                )
                if response.status_code == 200:
                    result = response.json().get("data", {})
                    st.session_state.preprocess_result = result
                    st.success("✅ Preprocessing complete!")
                else:
                    st.error("❌ Preprocessing failed")
    with col2:
        st.markdown("### Phase Status")
        st.info("Real-time preprocessing updates shown above")
    st.divider()
    st.markdown("### 📊 Preprocessing Summary")
    preprocess_result = st.session_state.get("preprocess_result", {})
    stages = preprocess_result.get("preprocessing_stages", [])
    total_samples = preprocess_result.get("total_samples", None)
    output_shapes = preprocess_result.get("output_shapes", {})
    if stages:
        for stage in stages:
            col1, col2, col3 = st.columns(3)
            col1.metric(stage.get("stage", "Stage"), total_samples or "?")
            col2.write(f"**Status:** {stage.get('status', 'N/A')}")
            col3.code(stage.get("output_shape", "?"))
    else:
        st.info("No preprocessing results yet. Click 'Start Preprocessing' above.")
    if st.button("➡️ Next: Model Selection"):
        st.session_state.workflow_stage = 4
        st.model_selected = False
        st.rerun()


def render_phase_4_model_selection():
    """Phase 4: Model Selection with Rationale."""
    st.header("Phase 4️⃣ - Automatic Model Selection")
    
    st.markdown("""
    **Selection Criteria:**
    - GPU Memory Check: <6GB (Lightweight) | 6-12GB (Medium) | >12GB (Large)
    - Dataset Size: <5k (45 epochs) | 5-50k (18 epochs) | >50k (6 epochs)
    - Task Complexity: Binary/Multiclass/Regression
    """)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("🤖 Select Models", use_container_width=True):
            with st.spinner("Selecting models and hyperparameters..."):
                schema_data = st.session_state.detected_schema.get("data", {})
                dataset_size = st.session_state.get('ingested_row_count', 1000)
                modalities = schema_data.get("modalities", [])
                problem_type = schema_data.get("problem_type", "Unknown")
                response = requests.post(
                    f"{API_BASE_URL}/select-model",
                    json={
                        "dataset_size": dataset_size,
                        "modalities": modalities,
                        "problem_type": problem_type
                    },
                    timeout=120
                )
                if response.status_code == 200:
                    result = response.json().get("data", {})
                    st.session_state.model_selection_result = result
                    st.success("✅ Models selected!")
                    st.session_state.model_selected = True
                else:
                    st.error("❌ Model selection failed")
    with col2:
        st.markdown("### Hardware Detection")
        api_status = get_api_status()
        if api_status:
            device_type = "🔴 GPU - RTX 3070 Ti" if api_status.get('gpu_available') else "🔵 CPU"
            st.metric("Device", device_type)
            st.caption("✅ CUDA 11.8 Ready" if api_status.get('gpu_available') else "Running on CPU")
    st.divider()
    if st.session_state.model_selected:
        st.markdown("### 🎯 Selected Model & Hyperparameters")
        result = st.session_state.get("model_selection_result", {})
        st.write(f"**Selected Model:** {result.get('selected_model', 'N/A')}")
        st.write(f"**Encoders:** {result.get('selected_encoders', {})}")
        st.write(f"**Hyperparameters:** {result.get('hyperparameters', {})}")
        st.info(result.get('selection_rationale', ''))
        if st.button("➡️ Next: Training"):
            st.session_state.workflow_stage = 5
            st.rerun()


def render_phase_5_training():
    """Phase 5: GPU Training."""
    st.header("Phase 5️⃣ - GPU Training")
    
    st.markdown("""
    **Safety Mechanisms:**
    - CUDA required (no CPU fallback)
    - torch.cuda.synchronize() for Windows WDDM safety
    - Per-batch: Forward → Loss → Backward → Update → Sync
    """)
    
    if not st.session_state.model_selected:
        st.warning("⚠️ Select models in Phase 4 first")
        return
    
    col1, col2 = st.columns([2, 1])
    with col1:
        start_training = st.button("🚀 Start Training", use_container_width=True)
        if start_training:
            st.markdown("### 📈 Training Progress")
            with st.spinner("Training model on GPU..."):
                # Prepare payload from selected model/hyperparameters
                model_info = st.session_state.get("model_selection_result", {})
                schema_data = st.session_state.detected_schema.get("data", {})
                dataset_size = st.session_state.get('ingested_row_count', 1000)
                payload = {
                    "selected_model": model_info.get("selected_model"),
                    "selected_encoders": model_info.get("selected_encoders"),
                    "hyperparameters": model_info.get("hyperparameters"),
                    "schema": schema_data,
                    "dataset_size": dataset_size
                }
                response = requests.post(
                    f"{API_BASE_URL}/train-pipeline",
                    json=payload,
                    timeout=600
                )
                if response.status_code == 200:
                    result = response.json().get("data", {})
                    st.session_state.training_result = result
                    st.success("✅ Training Complete!")
                    # Show results
                    st.markdown("### 📊 Training Metrics")
                    metrics = result.get("metrics", {})
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Final Loss", f"{metrics.get('final_loss', 'N/A')}")
                    col2.metric("Best Val Acc", f"{metrics.get('best_val_acc', 'N/A')}")
                    col3.metric("Training Time", metrics.get('training_time', 'N/A'))
                    st.write("#### Full Metrics:")
                    st.json(metrics)
                else:
                    st.error("❌ Training failed: " + response.text)
    with col2:
        st.markdown("### 💾 Model Info")
        model_info = st.session_state.get("model_selection_result", {})
        st.info(f"""
        - **Status**: Ready to train
        - **Device**: GPU
        - **Batch Size**: {model_info.get('hyperparameters', {}).get('Batch Size', 'N/A')}
        - **Epochs**: {model_info.get('hyperparameters', {}).get('Epochs', 'N/A')}
        """)
    if st.button("➡️ Next: Monitoring"):
        st.session_state.workflow_stage = 6
        st.rerun()


def render_phase_6_monitoring():
    """Phase 6: Monitoring & Drift Detection."""
    st.header("Phase 6️⃣ - Model Monitoring & Drift Detection")
    st.markdown("""
    **Drift Thresholds:**
    - PSI (Prediction Stability Index): > 0.25
    - KS (Kolmogorov-Smirnov): > 0.3
    - FDD (Feature/Embedding Drift): > 0.5
    """)
    tab1, tab2, tab3 = st.tabs(["Performance", "Drift Detection", "Model Registry"])
    with tab1:
        st.markdown("### 📊 Performance Metrics")
        with st.spinner("Fetching performance metrics..."):
            response = requests.get(f"{API_BASE_URL}/monitor/performance", timeout=60)
            if response.status_code == 200:
                metrics = response.json().get("data", {})
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", metrics.get("accuracy", "N/A"))
                col2.metric("F1 Score", metrics.get("f1_score", "N/A"))
                col3.metric("Precision", metrics.get("precision", "N/A"))
                col4.metric("Recall", metrics.get("recall", "N/A"))
                st.markdown("### 📈 Performance Trends")
                trends = metrics.get("trends")
                if trends:
                    trends_df = pd.DataFrame(trends)
                    st.line_chart(trends_df.set_index('date'))
                else:
                    st.info("No trend data available.")
            else:
                st.error("Failed to fetch performance metrics.")
    with tab2:
        st.markdown("### 🔍 Drift Detection Results")
        with st.spinner("Fetching drift detection results..."):
            response = requests.get(f"{API_BASE_URL}/monitor/drift", timeout=60)
            if response.status_code == 200:
                drift = response.json().get("data", {})
                drift_df = pd.DataFrame(drift.get("drift_metrics", []))
                st.dataframe(drift_df, use_container_width=True)
                st.info(drift.get("drift_status", "No drift status available."))
            else:
                st.error("Failed to fetch drift detection results.")
    with tab3:
        st.markdown("### 📦 Model Registry")
        with st.spinner("Fetching model registry info..."):
            response = requests.get(f"{API_BASE_URL}/model-registry", timeout=60)
            if response.status_code == 200:
                registry = response.json().get("data", {})
                models_df = pd.DataFrame(registry.get("models", []))
                st.dataframe(models_df, use_container_width=True)
            else:
                st.error("Failed to fetch model registry info.")
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Start New Workflow"):
            st.session_state.workflow_stage = 1
            st.session_state.dataset_uploaded = False
            st.session_state.schema_detected = False
            st.session_state.model_selected = False
            st.rerun()
    with col2:
        st.button("📥 Download Model")
    with col3:
        st.button("🚀 Deploy to Production")


def render_phase_7_prediction():
    """Phase 7: Multimodal Prediction."""
    st.header("Phase 7️⃣ - Make Predictions")
    
    st.markdown("""
    **Multimodal Prediction Engine:**
    - Accept image, text, and tabular inputs
    - Use trained model from Phase 5
    - Combine all modalities for prediction
    - Output prediction with confidence score
    """)
    
    if not st.session_state.schema_detected:
        st.warning("⚠️ Please run schema detection in Phase 2 first")
        return
    
    schema_data = st.session_state.detected_schema or {}
    modalities = schema_data.get("modalities_found", [])
    target_col = schema_data.get("target_column", "Unknown")
    
    st.info(f"**Target Column**: {target_col} | **Modalities**: {', '.join(modalities)}")
    
    # Input tabs based on detected modalities
    tabs = []
    if "image" in modalities:
        tabs.append("Image Input")
    if "text" in modalities:
        tabs.append("Text Input")
    if "tabular" in modalities:
        tabs.append("Tabular Input")
    
    if not tabs:
        st.warning("⚠️ No multimodal inputs available")
        return
    
    tab_contents = st.tabs(tabs + ["Summary & Predict"])
    
    input_data = {}
    
    # Process each tab
    tab_idx = 0
    
    if "image" in modalities:
        with tab_contents[tab_idx]:
            st.markdown("### 📷 Image Input")
            image_option = st.radio("Image source:", ["Upload", "URL"], key="image_source")
            
            if image_option == "Upload":
                uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
                if uploaded_image:
                    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
                    input_data["image_data"] = uploaded_image.getvalue()
            else:
                image_url = st.text_input("Image URL:")
                if image_url:
                    input_data["image_data"] = image_url
                    st.image(image_url, caption="Image from URL", use_column_width=True)
        
        tab_idx += 1
    
    if "text" in modalities:
        with tab_contents[tab_idx]:
            st.markdown("### 📝 Text Input")
            text_input = st.text_area("Enter text:", height=200)
            if text_input:
                input_data["text_data"] = text_input
                st.info(f"Text length: {len(text_input)} characters")
        
        tab_idx += 1
    
    if "tabular" in modalities:
        with tab_contents[tab_idx]:
            st.markdown("### 📊 Tabular Input")
            
            col1, col2 = st.columns(2)
            with col1:
                feature1 = st.number_input("Feature 1:", value=0.0)
            with col2:
                feature2 = st.number_input("Feature 2:", value=0.0)
            
            col3, col4 = st.columns(2)
            with col3:
                feature3 = st.number_input("Feature 3:", value=0.0)
            with col4:
                feature4 = st.selectbox("Category:", ["Class A", "Class B", "Class C"])
            
            input_data["tabular_data"] = {
                "feature_1": feature1,
                "feature_2": feature2,
                "feature_3": feature3,
                "category": feature4
            }
    
    # Summary and prediction
    with tab_contents[-1]:
        st.markdown("### 🔮 Make Prediction")
        
        st.write("**Inputs prepared:**")
        for key in input_data.keys():
            st.success(f"✅ {key.replace('_', ' ').title()}: Provided")
        
        if st.button("🚀 Generate Prediction", use_container_width=True):
            with st.spinner("Making prediction..."):
                try:
                    # Get model name from training or registry phase
                    model_info = st.session_state.get("training_result", {})
                    model_name = model_info.get("model_name") or model_info.get("model_id")
                    if not model_name:
                        model_name = st.session_state.get("model_selection_result", {}).get("selected_model", "default_model")
                    prediction_payload = {
                        "image_data": input_data.get("image_data"),
                        "text_data": input_data.get("text_data"),
                        "tabular_data": input_data.get("tabular_data"),
                        "model_name": model_name
                    }
                    # Remove None values for unused modalities
                    prediction_payload = {k: v for k, v in prediction_payload.items() if v is not None}
                    response = requests.post(
                        f"{API_BASE_URL}/predict",
                        json=prediction_payload,
                        timeout=60
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.divider()
                        st.success("✅ Prediction Generated!")
                        col1, col2 = st.columns(2)
                        col1.metric("Prediction", result.get("prediction", "N/A"))
                        col2.metric("Confidence", f"{result.get('confidence', 0):.1%}")
                        st.info(f"Model: {result.get('model_used', model_name)}")
                        # Show detailed results
                        with st.expander("📊 Detailed Results"):
                            st.json(result)
                    else:
                        st.error(f"❌ Prediction failed: {response.text}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    
    # Final actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 New Prediction"):
            st.rerun()
    
    with col2:
        st.button("📊 Batch Predict")
    
    with col3:
        st.button("💾 Save Results")


# Main execution
if __name__ == "__main__":
    # Check API connection in sidebar
    st.sidebar.markdown("### ⚙️ System Status")
    
    if check_api_connection():
        st.sidebar.success("✅ API Connected")
        api_info = get_api_status()
        if api_info:
            st.sidebar.caption(f"Version: {api_info.get('version', 'N/A')}")
            st.sidebar.caption(f"GPU: {'✅' if api_info.get('gpu_available') else '❌'}")
    else:
        st.sidebar.error("❌ API Disconnected")
        st.sidebar.caption("Start API: python run_api.py")
    
    # Sidebar navigation
    st.sidebar.divider()
    st.sidebar.markdown("### 📚 Documentation")
    
    if st.sidebar.button("📖 Workflow Guide"):
        st.sidebar.info("""
        **APEX AutoML Workflow:**
        1. Upload datasets (with caching)
        2. Auto-detect schema & columns
        3. Preprocess all modalities
        4. Select models intelligently
        5. Train with GPU acceleration
        6. Monitor performance & drift
        """)
    
    st.sidebar.markdown("### 🔗 Quick Links")
    st.sidebar.link_button("🌐 API Docs", "http://localhost:8001/docs")
    st.sidebar.link_button("📋 GitHub", "https://github.com/hrishi-cz/main-project")
    
    # Main workflow
    render_workflow_dashboard()

