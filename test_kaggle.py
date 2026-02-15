#!/usr/bin/env python
"""Test Kaggle dataset ingestion"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data_ingestion.ingestion_manager import DataIngestionManager

# Test Kaggle dataset
kaggle_url = "https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset"

print(f"🧪 Testing Kaggle Dataset Ingestion...")
print(f"Dataset: {kaggle_url}")
print("-" * 60)

manager = DataIngestionManager()

def progress_callback(progress, message):
    print(f"[{progress:3d}%] {message}")

print("\n📥 Downloading from Kaggle...")
loaded_data, metadata = manager.ingest_data(
    kaggle_url,
    progress_callback=progress_callback
)

print("\n✅ Ingestion Complete!")
print(f"Metadata: {metadata}")
print(f"\nLoaded data keys: {list(loaded_data.keys())}")

for key, data in loaded_data.items():
    if data is not None:
        print(f"\n📊 Dataset {key}:")
        print(f"  Shape: {data.shape}")
        print(f"  Columns: {list(data.columns[:5])}...")
        print(f"  First row:\n{data.iloc[0] if len(data) > 0 else 'Empty'}")
    else:
        print(f"❌ Dataset {key}: Failed to load")

print("\n📍 Cache location:")
import json
cache_meta = Path("./data/dataset_cache/cache_metadata.json")
if cache_meta.exists():
    with open(cache_meta) as f:
        metadata = json.load(f)
    print(f"  {metadata}")
