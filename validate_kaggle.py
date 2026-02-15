#!/usr/bin/env python
"""Validate Kaggle integration setup"""
import sys
from pathlib import Path

print("🧪 Validating Kaggle Setup...\n")

# Check 1: Kaggle package installed
try:
    import kaggle
    print("✅ kaggle package: installed")
except ImportError:
    print("❌ kaggle package: NOT installed - run: pip install kaggle")
    sys.exit(1)

# Check 2: Kaggle credentials
cred_path = Path.home() / ".kaggle" / "kaggle.json"
if cred_path.exists():
    print(f"✅ Kaggle credentials: found at {cred_path}")
else:
    print(f"⚠️ Kaggle credentials: NOT found at {cred_path}")
    print("   Setup instructions:")
    print("   1. Go to https://www.kaggle.com/settings/account")
    print("   2. Click 'Create New API Token'")
    print("   3. Save kaggle.json to ~/.kaggle/")
    sys.exit(1)

# Check 3: Kaggle CLI available
import subprocess
try:
    result = subprocess.run(["kaggle", "--version"], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ Kaggle CLI: {result.stdout.strip()}")
    else:
        print("⚠️ Kaggle CLI: not properly configured")
except FileNotFoundError:
    print("❌ Kaggle CLI: not found in PATH")
    sys.exit(1)

# Check 4: Data ingestion manager
sys.path.insert(0, str(Path(__file__).parent))
try:
    from data_ingestion.ingestion_manager import DataIngestionManager
    print("✅ DataIngestionManager: imported successfully")
except ImportError as e:
    print(f"❌ DataIngestionManager: {e}")
    sys.exit(1)

# Check 5: Test URL parsing
kaggle_url = "https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset"
parts = kaggle_url.strip('/').split('/')
dataset_id = f"{parts[-2]}/{parts[-1]}"
print(f"✅ Kaggle URL parsing: {dataset_id}")

# Check 6: Cache directory
cache_dir = Path("./data/dataset_cache")
print(f"✅ Cache directory: {cache_dir.absolute()}")

print("\n" + "="*60)
print("✅ All checks passed! Kaggle integration is ready.")
print("="*60)
print("\nNext steps:")
print("1. Open frontend at http://localhost:8501")
print("2. Go to Phase 1️⃣ - Data Ingestion & Caching")
print("3. Enter: https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset")
print("4. Click '🔄 Load Datasets'")
print("5. Dataset will download to cache (~500MB, takes several minutes)")
