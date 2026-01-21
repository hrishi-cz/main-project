@echo off
echo ==========================================
echo AutoVision Project Setup
echo ==========================================

REM Create directory structure
echo Creating directory structure...

mkdir data_ingestion\adapters 2>nul
mkdir pipeline 2>nul
mkdir automl 2>nul
mkdir models\encoders 2>nul
mkdir preprocessing 2>nul
mkdir registry 2>nul
mkdir api 2>nul
mkdir frontend 2>nul
mkdir utils 2>nul
mkdir data\dataset_cache\kaggle 2>nul
mkdir data\dataset_cache\remote 2>nul
mkdir registry\models 2>nul

REM Create __init__.py files
echo Creating __init__.py files...

type nul > data_ingestion\__init__.py
type nul > data_ingestion\adapters\__init__.py
type nul > pipeline\__init__.py
type nul > automl\__init__.py
type nul > models\__init__.py
type nul > models\encoders\__init__.py
type nul > preprocessing\__init__.py
type nul > registry\__init__.py
type nul > api\__init__.py
type nul > frontend\__init__.py
type nul > utils\__init__.py

echo [OK] Directory structure created

REM Install dependencies
echo.
echo Installing dependencies...
echo This may take several minutes...

pip install --upgrade pip

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install scikit-learn pandas numpy transformers timm pillow fastapi uvicorn[standard] python-multipart pydantic streamlit plotly streamlit-autorefresh requests datasets scipy tqdm joblib openpyxl xlrd python-dateutil kaggle

echo [OK] Dependencies installed

echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo Next steps:
echo 1. Add your Python files to the directories
echo 2. Run: uvicorn api.main:app --reload
echo.
pause
