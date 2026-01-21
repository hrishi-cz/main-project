"""Run API with correct Python path"""
import sys
from pathlib import Path

# Add apex directory to path
apex_dir = Path(__file__).parent
sys.path.insert(0, str(apex_dir))

# Now import and run
if __name__ == "__main__":
    from api.main_enhanced import app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)