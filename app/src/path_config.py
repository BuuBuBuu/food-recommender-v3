# path_config.py
import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Define common paths
DATA_DIR = PROJECT_ROOT / 'data'
APP_DIR = PROJECT_ROOT / 'app'
MODELS_DIR = APP_DIR / 'models'

# Specific file paths
TFIDF_VECTORIZER_PATH = DATA_DIR / 'tfidf_vectorizer.pkl'
PRODUCTION_MODEL_PATH = MODELS_DIR / 'production_ranking_model_enhanced.pkl'

# Create directories if they don't exist
def ensure_directories():
    """Ensure all required directories exist."""
    directories = [DATA_DIR, APP_DIR, MODELS_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Verify file existence
def verify_required_files():
    """Verify that only the required files for inference exist."""
    required_files = [
        (TFIDF_VECTORIZER_PATH, "TF-IDF Vectorizer"),
        (PRODUCTION_MODEL_PATH, "Production Model")
    ]

    missing_files = []
    for file_path, file_desc in required_files:
        if not file_path.exists():
            missing_files.append(f"{file_desc} not found at: {file_path}")

    if missing_files:
        raise FileNotFoundError("\n".join(missing_files))

def init_paths():
    """Initialize and verify all paths."""
    ensure_directories()
    verify_required_files()
