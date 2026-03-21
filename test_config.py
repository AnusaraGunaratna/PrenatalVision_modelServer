import os
import sys

# add current dir to path
sys.path.append(os.getcwd())

from app.config import Config
from app.infrastructure.model_loader import model_manager

print("Starting to load models with new Config attributes...")
try:
    model_manager.load_models(Config)
    print("CRL Models successfully loaded:", list(model_manager.get_models_for_task("CRL").keys()))
    print("NT Models successfully loaded:", list(model_manager.get_models_for_task("NT").keys()))
except AttributeError as e:
    print(f"FAILED: {e}")
except Exception as e:
    print(f"ANOTHER ERROR: {e}")
