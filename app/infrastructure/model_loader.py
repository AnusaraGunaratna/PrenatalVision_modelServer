import os
import sys
import logging
import functools
import torch
from ultralytics import YOLO
from typing import Dict, Any
from app.infrastructure.coordinate_attention import CoordinateAttention
from app.infrastructure.learnable_despeckling import LearnableDespeckling
from app.utils.model_downloader import ModelDownloader

_original_torch_load = torch.load
@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import __main__
__main__.CoordinateAttention = CoordinateAttention
__main__.LearnableDespeckling = LearnableDespeckling

import ultralytics.nn.modules as _modules
import ultralytics.nn.tasks as _tasks
_modules.CoordinateAttention = CoordinateAttention
_tasks.CoordinateAttention = CoordinateAttention
_modules.LearnableDespeckling = LearnableDespeckling
_tasks.LearnableDespeckling = LearnableDespeckling

if hasattr(_modules, 'conv'):
    _modules.conv.CoordinateAttention = CoordinateAttention
    _modules.conv.LearnableDespeckling = LearnableDespeckling
if hasattr(_modules, 'block'):
    _modules.block.CoordinateAttention = CoordinateAttention
    _modules.block.LearnableDespeckling = LearnableDespeckling

logger = logging.getLogger(__name__)

class ModelManager:
    """Singleton manager to load and cache YOLO models."""
    _instance = None
    _crl_models: Dict[str, Any] = {}
    _nt_models: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._crl_models = {}
            cls._instance._nt_models = {}
        return cls._instance

    def get_model(self, task: str, model_name: str, config) -> Any:
        """Lazy loading."""
        cache = self._nt_models if task.lower() == 'nt' else self._crl_models
        
        if model_name in cache:
            return cache[model_name]

        # Define paths mapping
        paths = {
            "CRL": {
                "PV-Hybrid": config.MODEL_PATH_CRL_HYBRID,
                "PV-Coord": config.MODEL_PATH_CRL_PVNET,
                "PV-LDB": config.MODEL_PATH_CRL_LDB,
                "YOLO8": config.MODEL_PATH_CRL_YOLO8,
                "YOLO11": config.MODEL_PATH_CRL_YOLO11,
            },
            "NT": {
                "PV-Hybrid": config.MODEL_PATH_NT_HYBRID,
                "PV-Coord": config.MODEL_PATH_NT_PVNET,
                "PV-LDB": config.MODEL_PATH_NT_LDB,
                "YOLO8": config.MODEL_PATH_NT_YOLO8,
                "YOLO11": config.MODEL_PATH_NT_YOLO11,
            }
        }

        task_key = task.upper()
        if task_key not in paths or model_name not in paths[task_key]:
            logger.error(f"Unknown model request: {task_key} {model_name}")
            return None

        path = paths[task_key][model_name]
        
        # Ensure model is available
        if not os.path.exists(path):
            logger.info(f"Model missing locally. Triggering on-demand download for: {model_name}")
            downloader = ModelDownloader(config.AZURE_STORAGE_CONNECTION_STRING, config.MODEL_CONTAINER_NAME)
            downloader.download_models()

        if not os.path.exists(path):
            logger.error(f"Model path still not found after download for {task_key} {model_name}: {path}")
            return None

        try:
            logger.info(f"Lazily loading {task_key} {model_name} from {path}...")
            model = YOLO(path)
            cache[model_name] = model
            return model
        except Exception as e:
            logger.error(f"Failed to load {task_key} {model_name} from {path}: {str(e)}")
            return None

    def load_models(self, config):
        """Lazy loading"""
        logger.info("Model Manager initialized (Lazy Loading enabled).")

model_manager = ModelManager()
