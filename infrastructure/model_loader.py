import os
import sys
import logging
import functools
import torch
from ultralytics import YOLO
from typing import Dict, Any
from app.infrastructure.coordinate_attention import CoordinateAttention

_original_torch_load = torch.load
@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import __main__
__main__.CoordinateAttention = CoordinateAttention

import ultralytics.nn.modules as _modules
import ultralytics.nn.tasks as _tasks
_modules.CoordinateAttention = CoordinateAttention
_tasks.CoordinateAttention = CoordinateAttention

if hasattr(_modules, 'conv'):
    _modules.conv.CoordinateAttention = CoordinateAttention
if hasattr(_modules, 'block'):
    _modules.block.CoordinateAttention = CoordinateAttention

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

    def load_models(self, config):
        crl_paths = {
            "Original YOLO11": config.MODEL_PATH_CRL_ORIGINAL_YOLO11,
            "Custom YOLO11": config.MODEL_PATH_CRL_CUSTOM_YOLO11,
            "Original YOLO8": config.MODEL_PATH_CRL_ORIGINAL_YOLO8,
            "Custom YOLO8 (P2)": config.MODEL_PATH_CRL_CUSTOM_YOLO8,
        }

        nt_paths = {
            "Original YOLO11": config.MODEL_PATH_NT_ORIGINAL_YOLO11,
            "Custom YOLO11": config.MODEL_PATH_NT_CUSTOM_YOLO11,
            "Original YOLO8": config.MODEL_PATH_NT_ORIGINAL_YOLO8,
            "Custom YOLO8 (P2)": config.MODEL_PATH_NT_CUSTOM_YOLO8,
        }

        self._load_group(crl_paths, self._crl_models, "CRL")
        self._load_group(nt_paths, self._nt_models, "NT")

    def _load_group(self, paths_dict, target_dict, group_name):
        for name, path in paths_dict.items():
            if os.path.exists(path):
                try:
                    logger.info(f"Loading {group_name} {name} from {path}...")
                    target_dict[name] = YOLO(path)
                except Exception as e:
                    logger.error(f"Failed to load {group_name} {name} from {path}: {str(e)}")
            else:
                logger.warning(f"Model path not found for {group_name} {name}: {path}")

    def get_models_for_task(self, task: str) -> Dict[str, Any]:
        if task.lower() == 'nt':
            return self._nt_models
        return self._crl_models

model_manager = ModelManager()
