from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, List, Dict
from datetime import datetime

class ScanType(str, Enum):
    CRL = "crl"
    NT = "nt"
    AUTO = "auto"

class DetectionResult(BaseModel):
    """Single detection bounding box with confidence and source model."""
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: List[float] = Field(..., min_length=4, max_length=4)
    source_model: Optional[str] = None

class BiometricMeasurement(BaseModel):
    """A generic measurement payload for the structure."""
    dimension_mm: Optional[str] = None
    thickness_mm: Optional[float] = None
    length_cm: Optional[float] = None
    length_mm: Optional[float] = None
    distance_px: Optional[float] = None
    BPD_mm: Optional[float] = None
    HC_mm: Optional[float] = None
    circumference_mm: Optional[float] = None
    height_mm: Optional[float] = None
    width_mm: Optional[float] = None
    confidence: Optional[float] = None
    approximate: Optional[bool] = None

class ModelResult(BaseModel):
    """Result from a specific YOLO model (for per-model comparison)."""
    model_name: str
    detections: List[DetectionResult]
    measurements: Dict[str, BiometricMeasurement]
    annotated_image_base64: str

class ScanResponse(BaseModel):
    """Aggregated response: best-confidence detections across all models."""
    scan_id: str
    scan_type: ScanType
    original_image_base64: str
    enhanced_image_base64: str
    detections: List[DetectionResult]
    measurements: Dict[str, BiometricMeasurement]
    annotated_image_base64: str
    models_comparison: List[ModelResult]
    processed_at: datetime
    calibration_ratio: float = Field(..., description="Pixel to mm ratio")

class BaseResponse(BaseModel):
    """Consistent API response envelope."""
    success: bool
    data: Optional[ScanResponse] = None
    error: Optional[Dict[str, str]] = None
