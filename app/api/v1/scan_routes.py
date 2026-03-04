import numpy as np
import cv2
import logging
from flask import request, jsonify
from pydantic import ValidationError
from app.api.v1 import scan_bp
from app.models.scan import ScanType, ScanResponse, BaseResponse
from app.services.prediction_service import PredictionService
from app.utils.constants import ALLOWED_EXTENSIONS

logger = logging.getLogger(__name__)


def _error(code: str, message: str, status: int, details=None):
    payload = {"code": code, "message": message}
    if details:
        payload["details"] = details
    return jsonify(BaseResponse(success=False, error=payload).model_dump()), status


def _allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@scan_bp.route('/analyze', methods=['POST'])
def analyze_scan():
    # Input validation
    if 'image' not in request.files or request.files['image'].filename == '':
        return _error("BAD_REQUEST", "Missing 'image' in multipart form data", 400)

    file = request.files['image']
    if not _allowed_file(file.filename):
        return _error("BAD_REQUEST", f"Unsupported file extension: {file.filename}", 400)

    scan_type_str = request.form.get('scan_type', 'crl').lower()
    try:
        scan_type = ScanType(scan_type_str)
    except ValueError:
        return _error("BAD_REQUEST", f"Invalid scan_type '{scan_type_str}'. Expected 'crl' or 'nt'", 400)

    ga_weeks_str = request.form.get('ga_weeks')
    ga_weeks = int(ga_weeks_str) if ga_weeks_str and ga_weeks_str.isdigit() else None

    # Decode image 
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img_arr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_arr is None:
        return _error("BAD_REQUEST", "cv2.imdecode returned None — file is not a valid image", 400)

    # Inference 
    try:
        result = PredictionService.run_all_models(img_arr, scan_type.value, ga_weeks)
        validated = ScanResponse(**result)
        return jsonify(BaseResponse(success=True, data=validated).model_dump()), 200
    except ValidationError as e:
        logger.error(f"Pydantic validation failed on response: {e.errors()}")
        return _error("VALIDATION_ERROR", str(e), 500, e.errors())
    except Exception as e:
        logger.error(f"Inference pipeline failed: {str(e)}", exc_info=True)
        return _error("INTERNAL_ERROR", str(e), 500)
