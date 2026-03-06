import numpy as np
import uuid
import logging
from datetime import datetime
from typing import Tuple
from app.infrastructure.model_loader import model_manager
from app.utils.constants import CONF_THRESHOLD, NT_BOX_CORRECTION, REFERENCE_SIZES
from app.utils.image_utils import enhance_ultrasound_image, image_to_base64, draw_annotations

logger = logging.getLogger(__name__)

# The primary model used
PRIMARY_MODEL = "Custom YOLO11"


class BiometricCalculator:
    def __init__(self, pixel_to_mm: float = 0.15):
        self.px = pixel_to_mm

    def measure_all(self, detections: list, task: str = 'CRL') -> dict:
        measurements = {}

        for det in detections:
            cls = det['class_name']
            x1, y1, x2, y2 = det['bbox']
            w_mm = (x2 - x1) * self.px
            h_mm = (y2 - y1) * self.px
            diag_mm = np.sqrt(w_mm ** 2 + h_mm ** 2)

            base = {
                'box_px': [x1, y1, x2, y2],
                'width_mm': round(w_mm, 2),
                'height_mm': round(h_mm, 2),
                'confidence': det['confidence'],
            }

            if cls == 'NT':
                measurements['NT'] = {**base, 'thickness_mm': round(h_mm * NT_BOX_CORRECTION, 2)}
            elif cls == 'NB':
                measurements['NB'] = {**base, 'length_mm': round(diag_mm, 2)}
            elif cls == 'H':
                measurements['Head'] = {
                    **base,
                    'BPD_mm': round(max(w_mm, h_mm), 2),
                    'HC_mm': round(np.pi * (w_mm + h_mm) / 2, 2),
                }
            elif cls == 'AB':
                measurements['Abdomen'] = {
                    **base,
                    'circumference_mm': round(np.pi * (w_mm + h_mm) / 2, 2),
                }
            else:
                measurements[cls] = {**base, 'dimension_mm': f'{round(w_mm, 2)} x {round(h_mm, 2)}'}

        # CRL: 
        if task.upper() == 'CRL' and len(detections) >= 2:
            pts = []
            for d in detections:
                bx1, by1, bx2, by2 = d['bbox']
                pts.extend([(bx1, by1), (bx2, by1), (bx1, by2), (bx2, by2)])

            max_dist, p1, p2 = 0, pts[0], pts[1]
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    dist = np.sqrt((pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2)
                    if dist > max_dist:
                        max_dist, p1, p2 = dist, pts[i], pts[j]

            crl_mm = max_dist * self.px
            measurements['CRL'] = {
                'length_cm': round(crl_mm / 10.0, 2),
                'length_mm': round(crl_mm, 2),
                'point1': (int(p1[0]), int(p1[1])),
                'point2': (int(p2[0]), int(p2[1])),
                'distance_px': round(max_dist, 1),
                'confidence': min(d['confidence'] for d in detections),
            }

        return measurements


def auto_calibrate(detections: list, ga_weeks: int = None) -> Tuple[float, list]:
    """Weighted pixel-to-mm ratio estimation from known anatomical references."""
    estimates, details = [], []

    for det in detections:
        cls = det['class_name']
        if cls not in REFERENCE_SIZES:
            continue

        ref = REFERENCE_SIZES[cls]
        x1, y1, x2, y2 = det['bbox']
        w, h = x2 - x1, y2 - y1

        px_size = {'width': w, 'height': h, 'max': max(w, h)}.get(ref['dimension'])
        if not px_size or px_size < 10:
            continue

        expected = ref['expected_mm'].get(ga_weeks, ref['default_mm']) if ga_weeks else ref['default_mm']
        ratio = expected / px_size
        estimates.append((ratio, ref['weight']))
        details.append({'class': cls, 'px_size': round(px_size, 1), 'expected_mm': expected, 'ratio': round(ratio, 5)})

    if not estimates:
        return 0.15, []

    total_w = sum(w for _, w in estimates)
    return round(sum(r * w for r, w in estimates) / total_w, 5), details


class PredictionService:

    @staticmethod
    def run_all_models(img_arr: np.ndarray, scan_type: str, ga_weeks: int = None) -> dict:
        original_b64 = image_to_base64(img_arr)
        enhanced_img = enhance_ultrasound_image(img_arr)
        enhanced_b64 = image_to_base64(enhanced_img)

        models = model_manager.get_models_for_task(scan_type)
        if not models:
            raise RuntimeError(f"No {scan_type.upper()} models found in app/weights/")

        primary = PRIMARY_MODEL if PRIMARY_MODEL in models else list(models.keys())[0]

        comparison = []
        best_measurements = {}
        best_calibration = 0.15

        for name, model in models.items():
            results = model.predict(enhanced_img, conf=CONF_THRESHOLD, verbose=False)[0]

            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                detections.append({
                    "class_name": model.names[cls_id],
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox": list(map(float, box.xyxy[0].cpu().numpy()))
                })

            px_mm, _ = auto_calibrate(detections, ga_weeks)
            measurements = BiometricCalculator(px_mm).measure_all(detections, scan_type)
            annotated_b64 = image_to_base64(draw_annotations(enhanced_img, detections, measurements))

            comparison.append({
                "model_name": name,
                "detections": detections,
                "measurements": measurements,
                "annotated_image_base64": annotated_b64,
            })

            if name == primary:
                best_measurements = measurements
                best_calibration = px_mm

        return {
            "scan_id": str(uuid.uuid4()),
            "scan_type": scan_type,
            "original_image_base64": original_b64,
            "enhanced_image_base64": enhanced_b64,
            "models_comparison": comparison,
            "best_model_name": primary,
            "best_model_measurements": best_measurements,
            "calibration_ratio": best_calibration,
            "processed_at": datetime.utcnow(),
        }

    @staticmethod
    def run_auto_mode(img_arr: np.ndarray, ga_weeks: int = None) -> dict:
        logger.info("AUTO mode: running all 8 models (4 CRL + 4 NT)")

        crl_result = PredictionService.run_all_models(img_arr, 'crl', ga_weeks)
        nt_result = PredictionService.run_all_models(img_arr, 'nt', ga_weeks)

        crl_structures = set()
        crl_total_detections = 0
        for model in crl_result['models_comparison']:
            for det in model['detections']:
                crl_structures.add(det['class_name'])
                crl_total_detections += 1

        nt_structures = set()
        nt_total_detections = 0
        for model in nt_result['models_comparison']:
            for det in model['detections']:
                nt_structures.add(det['class_name'])
                nt_total_detections += 1

        logger.info(
            f"AUTO mode comparison - "
            f"CRL: {len(crl_structures)} unique structures, {crl_total_detections} total detections | "
            f"NT: {len(nt_structures)} unique structures, {nt_total_detections} total detections"
        )

        if len(nt_structures) > len(crl_structures):
            winner = nt_result
            winner_type = 'nt'
        elif len(nt_structures) == len(crl_structures):
            winner = nt_result if nt_total_detections > crl_total_detections else crl_result
            winner_type = 'nt' if nt_total_detections > crl_total_detections else 'crl'
        else:
            winner = crl_result
            winner_type = 'crl'

        logger.info(f"AUTO mode selected: {winner_type.upper()}")
        return winner
