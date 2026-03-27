import numpy as np
import uuid
import logging
from datetime import datetime
from typing import Tuple
from app.infrastructure.model_loader import model_manager
from app.config import Config
from app.utils.constants import (
    CONF_THRESHOLD, NT_BOX_CORRECTION, REFERENCE_SIZES, CRL_FETAL_LANDMARKS,
    CRL_CALIBRATION_SLOPE, CRL_CALIBRATION_INTERCEPT,
    NT_CALIBRATION_SLOPE, NT_CALIBRATION_INTERCEPT,
)
from app.utils.image_utils import enhance_ultrasound_image, image_to_base64, draw_annotations

logger = logging.getLogger(__name__)


class BiometricCalculator:
    def __init__(self, pixel_to_mm: float = 0.15):
        self.px = pixel_to_mm

    @staticmethod
    def _ramanujan_ellipse_circumference(a: float, b: float) -> float:
        """Ramanujan's approximation for the perimeter of an ellipse.
        a, b are the SEMI-axes (half of width / height in mm)."""
        h = ((a - b) ** 2) / ((a + b) ** 2) if (a + b) > 0 else 0
        return np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))

    @staticmethod
    def _deduplicate(detections: list) -> list:
        best = {}
        for det in detections:
            cls = det['class_name']
            if cls not in best or det['confidence'] > best[cls]['confidence']:
                best[cls] = det
        return list(best.values())

    def measure_all(self, detections: list, task: str = 'CRL') -> dict:
        measurements = {}
        unique_dets = self._deduplicate(detections)

        for det in unique_dets:
            cls = det['class_name']
            x1, y1, x2, y2 = det['bbox']
            w_mm = (x2 - x1) * self.px
            h_mm = (y2 - y1) * self.px

            base = {
                'box_px': [x1, y1, x2, y2],
                'width_mm': round(w_mm, 2),
                'height_mm': round(h_mm, 2),
                'confidence': det['confidence'],
            }

            if cls == 'NT':
                raw_thickness = h_mm * NT_BOX_CORRECTION
                # Post-prediction linear calibration for NT
                calibrated_thickness = NT_CALIBRATION_SLOPE * raw_thickness + NT_CALIBRATION_INTERCEPT
                measurements['NT'] = {
                    **base,
                    'thickness_mm': round(calibrated_thickness, 2),
                    'approximate': True,
                }
            elif cls == 'NB':
                measurements['NB'] = {**base, 'length_mm': round(max(w_mm, h_mm), 2)}
            elif cls == 'H':
                a, b = w_mm / 2.0, h_mm / 2.0
                measurements['Head'] = {
                    **base,
                    'BPD_mm': round(min(w_mm, h_mm), 2),
                    'HC_mm': round(self._ramanujan_ellipse_circumference(a, b), 2),
                }
            elif cls == 'AB':
                a, b = w_mm / 2.0, h_mm / 2.0
                measurements['Abdomen'] = {
                    **base,
                    'circumference_mm': round(self._ramanujan_ellipse_circumference(a, b), 2),
                }

        if task.upper() == 'CRL':
            fetal_dets = [d for d in unique_dets if d['class_name'] in CRL_FETAL_LANDMARKS]
            if len(fetal_dets) >= 2:
                pts = []
                for d in fetal_dets:
                    bx1, by1, bx2, by2 = d['bbox']
                    pts.extend([(bx1, by1), (bx2, by1), (bx1, by2), (bx2, by2)])

                max_dist, p1, p2 = 0, pts[0], pts[1]
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        dist = np.sqrt((pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2)
                        if dist > max_dist:
                            max_dist, p1, p2 = dist, pts[i], pts[j]

                raw_crl_mm = max_dist * self.px
                # Post-prediction linear calibration for CRL
                calibrated_crl = CRL_CALIBRATION_SLOPE * raw_crl_mm + CRL_CALIBRATION_INTERCEPT
                measurements['CRL'] = {
                    'length_mm': round(calibrated_crl, 2),
                    'confidence': min(d['confidence'] for d in fetal_dets),
                }

        return measurements


def auto_calibrate(detections: list, ga_weeks: int = None) -> Tuple[float, list]:
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


def _select_best_detections(all_model_detections: dict) -> list:
    best_per_class: dict = {}

    for model_name, detections in all_model_detections.items():
        for det in detections:
            cls = det['class_name']
            if cls not in best_per_class or det['confidence'] > best_per_class[cls]['confidence']:
                best_per_class[cls] = {
                    "class_name": cls,
                    "confidence": det['confidence'],
                    "bbox": det['bbox'],
                    "source_model": model_name,
                }

    return list(best_per_class.values())


class PredictionService:

    @staticmethod
    def run_all_models(img_arr: np.ndarray, scan_type: str, ga_weeks: int = None) -> dict:
        original_b64 = image_to_base64(img_arr)
        enhanced_img = enhance_ultrasound_image(img_arr)
        enhanced_b64 = image_to_base64(enhanced_img)

        model_names = ["PV-Hybrid", "PV-Coord", "PV-LDB", "YOLO8", "YOLO11"] 
        all_model_detections: dict = {}
        
        for name in model_names:
            model = model_manager.get_model(scan_type, name, Config)
            if model is None:
                logger.warning(f"Skipping model {name} as it could not be loaded")
                continue
                
            results = model.predict(enhanced_img, conf=CONF_THRESHOLD, verbose=False)[0]
            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                detections.append({
                    "class_name": model.names[cls_id],
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox": list(map(float, box.xyxy[0].cpu().numpy())),
                    "source_model": name
                })
            all_model_detections[name] = detections

        # Using Ensemble Aggregation
        if all_model_detections:
            best_detections = _select_best_detections(all_model_detections)
        else:
            best_detections = []

        logger.info(
            f"Ensemble detections ({len(best_detections)}): "
            f"{[d['class_name'] + ' (' + d['source_model'] + ', ' + str(round(d['confidence'] * 100, 1)) + '%)' for d in best_detections]}"
        )

        shared_px_mm, _ = auto_calibrate(best_detections, ga_weeks)
        logger.info(f"Calibration ratio: {shared_px_mm} mm/px")

        calculator = BiometricCalculator(shared_px_mm)
        best_measurements = calculator.measure_all(best_detections, scan_type)

        annotated_b64 = image_to_base64(
            draw_annotations(enhanced_img, best_detections, best_measurements)
        )

        comparison = []
        for name, detections in all_model_detections.items():
            model_measurements = calculator.measure_all(detections, scan_type)
            model_annotated_b64 = image_to_base64(
                draw_annotations(enhanced_img, detections, model_measurements)
            )
            comparison.append({
                "model_name": name,
                "detections": detections,
                "measurements": model_measurements,
                "annotated_image_base64": model_annotated_b64,
            })

        return {
            "scan_id": str(uuid.uuid4()),
            "scan_type": scan_type,
            "original_image_base64": original_b64,
            "enhanced_image_base64": enhanced_b64,
            "detections": best_detections,
            "measurements": best_measurements,
            "annotated_image_base64": annotated_b64,
            "models_comparison": comparison,
            "calibration_ratio": shared_px_mm,
            "processed_at": datetime.utcnow(),
        }

    @staticmethod
    def run_auto_mode(img_arr: np.ndarray, ga_weeks: int = None) -> dict:
        logger.info("AUTO mode: running all 10 models (5 CRL + 5 NT)")

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
