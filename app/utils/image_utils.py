import cv2
import numpy as np
import base64
from app.utils.constants import (
    CLAHE_CLIP_LIMIT, CLAHE_GRID_SIZE, BILATERAL_DIAMETER,
    BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE, GAUSSIAN_BLUR_KS,
    GAUSSIAN_SIGMA, UNSHARP_WEIGHT_1, UNSHARP_WEIGHT_2, COLORS
)

def enhance_ultrasound_image(img_arr: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE, Bilateral Filter, and Unsharp Masking
    """
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    result = cv2.bilateralFilter(
        img_arr, BILATERAL_DIAMETER, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE
    )
    
    if len(result.shape) == 3:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    else:
        result = clahe.apply(result)
        
    gaussian = cv2.GaussianBlur(result, GAUSSIAN_BLUR_KS, GAUSSIAN_SIGMA)
    result = cv2.addWeighted(result, UNSHARP_WEIGHT_1, gaussian, UNSHARP_WEIGHT_2, 0)
    return result

def image_to_base64(img_arr: np.ndarray) -> str:
    """Encodes a numpy image to base64 string."""
    _, buffer = cv2.imencode('.jpg', img_arr)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_str}"

def draw_annotations(img: np.ndarray, detections: list, measurements: dict) -> np.ndarray:
    """Draws bounding boxes and measurements on the image."""
    annotated = img.copy()
    
    for det in detections:
        cls = det['class_name']
        conf_val = det['confidence']
        x1, y1, x2, y2 = map(int, det['bbox'])
        color = COLORS.get(cls, (0, 255, 0))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = cls
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


    return annotated
