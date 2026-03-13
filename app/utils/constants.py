# YOLO Detection Confidence Threshold
CONF_THRESHOLD = 0.25

# Nuchal Translucency Bounding Box Correction Factor
NT_BOX_CORRECTION = 0.40

# Reference sizes for anatomical auto-calibration at 11-14 weeks GA (in mm)
# Clinical References:
# - Head (H), Abdomen (AB): Hadlock FP, et al. Fetal biometry guidelines.
# - Nasal Bone (NB): Cicero S, et al. Ultrasound in Obstetrics & Gynecology (Fetal Medicine Foundation).
# - Cranium (C): Approximated from standard cranial vault dimensions at first trimester.
REFERENCE_SIZES = {
    'H': {
        'dimension': 'max',
        'expected_mm': {11: 17, 12: 21, 13: 25, 14: 28},
        'default_mm': 21.0,
        'weight': 1.0,
    },
    'AB': {
        'dimension': 'max',
        'expected_mm': {11: 15, 12: 19, 13: 23, 14: 27},
        'default_mm': 19.0,
        'weight': 0.7,
    },
    'C': {
        'dimension': 'height',
        'expected_mm': {11: 12, 12: 15, 13: 18, 14: 20},
        'default_mm': 15.0,
        'weight': 0.5,
    },
    'NB': {
        'dimension': 'max',
        'expected_mm': {11: 1.8, 12: 2.4, 13: 3.1, 14: 3.8},
        'default_mm': 2.5,
        'weight': 0.4,
    },
}
CRL_FETAL_LANDMARKS = {'H', 'B', 'MX', 'MDS', 'MLS', 'AB', 'LV', 'RBP', 'DP', 'NB', 'NT'}

# Image Preprocessing Constants
CLAHE_CLIP_LIMIT = 3.0
CLAHE_GRID_SIZE = (8, 8)
BILATERAL_DIAMETER = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75
GAUSSIAN_BLUR_KS = (0, 0)
GAUSSIAN_SIGMA = 2.0
UNSHARP_WEIGHT_1 = 1.5
UNSHARP_WEIGHT_2 = -0.5

# Bounding Box Colors by Class for Visualization
COLORS = {
    'NT': (0, 0, 255),    'NB': (255, 0, 0),    'H': (0, 255, 0),
    'C': (255, 255, 0),   'AB': (255, 0, 255),  'MX': (0, 255, 255),
    'MDS': (128, 255, 0), 'MLS': (255, 128, 0), 'LV': (0, 128, 255),
    'G': (128, 0, 255),   'B': (255, 128, 128), 'RBP': (128, 255, 128),
    'DP': (128, 128, 255), 'NTAPS': (255, 255, 128),
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dicom'}
