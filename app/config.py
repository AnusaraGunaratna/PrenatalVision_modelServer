import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret_key")
    DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")
    PORT = int(os.getenv("PORT", 5000))
    HOST = os.getenv("HOST", "0.0.0.0")
    API_KEY = os.getenv("API_KEY", "****")

    MODEL_PATH_CRL_ORIGINAL_YOLO11 = os.getenv("MODEL_PATH_CRL_ORIGINAL_YOLO11", os.path.join(os.path.dirname(__file__), "weights", "crl_yolo11m_best.pt"))
    MODEL_PATH_CRL_CUSTOM_YOLO11 = os.getenv("MODEL_PATH_CRL_CUSTOM_YOLO11", os.path.join(os.path.dirname(__file__), "weights", "crl_p2_coordattn_best.pt"))
    MODEL_PATH_CRL_ORIGINAL_YOLO8 = os.getenv("MODEL_PATH_CRL_ORIGINAL_YOLO8", os.path.join(os.path.dirname(__file__), "weights", "crl_best_yolo8.pt"))
    MODEL_PATH_CRL_CUSTOM_YOLO8 = os.getenv("MODEL_PATH_CRL_CUSTOM_YOLO8", os.path.join(os.path.dirname(__file__), "weights", "crl_p2_best_yolo8.pt"))

    MODEL_PATH_NT_ORIGINAL_YOLO11 = os.getenv("MODEL_PATH_NT_ORIGINAL_YOLO11", os.path.join(os.path.dirname(__file__), "weights", "nt_yolo11m_best.pt"))
    MODEL_PATH_NT_CUSTOM_YOLO11 = os.getenv("MODEL_PATH_NT_CUSTOM_YOLO11", os.path.join(os.path.dirname(__file__), "weights", "nt_p2_coordattn_best.pt"))
    MODEL_PATH_NT_ORIGINAL_YOLO8 = os.getenv("MODEL_PATH_NT_ORIGINAL_YOLO8", os.path.join(os.path.dirname(__file__), "weights", "nt_best_yolo8.pt"))
    MODEL_PATH_NT_CUSTOM_YOLO8 = os.getenv("MODEL_PATH_NT_CUSTOM_YOLO8", os.path.join(os.path.dirname(__file__), "weights", "nt_p2_best_yolo8.pt"))
