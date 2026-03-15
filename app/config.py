import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret_key")
    DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")
    PORT = int(os.getenv("PORT", 5000))
    HOST = os.getenv("HOST", "0.0.0.0")
    API_KEY = os.getenv("API_KEY", "****")

    MODEL_PATH_CRL_HYBRIDNET = os.getenv("MODEL_PATH_CRL_HYBRIDNET", os.path.join(os.path.dirname(__file__), "weights", "crl_hybrid.pt"))
    MODEL_PATH_CRL_COORDNET = os.getenv("MODEL_PATH_CRL_COORDNET", os.path.join(os.path.dirname(__file__), "weights", "crl_pvnet.pt"))
    MODEL_PATH_CRL_LDBNET = os.getenv("MODEL_PATH_CRL_LDBNET", os.path.join(os.path.dirname(__file__), "weights", "crl_ldb.pt"))
    MODEL_PATH_CRL_YOLO8 = os.getenv("MODEL_PATH_CRL_YOLO8", os.path.join(os.path.dirname(__file__), "weights", "crl_yolo8.pt"))
    MODEL_PATH_CRL_YOLO11 = os.getenv("MODEL_PATH_CRL_YOLO11", os.path.join(os.path.dirname(__file__), "weights", "crl_yolo11.pt"))

    MODEL_PATH_NT_HYBRIDNET = os.getenv("MODEL_PATH_NT_HYBRIDNET", os.path.join(os.path.dirname(__file__), "weights", "nt_hybrid.pt"))
    MODEL_PATH_NT_COORDNET = os.getenv("MODEL_PATH_NT_COORDNET", os.path.join(os.path.dirname(__file__), "weights", "nt_pvnet.pt"))
    MODEL_PATH_NT_LDBNET = os.getenv("MODEL_PATH_NT_LDBNET", os.path.join(os.path.dirname(__file__), "weights", "nt_ldb.pt"))
    MODEL_PATH_NT_YOLO8 = os.getenv("MODEL_PATH_NT_YOLO8", os.path.join(os.path.dirname(__file__), "weights", "nt_yolo8.pt"))
    MODEL_PATH_NT_YOLO11 = os.getenv("MODEL_PATH_NT_YOLO11", os.path.join(os.path.dirname(__file__), "weights", "nt_yolo11.pt"))