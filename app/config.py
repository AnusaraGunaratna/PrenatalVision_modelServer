import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret_key")
    DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")
    PORT = int(os.getenv("PORT", 5000))
    HOST = os.getenv("HOST", "0.0.0.0")
    API_KEY = os.getenv("API_KEY", "prenatalvision-internal-service-key")

    # Azure Storage for Models
    AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    AZURE_MODEL_CONTAINER = os.getenv("AZURE_MODEL_CONTAINER", "pvn-models")

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    WEIGHTS_DIR = os.path.join(ROOT_DIR, "weights")

    MODEL_PATH_CRL_HYBRID = os.getenv("MODEL_PATH_CRL_HYBRID", os.path.join(WEIGHTS_DIR, "crl_hybrid.pt"))
    MODEL_PATH_CRL_PVNET = os.getenv("MODEL_PATH_CRL_PVNET", os.path.join(WEIGHTS_DIR, "crl_pvnet.pt"))
    MODEL_PATH_CRL_LDB = os.getenv("MODEL_PATH_CRL_LDB", os.path.join(WEIGHTS_DIR, "crl_ldb.pt"))
    MODEL_PATH_CRL_YOLO8 = os.getenv("MODEL_PATH_CRL_YOLO8", os.path.join(WEIGHTS_DIR, "crl_yolo8.pt"))
    MODEL_PATH_CRL_YOLO11 = os.getenv("MODEL_PATH_CRL_YOLO11", os.path.join(WEIGHTS_DIR, "crl_yolo11.pt"))

    MODEL_PATH_NT_HYBRID = os.getenv("MODEL_PATH_NT_HYBRID", os.path.join(WEIGHTS_DIR, "nt_hybrid.pt"))
    MODEL_PATH_NT_PVNET = os.getenv("MODEL_PATH_NT_PVNET", os.path.join(WEIGHTS_DIR, "nt_pvnet.pt"))
    MODEL_PATH_NT_LDB = os.getenv("MODEL_PATH_NT_LDB", os.path.join(WEIGHTS_DIR, "nt_ldb.pt"))
    MODEL_PATH_NT_YOLO8 = os.getenv("MODEL_PATH_NT_YOLO8", os.path.join(WEIGHTS_DIR, "nt_yolo8.pt"))
    MODEL_PATH_NT_YOLO11 = os.getenv("MODEL_PATH_NT_YOLO11", os.path.join(WEIGHTS_DIR, "nt_yolo11.pt"))