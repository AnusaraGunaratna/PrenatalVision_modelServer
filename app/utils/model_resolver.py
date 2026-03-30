import os
import logging

logger = logging.getLogger(__name__)


def ensure_model_available(model_path, config):
    if os.path.exists(model_path):
        return True

    connection_string = getattr(config, 'AZURE_STORAGE_CONNECTION_STRING', None)
    if not connection_string:
        logger.error(
            f"Model not found locally at {model_path}"
        )
        return False

    weights_dir = os.path.dirname(model_path)
    os.makedirs(weights_dir, exist_ok=True)

    blob_name = os.path.basename(model_path)
    container_name = getattr(config, 'AZURE_MODEL_CONTAINER', 'pvn-models')

    try:
        from azure.storage.blob import BlobServiceClient

        logger.info(f"Downloading {blob_name} from Azure Blob Storage ({container_name})...")
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        with open(model_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())

        logger.info(f"Successfully downloaded {blob_name} to {model_path}.")
        return True
    except ImportError:
        logger.error("azure-storage-blob package not installed")
        return False
    except Exception as e:
        logger.error(f"Failed to download {blob_name} from Azure: {str(e)}")
        return False
