import os
import logging
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger(__name__)

def download_model_if_missing(model_path, config):
    """
    Downloads the model from Azure Blob Storage if it doesn't exist locally.
    """
    if os.path.exists(model_path):
        return True

    weights_dir = os.path.dirname(model_path)
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir, exist_ok=True)

    blob_name = os.path.basename(model_path)
    connection_string = getattr(config, 'AZURE_STORAGE_CONNECTION_STRING', None)
    container_name = getattr(config, 'AZURE_MODEL_CONTAINER', 'pvn-models')

    if not connection_string:
        logger.error(f"Missing AZURE_STORAGE_CONNECTION_STRING. Cannot download {blob_name}.")
        return False

    try:
        logger.info(f"Downloading {blob_name} from Azure Blob Storage ({container_name})...")
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        with open(model_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        
        logger.info(f"Successfully downloaded {blob_name} to {model_path}.")
        return True
    except Exception as e:
        logger.error(f"Failed to download {blob_name} from Azure: {str(e)}")
        return False
