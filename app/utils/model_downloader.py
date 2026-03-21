import os
import logging
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger(__name__)

class ModelDownloader:
    def __init__(self, connection_string, container_name="pvn-models"):
        self.connection_string = connection_string
        self.container_name = container_name
        self.weights_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
        
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)

    def download_models(self):
        if not self.connection_string:
            logger.warning("No Azure Storage connection string found. Skipping model download.")
            return

        try:
            blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
            container_client = blob_service_client.get_container_client(self.container_name)

            logger.info(f"Checking for models in container: {self.container_name}")
            blobs = container_client.list_blobs()
            
            for blob in blobs:
                if blob.name.endswith(".pt"):
                    target_path = os.path.join(self.weights_dir, blob.name)
                    
                    if not os.path.exists(target_path):
                        logger.info(f"Downloading model: {blob.name}...")
                        with open(target_path, "wb") as f:
                            data = container_client.get_blob_client(blob).download_blob()
                            f.write(data.readall())
                        logger.info(f"Successfully downloaded {blob.name}")
                    else:
                        logger.debug(f"Model {blob.name} already exists. Skipping.")
                        
        except Exception as e:
            logger.error(f"Error downloading models from Azure: {str(e)}")
            # We don't raise the error here to allow the server to start even if download fails
            # (it will fail later when trying to load models if they are missing)
