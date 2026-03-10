import os
import logging
import boto3
from botocore import UNSIGNED
from botocore.config import Config

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# S3 bucket and prefix for Mars CTX DTMs (public AWS Open Data bucket)
BUCKET_NAME = "astrogeo-ard"
PREFIX = "mars/mro/ctx/controlled/usgs/"

def download_ctx_dtms():
    """
    Connect to the S3 bucket anonymously, list CTX DTM GeoTIFF keys, 
    and download each to the local data/ directory.
    """
    # Initialize S3 client with unsigned requests (no AWS credentials needed):contentReference[oaicite:5]{index=5}
    s3_client = boto3.client('s3', region_name='us-west-2', config=Config(signature_version=UNSIGNED))
    
    # Use S3 resource to iterate over objects with the given prefix (handles pagination):contentReference[oaicite:6]{index=6}
    s3_resource = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
    bucket = s3_resource.Bucket(BUCKET_NAME)
    
    # Ensure local data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Iterate through all objects under the prefix
    for obj in bucket.objects.filter(Prefix=PREFIX):
        key = obj.key
        # Filter for GeoTIFF DTMs: ends with .tif and skip orthos or masks (e.g., "-DRG" or "GoodPixelMap"):contentReference[oaicite:7]{index=7}
        key_lower = key.lower()
        if not key_lower.endswith(".tif"):
            continue  # skip non-tif files (e.g., XML metadata or other formats)
        if "-drg" in key_lower or "goodpixelmap" in key_lower:
            continue  # skip orthoimage tiles and quality mask files
        # (Optionally ensure it's a DTM by checking '-dem' in filename)
        if "-dem" not in key_lower:
            continue  # skip any .tif that is not a DEM (for safety, e.g., error images if present)
        
        # Determine local file path (preserve sub-folder structure after the prefix)
        relative_path = key[len(PREFIX):]  # path under the usgs/ prefix
        local_path = os.path.join("data", relative_path)
        # Create subdirectories if any
        local_dir = os.path.dirname(local_path)
        if local_dir: 
            os.makedirs(local_dir, exist_ok=True)
        
        try:
            # Download the S3 object to the local file
            s3_client.download_file(BUCKET_NAME, key, local_path)
            logger.info(f"Downloaded {key} to {local_path}")
        except Exception as e:
            # Log any download errors
            logger.error(f"Failed to download {key}: {e}")

if __name__ == "__main__":
    download_ctx_dtms()