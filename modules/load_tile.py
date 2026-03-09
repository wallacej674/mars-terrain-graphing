# modules/load_tile.py

import rasterio
import numpy as np

def load_dtm_tile(tile_path):
    """
    Loads a Mars DEM tile and returns the elevation array and raster metadata.
    
    Args:
        tile_path (str): Path to the GeoTIFF file.
        
    Returns:
        elevation (np.ndarray): 2D elevation values.
        profile (dict): Raster metadata.
        transform (Affine): Raster affine transformation (pixel <-> geo).
    """
    with rasterio.open(tile_path) as src:
        elevation = src.read(1, masked=True)
        profile = src.profile
        transform = src.transform
    
    return elevation, profile, transform