# modules/save_utils.py

import numpy as np
import rasterio
import os
import json

def save_raster(array, profile, out_path):
    """
    Saves a 2D array as a GeoTIFF using the given profile.
    """
    profile = profile.copy()
    profile.update(dtype=rasterio.float32, count=1)
    
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(array.astype(rasterio.float32), 1)

def save_json(obj, out_path):
    """
    Saves a dictionary as a JSON file.
    """
    with open(out_path, 'w') as f:
        json.dump(obj, f, indent=4)