# main.py

import os
from config import TILE_PATH, OUTPUT_DIR
from modules.load_tile import load_dtm_tile
from modules.compute_features import compute_slope, compute_aspect, compute_curvature
from modules.save_utils import save_raster

# Load elevation and metadata
elevation, profile, transform = load_dtm_tile(TILE_PATH)
pixel_size = abs(transform.a)

# Compute topographic features
slope = compute_slope(elevation, pixel_size)
aspect = compute_aspect(elevation, pixel_size)
curvature = compute_curvature(elevation, pixel_size)

# Save each raster
os.makedirs(OUTPUT_DIR, exist_ok=True)
save_raster(slope, profile, os.path.join(OUTPUT_DIR, "slope.tif"))
save_raster(aspect, profile, os.path.join(OUTPUT_DIR, "aspect.tif"))
save_raster(curvature, profile, os.path.join(OUTPUT_DIR, "curvature.tif"))

print("Feature rasters saved.")