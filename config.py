# config.py

# Path to DTM tile (set manually or loop later)
TILE_PATH = "data/T01_000893_1847_XN_04N191W__N09_065929_1853_XN_05N191W-DEM.tif"

# Output directory
OUTPUT_DIR = "output"

# Segmentation and graph parameters
N_CLUSTERS = 6
MIN_REGION_SIZE = 100
RANDOM_SEED = 42

# Optional A* demo region IDs (set to None to skip)
START_REGION_ID = None
GOAL_REGION_ID = None
