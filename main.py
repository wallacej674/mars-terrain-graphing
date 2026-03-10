# main.py

import os
import json
import numpy as np

from config import (
    TILE_PATH,
    OUTPUT_DIR,
    N_CLUSTERS,
    MIN_REGION_SIZE,
    RANDOM_SEED,
    START_REGION_ID,
    GOAL_REGION_ID,
)
from modules.load_tile import load_dtm_tile
from modules.compute_features import (
    compute_slope,
    compute_aspect,
    compute_curvature,
    compute_roughness,
)
from modules.save_utils import save_raster, save_json
from modules.segmentation import segment_terrain
from modules.region_stats import compute_region_attributes
from modules.graph_export import (
    build_region_adjacency,
    build_weighted_edges,
    export_nodes_csv,
    export_edges_csv,
)
from modules.path_planner import build_region_graph, astar_path_regions, path_cost

# Load elevation and metadata
elevation, profile, transform = load_dtm_tile(TILE_PATH)
pixel_size = abs(transform.a)
tile_id = os.path.splitext(os.path.basename(TILE_PATH))[0]

# Compute topographic features
slope = compute_slope(elevation, pixel_size)
aspect = compute_aspect(elevation, pixel_size)
curvature = compute_curvature(elevation, pixel_size)
roughness = compute_roughness(elevation, window_size=5)

# Save feature rasters
os.makedirs(OUTPUT_DIR, exist_ok=True)
save_raster(slope, profile, os.path.join(OUTPUT_DIR, "slope.tif"))
save_raster(aspect, profile, os.path.join(OUTPUT_DIR, "aspect.tif"))
save_raster(curvature, profile, os.path.join(OUTPUT_DIR, "curvature.tif"))
save_raster(roughness, profile, os.path.join(OUTPUT_DIR, "roughness.tif"))

features = {
    "elevation": elevation,
    "slope": slope,
    "curvature": curvature,
    "roughness": roughness,
}
valid_mask = (~np.ma.getmaskarray(elevation)) & np.isfinite(np.asarray(elevation))

# Segment terrain into contiguous region IDs
cluster_map, region_ids, used_features = segment_terrain(
    features={"slope": slope, "curvature": curvature, "roughness": roughness},
    valid_mask=valid_mask,
    n_clusters=N_CLUSTERS,
    min_region_size=MIN_REGION_SIZE,
    seed=RANDOM_SEED,
)

save_raster(cluster_map, profile, os.path.join(OUTPUT_DIR, "clusters.tif"))
save_raster(region_ids, profile, os.path.join(OUTPUT_DIR, "regions.tif"))

# Compute region attributes and export tabular graph data
region_attributes = compute_region_attributes(
    region_ids=region_ids,
    features=features,
    transform=transform,
    crs=profile.get("crs"),
    tile_id=tile_id,
)

edge_pairs = build_region_adjacency(region_ids, connectivity=4)
weighted_edges = build_weighted_edges(edge_pairs, region_attributes)

save_json(region_attributes, os.path.join(OUTPUT_DIR, "regions.json"))
export_nodes_csv(region_attributes, os.path.join(OUTPUT_DIR, "nodes.csv"))
export_edges_csv(weighted_edges, os.path.join(OUTPUT_DIR, "edges.csv"))

# Optional A* demonstration if IDs are configured
if START_REGION_ID is not None and GOAL_REGION_ID is not None:
    graph = build_region_graph(region_attributes, weighted_edges)
    path = astar_path_regions(graph, START_REGION_ID, GOAL_REGION_ID)
    path_total_cost = path_cost(graph, path)
    path_payload = {"path": path, "total_cost": path_total_cost}
    with open(os.path.join(OUTPUT_DIR, "path.json"), "w", encoding="utf-8") as f:
        json.dump(path_payload, f, indent=2)
    print(f"A* path saved with {len(path)} regions and cost {path_total_cost:.3f}")

print(f"Pipeline complete. Features used for clustering: {used_features}")
