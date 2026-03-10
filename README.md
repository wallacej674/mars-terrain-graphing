# Mars Terrain Graphing

Python pipeline for analyzing Martian CTX digital terrain models (DTMs), segmenting terrain into contiguous regions, and exporting a region graph for downstream reasoning and rover-style path planning.

## Project Goals

This project builds an analysis-ready terrain workflow for Mars that:

1. Extracts topographic features from DTM rasters.
2. Segments terrain into coherent contiguous regions.
3. Computes semantic/geometric attributes per region.
4. Builds adjacency relationships as a graph representation.
5. Supports cost-based path planning (A* demo) over region nodes.

The long-term target is a knowledge-graph-ready terrain representation that can support scientific queries and mission planning.

## Current Pipeline

`main.py` runs a full single-tile pipeline:

1. Load one GeoTIFF DTM tile.
2. Compute feature rasters:
   - Slope
   - Aspect
   - Curvature
   - Roughness
3. Segment terrain:
   - K-means clustering in feature space (`slope`, `curvature`, `roughness`)
   - Connected-component split into contiguous `region_id`s
4. Compute per-region attributes:
   - Area
   - Centroid
   - Mean elevation/slope/curvature/roughness
   - Rule-based label (starter taxonomy)
5. Build region adjacency edges and traversal costs.
6. Export graph-friendly files (`nodes.csv`, `edges.csv`) and region metadata.
7. Optionally run A* between configured start/goal region IDs.

## Data Source

DTMs are downloaded from the public USGS Astrogeology ARD S3 bucket via `download_tiles.py`:

- Bucket: `astrogeo-ard`
- Prefix: `mars/mro/ctx/controlled/usgs/`
- Filtered to DEM GeoTIFF products (`-DEM`), excluding DRG/mask products.

## Repository Structure

```text
mars-terrain-graphing/
  main.py
  config.py
  download_tiles.py
  requirements.txt
  modules/
    load_tile.py
    compute_features.py
    segmentation.py
    region_stats.py
    graph_export.py
    path_planner.py
    save_utils.py
  data/
  output/
```

## Architecture (Mermaid)

```mermaid
flowchart TD
    A[USGS Astrogeo ARD S3] --> B[download_tiles.py]
    B --> C[data/*.tif DEM tiles]
    C --> D[main.py]

    D --> E[load_tile.py]
    E --> F[compute_features.py<br/>slope/aspect/curvature/roughness]
    F --> G[Feature Rasters<br/>output/*.tif]

    F --> H[segmentation.py<br/>k-means + connected components]
    H --> I[regions.tif + clusters.tif]

    I --> J[region_stats.py<br/>region attributes + labels]
    J --> K[regions.json]

    I --> L[graph_export.py<br/>adjacency + edge costs]
    J --> L
    L --> M[nodes.csv + edges.csv]

    M --> N[path_planner.py<br/>networkx graph + A*]
    J --> N
    N --> O[path.json (optional)]
```

## Module Responsibilities

- `modules/load_tile.py`
  - Loads DTM raster, profile, and affine transform.
- `modules/compute_features.py`
  - Computes slope, aspect, curvature, roughness.
- `modules/segmentation.py`
  - Converts feature rasters to a clustering matrix.
  - Runs k-means.
  - Produces contiguous region IDs with connected components.
- `modules/region_stats.py`
  - Aggregates region-level metrics and assigns starter class labels.
- `modules/graph_export.py`
  - Derives adjacency edges from region raster.
  - Computes edge traversal cost from region attributes.
  - Exports node/edge CSVs for Neo4j or other graph systems.
- `modules/path_planner.py`
  - Builds a `networkx` region graph.
  - Runs A* and computes total path cost.
- `modules/save_utils.py`
  - Writes rasters and JSON artifacts.

## Configuration

Set runtime controls in `config.py`:

- `TILE_PATH`: input DTM file.
- `OUTPUT_DIR`: output folder.
- `N_CLUSTERS`: k-means cluster count.
- `MIN_REGION_SIZE`: minimum connected-component size to keep.
- `RANDOM_SEED`: reproducible clustering.
- `START_REGION_ID`, `GOAL_REGION_ID`: optional A* demo.

## Installation

```bash
pip install -r requirements.txt
```

Required packages:

- `rasterio`
- `scipy`
- `numpy`
- `matplotlib`
- `networkx`

## Usage

### 1. Download DEM tiles (optional, if not already present)

```bash
python download_tiles.py
```

### 2. Configure input tile and parameters

Edit `config.py`:

- point `TILE_PATH` to a DEM `.tif`
- tune segmentation knobs
- optionally set start/goal region IDs for A*

### 3. Run pipeline

```bash
python main.py
```

## Outputs

Artifacts written to `output/`:

- `slope.tif`
- `aspect.tif`
- `curvature.tif`
- `roughness.tif`
- `clusters.tif`
- `regions.tif`
- `regions.json` (region attributes + class labels)
- `nodes.csv` (graph nodes)
- `edges.csv` (graph edges with cost)
- `path.json` (only if A* start/goal are configured)

## Graph Semantics

- Node: one contiguous terrain region.
- Edge: adjacency between two regions.
- Edge cost (current heuristic):
  - average slope term
  - average roughness term
  - constant base cost

This produces a navigability graph suitable for shortest-path methods and later Neo4j ingestion.

## Known Limitations (Current Stage)

1. Pipeline currently runs one tile per execution.
2. Roughness uses a simple moving-window implementation and can be slow on large rasters.
3. Rule-based class labels are starter heuristics, not validated geomorphological taxonomy.
4. No cross-tile stitching/adjacency resolution yet.
5. No direct Neo4j write path yet (CSV export currently).

## Next Steps

1. Add batch processing over all DEM tiles.
2. Improve nodata/mask propagation across all derivatives.
3. Vectorize roughness for speed.
4. Refine segmentation with feature scaling and post-processing.
5. Add crater-aware labeling and better landform rules.
6. Add Neo4j ingestion script (nodes/edges bulk load).
7. Demonstrate start-goal routing over multiple connected tiles.

## Why This Design

- Raster-based feature extraction preserves geospatial fidelity.
- Region graph abstraction reduces path-planning complexity relative to full pixel grids.
- Exporting both raster and graph artifacts keeps the pipeline explainable, testable, and extensible.
