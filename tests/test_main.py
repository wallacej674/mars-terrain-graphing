import runpy
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


class _Transform:
    a = 20.0
    e = -20.0


class TestMainScript(unittest.TestCase):
    def test_main_pipeline_orchestration_with_mocks(self):
        elevation = np.ma.array([[1.0, 2.0], [3.0, 4.0]], mask=[[0, 0], [0, 0]])
        profile = {"crs": "EPSG:4326"}
        transform = _Transform()

        cfg = types.ModuleType("config")
        cfg.TILE_PATH = "data/test-DEM.tif"
        cfg.OUTPUT_DIR = "output"
        cfg.N_CLUSTERS = 2
        cfg.MIN_REGION_SIZE = 1
        cfg.RANDOM_SEED = 42
        cfg.START_REGION_ID = None
        cfg.GOAL_REGION_ID = None

        pkg = types.ModuleType("modules")
        pkg.__path__ = []

        mod_load = types.ModuleType("modules.load_tile")
        mod_load.load_dtm_tile = MagicMock(return_value=(elevation, profile, transform))

        mod_compute = types.ModuleType("modules.compute_features")
        mod_compute.compute_slope = MagicMock(return_value=np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32))
        mod_compute.compute_aspect = MagicMock(return_value=np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32))
        mod_compute.compute_curvature = MagicMock(return_value=np.array([[3.0, 3.0], [3.0, 3.0]], dtype=np.float32))
        mod_compute.compute_roughness = MagicMock(return_value=np.array([[4.0, 4.0], [4.0, 4.0]], dtype=np.float32))

        mod_save = types.ModuleType("modules.save_utils")
        mod_save.save_raster = MagicMock()
        mod_save.save_json = MagicMock()

        mod_seg = types.ModuleType("modules.segmentation")
        mod_seg.segment_terrain = MagicMock(
            return_value=(
                np.array([[0, 0], [1, 1]], dtype=np.int32),
                np.array([[1, 1], [2, 2]], dtype=np.int32),
                ["slope", "curvature", "roughness"],
            )
        )

        mod_stats = types.ModuleType("modules.region_stats")
        mod_stats.compute_region_attributes = MagicMock(
            return_value=[
                {"region_id": 1, "centroid_lon": 0.0, "centroid_lat": 0.0},
                {"region_id": 2, "centroid_lon": 1.0, "centroid_lat": 0.0},
            ]
        )

        mod_graph = types.ModuleType("modules.graph_export")
        mod_graph.build_region_adjacency = MagicMock(return_value={(1, 2)})
        mod_graph.build_weighted_edges = MagicMock(return_value=[{"source": 1, "target": 2, "cost": 2.0}])
        mod_graph.export_nodes_csv = MagicMock()
        mod_graph.export_edges_csv = MagicMock()

        mod_path = types.ModuleType("modules.path_planner")
        mod_path.build_region_graph = MagicMock(return_value=object())
        mod_path.astar_path_regions = MagicMock(return_value=[1, 2])
        mod_path.path_cost = MagicMock(return_value=2.0)

        patched = {
            "config": cfg,
            "modules": pkg,
            "modules.load_tile": mod_load,
            "modules.compute_features": mod_compute,
            "modules.save_utils": mod_save,
            "modules.segmentation": mod_seg,
            "modules.region_stats": mod_stats,
            "modules.graph_export": mod_graph,
            "modules.path_planner": mod_path,
        }

        with patch.dict("sys.modules", patched):
            runpy.run_module("main", run_name="__main__")

        mod_load.load_dtm_tile.assert_called_once_with(cfg.TILE_PATH)
        self.assertEqual(mod_save.save_raster.call_count, 6)  # four features + clusters + regions
        mod_seg.segment_terrain.assert_called_once()
        mod_stats.compute_region_attributes.assert_called_once()
        mod_graph.build_region_adjacency.assert_called_once()
        mod_graph.build_weighted_edges.assert_called_once()
        mod_graph.export_nodes_csv.assert_called_once()
        mod_graph.export_edges_csv.assert_called_once()
        mod_save.save_json.assert_called_once()

        # Optional path branch should be skipped with None IDs.
        mod_path.build_region_graph.assert_not_called()
        mod_path.astar_path_regions.assert_not_called()
        mod_path.path_cost.assert_not_called()


if __name__ == "__main__":
    unittest.main()
