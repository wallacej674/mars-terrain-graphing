import unittest

import numpy as np
from rasterio.transform import from_origin

from modules.region_stats import assign_rule_based_label, compute_region_attributes


class TestRegionStats(unittest.TestCase):
    def test_assign_rule_based_label(self):
        self.assertEqual(assign_rule_based_label(2.0, -0.01), "valley_or_depression")
        self.assertEqual(assign_rule_based_label(12.0, 0.01), "ridge_or_hill")
        self.assertEqual(assign_rule_based_label(2.0, 0.0), "plain")
        self.assertEqual(assign_rule_based_label(8.0, 0.0, mean_roughness=20.0), "rough_terrain")
        self.assertEqual(assign_rule_based_label(7.0, 0.0), "mixed")

    def test_compute_region_attributes(self):
        region_ids = np.array(
            [
                [1, 1, -1],
                [2, 2, 2],
                [-1, 2, 1],
            ],
            dtype=np.int32,
        )
        features = {
            "elevation": np.array(
                [
                    [100, 100, np.nan],
                    [120, 120, 120],
                    [np.nan, 120, 100],
                ],
                dtype=np.float32,
            ),
            "slope": np.array(
                [
                    [2, 2, np.nan],
                    [11, 11, 11],
                    [np.nan, 11, 2],
                ],
                dtype=np.float32,
            ),
            "curvature": np.array(
                [
                    [-0.01, -0.01, np.nan],
                    [0.01, 0.01, 0.01],
                    [np.nan, 0.01, -0.01],
                ],
                dtype=np.float32,
            ),
            "roughness": np.array(
                [
                    [3, 3, np.nan],
                    [12, 12, 12],
                    [np.nan, 12, 3],
                ],
                dtype=np.float32,
            ),
        }
        transform = from_origin(100.0, 50.0, 20.0, 20.0)
        attrs = compute_region_attributes(
            region_ids=region_ids,
            features=features,
            transform=transform,
            crs="EPSG:4326",
            tile_id="tile_001",
        )

        self.assertEqual(len(attrs), 2)
        ids = {row["region_id"] for row in attrs}
        self.assertEqual(ids, {1, 2})
        for row in attrs:
            self.assertIn("class_label", row)
            self.assertEqual(row["tile_id"], "tile_001")
            self.assertEqual(row["crs"], "EPSG:4326")
            self.assertGreater(row["area_m2"], 0.0)
            self.assertIsInstance(row["centroid_lon"], float)
            self.assertIsInstance(row["centroid_lat"], float)


if __name__ == "__main__":
    unittest.main()
