import unittest

import numpy as np

from modules.segmentation import (
    connected_regions,
    kmeans_segment,
    segment_terrain,
    stack_features,
)


class TestSegmentation(unittest.TestCase):
    def test_stack_features_requires_non_empty_dict(self):
        with self.assertRaises(ValueError):
            stack_features({})

    def test_stack_features_shape_validation(self):
        features = {
            "a": np.zeros((2, 2), dtype=np.float32),
            "b": np.zeros((3, 3), dtype=np.float32),
        }
        with self.assertRaises(ValueError):
            stack_features(features)

    def test_stack_features_applies_mask_and_finite_filter(self):
        a = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        valid_mask = np.array([[True, True], [False, True]])
        X, out_mask, names = stack_features({"a": a, "b": b}, valid_mask=valid_mask)
        self.assertEqual(names, ["a", "b"])
        self.assertEqual(X.shape, (2, 2))
        self.assertTrue(out_mask[0, 0])
        self.assertFalse(out_mask[0, 1])  # removed due to NaN in a
        self.assertFalse(out_mask[1, 0])  # removed due to explicit valid_mask=False

    def test_stack_features_normalizes_by_default(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
        X, _, _ = stack_features({"a": a, "b": b})
        self.assertTrue(np.allclose(np.mean(X, axis=0), 0.0, atol=1e-6))
        self.assertTrue(np.allclose(np.std(X, axis=0), 1.0, atol=1e-6))

    def test_kmeans_segment_validates_inputs(self):
        with self.assertRaises(ValueError):
            kmeans_segment(np.array([]), np.ones((1, 1), dtype=bool), (1, 1), n_clusters=2)

        with self.assertRaises(ValueError):
            kmeans_segment(np.array([[1.0], [2.0]]), np.ones((1, 2), dtype=bool), (1, 2), n_clusters=1)

    def test_connected_regions_respects_min_size(self):
        cluster_map = np.array(
            [
                [0, 0, -1],
                [0, 1, 1],
                [-1, 1, 2],
            ],
            dtype=np.int32,
        )
        regions = connected_regions(cluster_map, min_region_size=3, connectivity=1)
        unique = set(np.unique(regions))
        self.assertIn(1, unique)       # cluster 0 component kept (size 3)
        self.assertIn(2, unique)       # cluster 1 component kept (size 3)
        self.assertIn(-1, unique)      # nodata + filtered small components
        self.assertNotIn(3, unique)    # cluster 2 single pixel filtered

    def test_segment_terrain_end_to_end(self):
        slope = np.array(
            [
                [1, 1, 10, 10],
                [1, 1, 10, 10],
                [1, 1, 10, 10],
                [1, 1, 10, 10],
            ],
            dtype=np.float32,
        )
        curvature = np.zeros_like(slope)
        roughness = np.zeros_like(slope)
        cluster_map, region_ids, names = segment_terrain(
            {"slope": slope, "curvature": curvature, "roughness": roughness},
            n_clusters=2,
            min_region_size=1,
            seed=123,
        )
        self.assertEqual(cluster_map.shape, slope.shape)
        self.assertEqual(region_ids.shape, slope.shape)
        self.assertEqual(names, ["slope", "curvature", "roughness"])
        self.assertTrue(np.all(region_ids > 0))


if __name__ == "__main__":
    unittest.main()
