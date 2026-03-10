import unittest

import numpy as np

from modules.compute_features import (
    compute_aspect,
    compute_curvature,
    compute_roughness,
    compute_slope,
)


class TestComputeFeatures(unittest.TestCase):
    def test_slope_on_flat_surface_is_zero(self):
        elevation = np.zeros((5, 5), dtype=np.float32)
        slope = compute_slope(elevation, pixel_size=1.0)
        self.assertEqual(slope.shape, elevation.shape)
        self.assertTrue(np.allclose(slope, 0.0))

    def test_aspect_range_is_0_to_360(self):
        elevation = np.array(
            [
                [0, 1, 2],
                [0, 1, 2],
                [0, 1, 2],
            ],
            dtype=np.float32,
        )
        aspect = compute_aspect(elevation, pixel_size=1.0)
        self.assertEqual(aspect.shape, elevation.shape)
        self.assertTrue(np.all(aspect >= 0.0))
        self.assertTrue(np.all(aspect < 360.0))

    def test_curvature_on_linear_plane_near_zero(self):
        x = np.tile(np.arange(7, dtype=np.float32), (7, 1))
        curvature = compute_curvature(x, pixel_size=1.0)
        self.assertEqual(curvature.shape, x.shape)
        # Sobel-based second derivative introduces edge artifacts;
        # center band should remain near zero for a linear plane.
        self.assertTrue(np.allclose(curvature[:, 2:-2], 0.0, atol=1e-5))

    def test_roughness_on_flat_surface_is_zero(self):
        elevation = np.ones((6, 6), dtype=np.float32) * 5.0
        roughness = compute_roughness(elevation, window_size=3)
        self.assertEqual(roughness.shape, elevation.shape)
        self.assertTrue(np.allclose(roughness, 0.0))

    def test_roughness_window_validation(self):
        elevation = np.ones((5, 5), dtype=np.float32)
        with self.assertRaises(ValueError):
            compute_roughness(elevation, window_size=2)
        with self.assertRaises(ValueError):
            compute_roughness(elevation, window_size=1)


if __name__ == "__main__":
    unittest.main()
