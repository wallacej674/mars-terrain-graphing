import json
import os
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

from modules.load_tile import load_dtm_tile
from modules.save_utils import save_json, save_raster


class TestIOModules(unittest.TestCase):
    def test_save_and_load_raster(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        profile = {
            "driver": "GTiff",
            "height": 2,
            "width": 2,
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:4326",
            "transform": from_origin(0.0, 2.0, 1.0, 1.0),
            "nodata": -9999.0,
        }
        with tempfile.TemporaryDirectory() as td:
            tif_path = os.path.join(td, "test.tif")
            save_raster(arr, profile, tif_path)

            elevation, loaded_profile, transform = load_dtm_tile(tif_path)
            np.testing.assert_allclose(np.asarray(elevation), arr)
            self.assertEqual(loaded_profile["width"], 2)
            self.assertEqual(loaded_profile["height"], 2)
            self.assertTrue(isinstance(transform, rasterio.Affine))

    def test_save_json(self):
        payload = {"a": 1, "b": [1, 2, 3]}
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "obj.json")
            save_json(payload, path)
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
        self.assertEqual(loaded, payload)


if __name__ == "__main__":
    unittest.main()
