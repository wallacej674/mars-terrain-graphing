import csv
import os
import tempfile
import unittest

import numpy as np

from modules.graph_export import (
    build_region_adjacency,
    build_weighted_edges,
    export_edges_csv,
    export_nodes_csv,
)


class TestGraphExport(unittest.TestCase):
    def test_build_region_adjacency(self):
        region_ids = np.array(
            [
                [1, 1, 2],
                [1, 3, 2],
                [4, 3, 2],
            ],
            dtype=np.int32,
        )
        edges = build_region_adjacency(region_ids, connectivity=4)
        self.assertIn((1, 2), edges)
        self.assertIn((1, 3), edges)
        self.assertIn((2, 3), edges)
        self.assertIn((3, 4), edges)
        self.assertNotIn((2, 4), edges)  # no 4-neighbor contact

    def test_build_region_adjacency_validation(self):
        with self.assertRaises(ValueError):
            build_region_adjacency(np.array([[1]], dtype=np.int32), connectivity=5)

    def test_build_weighted_edges(self):
        edge_pairs = {(1, 2)}
        attrs = [
            {"region_id": 1, "mean_slope": 10.0, "mean_roughness": 4.0},
            {"region_id": 2, "mean_slope": 20.0, "mean_roughness": 8.0},
        ]
        weighted = build_weighted_edges(edge_pairs, attrs, slope_weight=1.0, roughness_weight=0.25)
        self.assertEqual(len(weighted), 1)
        item = weighted[0]
        self.assertEqual((item["source"], item["target"]), (1, 2))
        expected = 1.0 + 0.5 * (10.0 + 20.0) + 0.25 * 0.5 * (4.0 + 8.0)
        self.assertAlmostEqual(item["cost"], expected)

    def test_csv_exports(self):
        nodes = [
            {"region_id": 1, "tile_id": "t", "class_label": "plain"},
            {"region_id": 2, "tile_id": "t", "class_label": "ridge"},
        ]
        edges = [{"source": 1, "target": 2, "cost": 3.5}]

        with tempfile.TemporaryDirectory() as td:
            nodes_path = os.path.join(td, "nodes.csv")
            edges_path = os.path.join(td, "edges.csv")
            export_nodes_csv(nodes, nodes_path)
            export_edges_csv(edges, edges_path)

            with open(nodes_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["region_id"], "1")

            with open(edges_path, newline="", encoding="utf-8") as f:
                edge_rows = list(csv.DictReader(f))
            self.assertEqual(len(edge_rows), 1)
            self.assertEqual(edge_rows[0]["source"], "1")
            self.assertEqual(edge_rows[0]["target"], "2")


if __name__ == "__main__":
    unittest.main()
