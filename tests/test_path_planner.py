import unittest

from modules.path_planner import astar_path_regions, build_region_graph, path_cost


class TestPathPlanner(unittest.TestCase):
    def setUp(self):
        self.region_attributes = [
            {"region_id": 1, "centroid_lon": 0.0, "centroid_lat": 0.0},
            {"region_id": 2, "centroid_lon": 1.0, "centroid_lat": 0.0},
            {"region_id": 3, "centroid_lon": 2.0, "centroid_lat": 0.0},
        ]
        self.weighted_edges = [
            {"source": 1, "target": 2, "cost": 1.0},
            {"source": 2, "target": 3, "cost": 1.0},
            {"source": 1, "target": 3, "cost": 5.0},
        ]

    def test_build_region_graph(self):
        graph = build_region_graph(self.region_attributes, self.weighted_edges)
        self.assertEqual(set(graph.nodes), {1, 2, 3})
        self.assertTrue(graph.has_edge(1, 2))
        self.assertEqual(graph[1][2]["cost"], 1.0)

    def test_astar_path_regions(self):
        graph = build_region_graph(self.region_attributes, self.weighted_edges)
        path = astar_path_regions(graph, 1, 3)
        self.assertEqual(path, [1, 2, 3])

    def test_path_cost(self):
        graph = build_region_graph(self.region_attributes, self.weighted_edges)
        self.assertEqual(path_cost(graph, [1]), 0.0)
        self.assertEqual(path_cost(graph, [1, 2, 3]), 2.0)


if __name__ == "__main__":
    unittest.main()
