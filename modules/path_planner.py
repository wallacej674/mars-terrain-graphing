import math

import networkx as nx


def build_region_graph(region_attributes, weighted_edges):
    """
    Build a graph where each node is a region and each edge has traversal cost.
    """
    graph = nx.Graph()

    for row in region_attributes:
        node_id = int(row["region_id"])
        graph.add_node(node_id, **row)

    for edge in weighted_edges:
        graph.add_edge(
            int(edge["source"]),
            int(edge["target"]),
            cost=float(edge["cost"]),
        )

    return graph


def _heuristic(graph, n1, n2):
    """
    Straight-line distance over region centroids as admissible A* heuristic.
    """
    a = graph.nodes[n1]
    b = graph.nodes[n2]
    dx = float(a["centroid_lon"]) - float(b["centroid_lon"])
    dy = float(a["centroid_lat"]) - float(b["centroid_lat"])
    return math.hypot(dx, dy)


def astar_path_regions(graph, start_region_id, goal_region_id):
    """
    Compute least-cost path across region graph using A*.
    """
    return nx.astar_path(
        graph,
        start_region_id,
        goal_region_id,
        heuristic=lambda u, v: _heuristic(graph, u, v),
        weight="cost",
    )


def path_cost(graph, path):
    """
    Sum edge costs along a region path.
    """
    if len(path) < 2:
        return 0.0
    return float(
        sum(
            graph[path[i]][path[i + 1]].get("cost", 1.0)
            for i in range(len(path) - 1)
        )
    )
