import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def load_graph(nodes_path, edges_path):
    graph = nx.Graph()

    with open(nodes_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            node_id = int(row["region_id"])
            graph.add_node(node_id, **row)

    with open(edges_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            source = int(row["source"])
            target = int(row["target"])
            cost = float(row["cost"])
            graph.add_edge(source, target, cost=cost)

    return graph


def build_positions(graph):
    # Use geographic centroids when available; otherwise use a layout algorithm.
    has_geo = all(
        ("centroid_lon" in data and "centroid_lat" in data)
        for _, data in graph.nodes(data=True)
    )
    if has_geo:
        return {
            node: (float(data["centroid_lon"]), float(data["centroid_lat"]))
            for node, data in graph.nodes(data=True)
        }
    return nx.spring_layout(graph, seed=42)


def load_path(path_json):
    path_file = Path(path_json)
    if not path_file.exists():
        return None
    with open(path_file, encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("path")


def main():
    parser = argparse.ArgumentParser(description="Visualize terrain region graph.")
    parser.add_argument("--nodes", default="output/nodes.csv", help="Path to nodes.csv")
    parser.add_argument("--edges", default="output/edges.csv", help="Path to edges.csv")
    parser.add_argument("--path", default="output/path.json", help="Optional path.json for highlighting A* route")
    parser.add_argument("--save", default=None, help="Optional output image path (e.g., output/graph.png)")
    args = parser.parse_args()

    graph = load_graph(args.nodes, args.edges)
    if graph.number_of_nodes() == 0:
        raise ValueError("Graph is empty. Check nodes/edges files.")

    pos = build_positions(graph)
    plt.figure(figsize=(11, 8))

    nx.draw_networkx_edges(graph, pos, width=0.5, alpha=0.35, edge_color="#666666")
    nx.draw_networkx_nodes(graph, pos, node_size=25, node_color="#1f77b4", alpha=0.9)

    path_nodes = load_path(args.path)
    if path_nodes and len(path_nodes) > 1:
        path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
        nx.draw_networkx_nodes(graph, pos, nodelist=path_nodes, node_size=45, node_color="#d62728")
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, width=2.0, edge_color="#d62728")

    plt.title("Mars Terrain Region Graph")
    plt.axis("off")
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=200)
        print(f"Saved graph image to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
