import csv


def build_region_adjacency(region_ids, connectivity=4):
    """
    Build adjacency pairs from a region ID raster.

    Returns:
        set[tuple[int, int]]: undirected edge pairs (u, v) with u < v
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    edges = set()
    rows, cols = region_ids.shape
    neighbors = [(1, 0), (0, 1)]
    if connectivity == 8:
        neighbors.extend([(1, 1), (1, -1)])

    for r in range(rows):
        for c in range(cols):
            u = int(region_ids[r, c])
            if u <= 0:
                continue
            for dr, dc in neighbors:
                rr = r + dr
                cc = c + dc
                if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                    continue
                v = int(region_ids[rr, cc])
                if v <= 0 or v == u:
                    continue
                a, b = (u, v) if u < v else (v, u)
                edges.add((a, b))
    return edges


def build_weighted_edges(edge_pairs, region_attributes, slope_weight=1.0, roughness_weight=0.25):
    """
    Convert edge pairs into dictionaries with traversal weights.
    """
    by_region = {row["region_id"]: row for row in region_attributes}
    weighted = []
    for u, v in sorted(edge_pairs):
        ru = by_region[u]
        rv = by_region[v]
        slope_u = ru.get("mean_slope") or 0.0
        slope_v = rv.get("mean_slope") or 0.0
        rough_u = ru.get("mean_roughness") or 0.0
        rough_v = rv.get("mean_roughness") or 0.0
        cost = (
            slope_weight * 0.5 * (slope_u + slope_v)
            + roughness_weight * 0.5 * (rough_u + rough_v)
            + 1.0
        )
        weighted.append({"source": u, "target": v, "cost": float(cost)})
    return weighted


def export_nodes_csv(region_attributes, out_path):
    if not region_attributes:
        return
    fieldnames = list(region_attributes[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(region_attributes)


def export_edges_csv(weighted_edges, out_path):
    if not weighted_edges:
        return
    fieldnames = ["source", "target", "cost"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(weighted_edges)
