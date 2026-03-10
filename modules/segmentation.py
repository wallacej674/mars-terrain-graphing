import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.ndimage import label


def stack_features(features, valid_mask=None, normalize=True):
    """
    Convert feature rasters into a 2D matrix for clustering.

    Args:
        features (dict[str, np.ndarray]): Feature name -> 2D raster.
        valid_mask (np.ndarray | None): Boolean mask where True means valid pixel.

    Returns:
        tuple[np.ndarray, np.ndarray, list[str]]:
            - X: (n_valid_pixels, n_features)
            - mask: (rows, cols) boolean mask of valid pixels
            - names: ordered feature names used in X
    """
    if not features:
        raise ValueError("features cannot be empty")

    names = list(features.keys())
    first = features[names[0]]
    shape = first.shape
    for name in names:
        if features[name].shape != shape:
            raise ValueError(f"feature '{name}' shape does not match {shape}")

    if valid_mask is None:
        valid_mask = np.ones(shape, dtype=bool)

    # Require finite values in every feature at valid pixels.
    for name in names:
        valid_mask = valid_mask & np.isfinite(features[name])

    X = np.column_stack([features[name][valid_mask] for name in names]).astype(np.float32)

    if normalize:
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma[sigma == 0.0] = 1.0
        X = (X - mu) / sigma

    return X, valid_mask, names


def kmeans_segment(X, valid_mask, shape, n_clusters=6, seed=42, n_iter=25):
    """
    Run k-means in feature space and map labels back to raster space.

    Args:
        X (np.ndarray): (n_valid_pixels, n_features)
        valid_mask (np.ndarray): (rows, cols) mask of valid pixels
        shape (tuple[int, int]): Raster shape
        n_clusters (int): Number of clusters for segmentation
        seed (int): Random seed
        n_iter (int): Number of k-means iterations

    Returns:
        np.ndarray: cluster map with -1 for invalid pixels
    """
    if X.size == 0:
        raise ValueError("no valid pixels available for clustering")
    if n_clusters < 2:
        raise ValueError("n_clusters must be >= 2")

    rng = np.random.default_rng(seed)
    centroids, labels = kmeans2(X, k=n_clusters, iter=n_iter, minit="points", seed=rng)
    _ = centroids

    cluster_map = np.full(shape, -1, dtype=np.int32)
    cluster_map[valid_mask] = labels.astype(np.int32)
    return cluster_map


def connected_regions(cluster_map, min_region_size=100, connectivity=1):
    """
    Split cluster map into contiguous region IDs.

    Args:
        cluster_map (np.ndarray): 2D cluster labels with -1 as nodata
        min_region_size (int): Small regions below this size are set to -1
        connectivity (int): 1 for 4-neighbor, 2 for 8-neighbor connectivity

    Returns:
        np.ndarray: region IDs (1..N) and -1 for nodata/small components
    """
    regions = np.full(cluster_map.shape, -1, dtype=np.int32)
    next_region_id = 1

    structure = np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        dtype=np.int8,
    )
    if connectivity == 2:
        structure = np.ones((3, 3), dtype=np.int8)

    for cls in np.unique(cluster_map):
        if cls < 0:
            continue
        mask = cluster_map == cls
        comp, n_comp = label(mask, structure=structure)
        for comp_id in range(1, n_comp + 1):
            comp_mask = comp == comp_id
            size = int(comp_mask.sum())
            if size < min_region_size:
                continue
            regions[comp_mask] = next_region_id
            next_region_id += 1

    return regions


def segment_terrain(
    features,
    valid_mask=None,
    n_clusters=6,
    min_region_size=100,
    seed=42,
    normalize=True,
):
    """
    End-to-end terrain segmentation: features -> clusters -> connected regions.

    Returns:
        tuple[np.ndarray, np.ndarray, list[str]]:
            cluster_map, region_ids, feature_names
    """
    X, mask, names = stack_features(features, valid_mask=valid_mask, normalize=normalize)
    shape = next(iter(features.values())).shape
    cluster_map = kmeans_segment(X, mask, shape, n_clusters=n_clusters, seed=seed)
    region_ids = connected_regions(cluster_map, min_region_size=min_region_size)
    return cluster_map, region_ids, names
