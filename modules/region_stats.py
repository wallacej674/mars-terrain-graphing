import numpy as np
from rasterio.transform import xy


def assign_rule_based_label(mean_slope, mean_curvature, mean_roughness=None):
    """
    Basic starter taxonomy for region labels.
    Tune thresholds after inspecting your site-specific statistics.
    """
    if mean_slope < 5 and mean_curvature < -0.001:
        return "valley_or_depression"
    if mean_slope >= 10 and mean_curvature > 0.001:
        return "ridge_or_hill"
    if mean_slope < 5 and abs(mean_curvature) <= 0.001:
        return "plain"
    if mean_roughness is not None and mean_roughness > 10:
        return "rough_terrain"
    return "mixed"


def _region_centroid_lon_lat(region_mask, transform):
    rows, cols = np.where(region_mask)
    row_c = float(rows.mean())
    col_c = float(cols.mean())
    lon, lat = xy(transform, row_c, col_c)
    return float(lon), float(lat)


def compute_region_attributes(region_ids, features, transform, crs, tile_id):
    """
    Compute aggregated region attributes from pixel-level features.

    Args:
        region_ids (np.ndarray): region raster with IDs (1..N), -1 invalid
        features (dict[str, np.ndarray]): feature rasters aligned with region_ids
        transform (Affine): raster transform
        crs: raster CRS
        tile_id (str): source tile identifier

    Returns:
        list[dict]: one dictionary per region
    """
    attrs = []
    unique_ids = [rid for rid in np.unique(region_ids) if rid > 0]

    # m^2 per pixel from affine transform terms
    pixel_area_m2 = abs(transform.a * transform.e)

    roughness = features.get("roughness")
    for region_id in unique_ids:
        mask = region_ids == region_id
        area_m2 = float(mask.sum() * pixel_area_m2)
        mean_elevation = float(np.nanmean(features["elevation"][mask])) if "elevation" in features else None
        mean_slope = float(np.nanmean(features["slope"][mask])) if "slope" in features else None
        mean_curvature = float(np.nanmean(features["curvature"][mask])) if "curvature" in features else None
        mean_roughness = float(np.nanmean(roughness[mask])) if roughness is not None else None
        lon, lat = _region_centroid_lon_lat(mask, transform)

        label = assign_rule_based_label(
            mean_slope if mean_slope is not None else 0.0,
            mean_curvature if mean_curvature is not None else 0.0,
            mean_roughness=mean_roughness,
        )

        attrs.append(
            {
                "region_id": int(region_id),
                "tile_id": tile_id,
                "class_label": label,
                "crs": str(crs),
                "centroid_lon": lon,
                "centroid_lat": lat,
                "area_m2": area_m2,
                "mean_elevation": mean_elevation,
                "mean_slope": mean_slope,
                "mean_curvature": mean_curvature,
                "mean_roughness": mean_roughness,
            }
        )

    return attrs
