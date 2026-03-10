# modules/compute_features.py

import numpy as np
from scipy.ndimage import sobel

def compute_slope(elevation, pixel_size):
    """
    Computes slope using Sobel gradients.

    Args:
        elevation (np.ndarray): 2D elevation array.
        pixel_size (float): Spatial resolution in meters.

    Returns:
        slope (np.ndarray): Slope in degrees.
    """
    dzdx = sobel(elevation, axis=1) / (8 * pixel_size)
    dzdy = sobel(elevation, axis=0) / (8 * pixel_size)

    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope_deg = np.degrees(slope_rad)
    
    return slope_deg

def compute_aspect(elevation, pixel_size):
    """
    Computes aspect (direction of steepest slope).

    Returns:
        aspect (np.ndarray): Aspect in degrees [0-360).
    """
    dzdx = sobel(elevation, axis=1) / (8 * pixel_size)
    dzdy = sobel(elevation, axis=0) / (8 * pixel_size)

    aspect_rad = np.arctan2(dzdy, -dzdx)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = (aspect_deg + 360) % 360

    return aspect_deg

def compute_curvature(elevation, pixel_size):
    """
    Approximates curvature from second derivatives.

    Returns:
        curvature (np.ndarray): Relative curvature (unitless).
    """
    dzdx = sobel(elevation, axis=1) / (8 * pixel_size)
    dzdy = sobel(elevation, axis=0) / (8 * pixel_size)
    d2zdx2 = sobel(dzdx, axis=1) / (8 * pixel_size)
    d2zdy2 = sobel(dzdy, axis=0) / (8 * pixel_size)

    curvature = d2zdx2 + d2zdy2
    return curvature


def compute_roughness(elevation, window_size=5):
    """
    Compute local roughness as moving-window elevation std-dev.

    Args:
        elevation (np.ndarray): 2D elevation array.
        window_size (int): Odd moving-window size.

    Returns:
        np.ndarray: roughness raster in elevation units.
    """
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")

    pad = window_size // 2
    arr = np.asarray(elevation, dtype=np.float32)
    padded = np.pad(arr, pad_width=pad, mode="reflect")
    out = np.zeros_like(arr, dtype=np.float32)

    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            window = padded[r : r + window_size, c : c + window_size]
            out[r, c] = np.nanstd(window)

    return out
