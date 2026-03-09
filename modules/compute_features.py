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