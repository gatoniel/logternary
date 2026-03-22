"""
Core coordinate transformations for the log-ternary plot.

The log-ternary plot maps points (a, b, c) from the positive projective space
A₂ = {x ∈ RP², xᵢ > 0} to R² using the diffeomorphism:

    r(a, b, c) = ( √3/2 · log_β(b/c),  log_β(a/c) − ½ · log_β(b/c) )

This is equivalent to an isometric log-ratio (ILR) transform with a symmetric
contrast matrix that yields three axes at 0°, 120°, and 240°.

Reference:
    Netter, N. (2023). Visualization and analysis of fold-changes between
    multiple conditions.
"""

import numpy as np

# Axis unit vectors (directions of the three axes in xy-space)
# a-axis: up (0°)
# b-axis: lower-right (-120° from a, i.e. 330° from +x = -30°)
# c-axis: lower-left (+120° from a, i.e. 210° from +x)
AXIS_VECTORS = np.array(
    [
        [0.0, 1.0],  # a
        [np.sqrt(3) / 2, -0.5],  # b
        [-np.sqrt(3) / 2, -0.5],  # c
    ]
)


def to_xy(a, b, c, base=2.0):
    """
    Transform (a, b, c) coordinates to log-ternary (x, y) coordinates.

    Parameters
    ----------
    a, b, c : array_like
        Positive values representing the three conditions.
        Only ratios matter — (a, b, c) and (ka, kb, kc) map to the same point.
    base : float
        Logarithm base. Controls the scale: a fold-change of `base` in one
        condition corresponds to one unit along the respective axis.

    Returns
    -------
    x, y : ndarray
        Coordinates in the log-ternary plane.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    if np.any(a <= 0) or np.any(b <= 0) or np.any(c <= 0):
        raise ValueError("All values must be strictly positive.")

    log_bc = np.log(b / c) / np.log(base)
    log_ac = np.log(a / c) / np.log(base)

    x = np.sqrt(3) / 2 * log_bc
    y = log_ac - 0.5 * log_bc

    return x, y


def from_xy(x, y, base=2.0):
    """
    Inverse transform: log-ternary (x, y) back to a representative (a, b, c).

    The third component c is fixed to 1. The returned (a, b, c) is one
    representative of the equivalence class in projective space.

    Parameters
    ----------
    x, y : array_like
        Log-ternary coordinates.
    base : float
        Logarithm base (must match the forward transform).

    Returns
    -------
    a, b, c : ndarray
        Representative point with c = 1.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    a = base ** (y + x / np.sqrt(3))
    b = base ** (2 * x / np.sqrt(3))
    c = np.ones_like(a)

    return a, b, c


def project_onto_axis(x, y, axis="a"):
    """
    Project an (x, y) point onto one of the three axes.

    The projection value k means that the corresponding condition has a
    fold-change of base^k relative to the geometric mean of the other two.

    Parameters
    ----------
    x, y : array_like
        Log-ternary coordinates.
    axis : str
        One of 'a', 'b', 'c'.

    Returns
    -------
    k : ndarray
        Projection value (in units of log_base fold-change).
    """
    idx = {"a": 0, "b": 1, "c": 2}[axis]
    e = AXIS_VECTORS[idx]
    return np.asarray(x) * e[0] + np.asarray(y) * e[1]
