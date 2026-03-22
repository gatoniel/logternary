"""
logternary — Log-ternary plots as a native matplotlib projection.

A log-ternary plot maps positive triples (a, b, c) — where only ratios carry
meaning — to ℝ² via a symmetric isometric log-ratio transform.  The resulting
plot has three axes at 120° separation, each representing fold-changes in one
condition relative to the geometric mean of the other two.

Quick start::

    import matplotlib.pyplot as plt
    import logternary                # registers the 'logternary' projection

    fig, ax = plt.subplots(
        subplot_kw={'projection': 'logternary', 'base': 2, 'max_level': 3,
                     'labels': ('Wild type', 'Knockout', 'Rescue')}
    )
    ax.scatter(a, b, c, color='steelblue', s=12)
    ax.grid(False)      # toggle the triangular grid
    plt.show()

Importing the package is sufficient to register the projection; there is no
need to call any setup function.
"""

from .projection import LogTernaryAxes  # registers 'logternary'
from .transforms import AXIS_VECTORS, from_xy, project_onto_axis, to_xy

__version__ = "0.1.0"
__all__ = [
    "AXIS_VECTORS",
    "LogTernaryAxes",
    "from_xy",
    "project_onto_axis",
    "to_xy",
]
