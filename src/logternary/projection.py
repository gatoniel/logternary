"""
Matplotlib projection subclass for the log-ternary plot.

Registers ``'logternary'`` as a matplotlib projection so that it can be
used with the standard subplot machinery::

    import logternary  # registers the projection on import

    fig, ax = plt.subplots(
        subplot_kw={'projection': 'logternary', 'base': 2, 'max_level': 3}
    )
    ax.scatter(a, b, c, color='steelblue', s=12)

The three-argument forms ``ax.scatter(a, b, c)``, ``ax.plot(a, b, c)``,
and ``ax.annotate(text, a, b, c)`` are overridden to accept (a, b, c)
coordinates in the positive projective space and transform them
automatically.  Two-argument calls fall through to standard matplotlib
behaviour, so internal xy-space is still accessible when needed.

Grid visibility is toggled with ``ax.grid(True)`` / ``ax.grid(False)``.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.projections import register_projection

from .transforms import AXIS_VECTORS, from_xy, to_xy

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
_GRID_COLOR = "#b0b0b0"
_GRID_ALPHA = 0.35
_GRID_LW = 0.5
_AXIS_COLOR = "#222222"
_AXIS_LW = 1.4
_TICK_FONTSIZE = 8
_LABEL_FONTSIZE = 12


class LogTernaryAxes(Axes):
    """
    A matplotlib ``Axes`` subclass for log-ternary plots.

    The projection maps positive triples (a, b, c) — where only ratios
    carry meaning — to ℝ² via the symmetric isometric log-ratio transform
    described in Netter (2023).  The resulting plot has three axes at 120°
    separation, each representing fold-changes in one condition relative to
    the geometric mean of the other two.

    Parameters (passed via *subplot_kw*)
    -------------------------------------
    base : float
        Logarithm base.  A fold-change of *base* in one condition equals
        one grid unit.  Common choices: 2 (default), 10, e.
    max_level : int
        Number of grid levels per axis direction.  For ``base=2,
        max_level=3`` the grid spans fold-changes from 1/8× to 8×.
    labels : tuple[str, str, str]
        Labels for the three axes (a, b, c).
    tick_format : str or callable
        ``'fold'`` shows fold-change values (e.g. "2×", "¼×");
        ``'log'`` shows exponents; or pass ``f(level, base) -> str``.
    """

    name = "logternary"

    def __init__(
        self, *args, base=2.0, max_level=3, labels=("a", "b", "c"), tick_format="fold", **kwargs
    ):
        # Store configuration *before* super().__init__, which calls clear().
        self._lt_base = float(base)
        self._lt_max_level = int(max_level)
        self._lt_labels = tuple(labels)
        self._lt_tick_format = tick_format
        self._lt_show_grid = True
        self._lt_clearing = False

        # Artist bookkeeping (populated in _draw_decorations)
        self._lt_grid_lc: LineCollection | None = None
        self._lt_tick_artists: list = []
        self._lt_label_artists: list = []
        self._lt_axis_artists: list = []

        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def base(self) -> float:
        """Logarithm base of this axes."""
        return self._lt_base

    @property
    def max_level(self) -> int:
        """Number of grid levels per axis direction."""
        return self._lt_max_level

    # ------------------------------------------------------------------
    # Axes lifecycle
    # ------------------------------------------------------------------

    def clear(self):
        """Reset the axes and redraw log-ternary decorations."""
        # Reset artist lists *before* super().clear() removes them
        self._lt_grid_lc = None
        self._lt_tick_artists = []
        self._lt_label_artists = []
        self._lt_axis_artists = []

        # Suppress internal grid() calls from matplotlib during clear
        self._lt_clearing = True
        try:
            super().clear()
        finally:
            self._lt_clearing = False

        self._setup_view()
        self._draw_decorations()

    # ------------------------------------------------------------------
    # Public API — overridden plotting methods
    # ------------------------------------------------------------------

    def scatter(self, *args, **kwargs):
        """
        Scatter plot in (a, b, c) coordinates.

        If three positional array-like arguments are given they are
        interpreted as ``(a, b, c)`` and transformed to log-ternary
        xy-space.  Additional positional arguments (e.g. sizes) and all
        keyword arguments are forwarded to ``Axes.scatter``.

        Falls back to standard ``Axes.scatter(x, y, …)`` when fewer
        than three positional arguments are given.
        """
        if len(args) >= 3 and not isinstance(args[2], str):
            a, b, c = args[:3]
            x, y = to_xy(a, b, c, self._lt_base)
            kwargs.setdefault("zorder", 5)
            kwargs.setdefault("s", 20)
            return super().scatter(x, y, *args[3:], **kwargs)
        return super().scatter(*args, **kwargs)

    def plot(self, *args, **kwargs):
        """
        Line plot in (a, b, c) coordinates.

        Three positional array-like arguments are interpreted as
        ``(a, b, c)``.  A string third argument (format specifier like
        ``'r--'``) triggers normal ``Axes.plot(x, y, fmt)`` behaviour.
        """
        if len(args) >= 3 and not isinstance(args[2], str):
            a, b, c = args[:3]
            x, y = to_xy(a, b, c, self._lt_base)
            kwargs.setdefault("zorder", 5)
            return super().plot(x, y, *args[3:], **kwargs)
        return super().plot(*args, **kwargs)

    def annotate(self, text, *args, **kwargs):
        """
        Text annotation at (a, b, c).

        ``ax.annotate('label', a, b, c, xytext=…)`` transforms the
        coordinates and forwards everything else to ``Axes.annotate``.
        The standard two-argument form ``ax.annotate('label', (x, y))``
        is also supported.
        """
        if len(args) >= 3 and not isinstance(args[0], tuple):
            a, b, c = args[:3]
            x, y = to_xy(a, b, c, self._lt_base)
            xy = (float(np.squeeze(x)), float(np.squeeze(y)))
            return super().annotate(text, xy, *args[3:], **kwargs)
        return super().annotate(text, *args, **kwargs)

    def fill(self, *args, **kwargs):
        """Filled polygon in (a, b, c) coordinates."""
        if len(args) >= 3 and not isinstance(args[2], str):
            a, b, c = args[:3]
            x, y = to_xy(a, b, c, self._lt_base)
            return super().fill(x, y, *args[3:], **kwargs)
        return super().fill(*args, **kwargs)

    # ------------------------------------------------------------------
    # Convenience methods (not overrides of Axes methods)
    # ------------------------------------------------------------------

    def arrow_abc(self, a1, b1, c1, a2, b2, c2, **kwargs):
        """Draw an arrow from (a1, b1, c1) to (a2, b2, c2)."""
        x1, y1 = to_xy(a1, b1, c1, self._lt_base)
        x2, y2 = to_xy(a2, b2, c2, self._lt_base)
        x1, y1 = float(np.squeeze(x1)), float(np.squeeze(y1))
        x2, y2 = float(np.squeeze(x2)), float(np.squeeze(y2))
        kwargs.setdefault("color", _AXIS_COLOR)
        kwargs.setdefault("zorder", 5)
        kwargs.setdefault("head_width", 0.06)
        kwargs.setdefault("head_length", 0.04)
        kwargs.setdefault("length_includes_head", True)
        return super().arrow(x1, y1, x2 - x1, y2 - y1, **kwargs)

    def hexbin(self, *args, **kwargs):
        """
        Hexagonal binning in (a, b, c) coordinates.

        Three positional arguments are transformed; two are passed
        through directly.
        """
        if len(args) >= 3:
            a, b, c = args[:3]
            x, y = to_xy(a, b, c, self._lt_base)
            kwargs.setdefault("gridsize", 30)
            kwargs.setdefault("cmap", "YlOrRd")
            kwargs.setdefault("mincnt", 1)
            kwargs.setdefault("zorder", 3)
            kwargs.setdefault("linewidths", 0.2)
            return super().hexbin(x, y, *args[3:], **kwargs)
        return super().hexbin(*args, **kwargs)

    def contour_abc(self, func, n_grid=200, **kwargs):
        """
        Contour lines of ``func(a, b, c) -> scalar`` over the visible area.
        """
        xlim = self.get_xlim()
        ylim = self.get_ylim()
        xg = np.linspace(xlim[0], xlim[1], n_grid)
        yg = np.linspace(ylim[0], ylim[1], n_grid)
        X, Y = np.meshgrid(xg, yg)
        A, B, C = from_xy(X, Y, self._lt_base)
        Z = func(A, B, C)
        kwargs.setdefault("zorder", 3)
        return super().contour(X, Y, Z, **kwargs)

    def contourf_abc(self, func, n_grid=200, **kwargs):
        """
        Filled contour of ``func(a, b, c) -> scalar`` over the visible area.
        """
        xlim = self.get_xlim()
        ylim = self.get_ylim()
        xg = np.linspace(xlim[0], xlim[1], n_grid)
        yg = np.linspace(ylim[0], ylim[1], n_grid)
        X, Y = np.meshgrid(xg, yg)
        A, B, C = from_xy(X, Y, self._lt_base)
        Z = func(A, B, C)
        kwargs.setdefault("zorder", 3)
        return super().contourf(X, Y, Z, **kwargs)

    def density_abc(self, a, b, c, n_grid=100, **kwargs):
        """
        Gaussian KDE density plot in (a, b, c) coordinates.

        Requires scipy.
        """
        from scipy.stats import gaussian_kde

        x, y = to_xy(a, b, c, self._lt_base)
        kde = gaussian_kde(np.vstack([x, y]))
        xlim = self.get_xlim()
        ylim = self.get_ylim()
        xg = np.linspace(xlim[0], xlim[1], n_grid)
        yg = np.linspace(ylim[0], ylim[1], n_grid)
        X, Y = np.meshgrid(xg, yg)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        kwargs.setdefault("cmap", "Blues")
        kwargs.setdefault("zorder", 2)
        kwargs.setdefault("alpha", 0.6)
        return super().contourf(X, Y, Z, **kwargs)

    # ------------------------------------------------------------------
    # Grid control
    # ------------------------------------------------------------------

    def grid(self, visible=None, which=None, **kwargs):
        """
        Toggle or set grid visibility.

        Parameters
        ----------
        visible : bool or None
            *True* / *False* to show / hide.  *None* toggles.
        which, **kwargs : ignored
            Accepted for compatibility with matplotlib's internal calls.
        """
        # Ignore grid() calls triggered internally by super().clear()
        if getattr(self, "_lt_clearing", False):
            return
        if visible is None:
            self._lt_show_grid = not self._lt_show_grid
        else:
            self._lt_show_grid = bool(visible)
        if self._lt_grid_lc is not None:
            self._lt_grid_lc.set_visible(self._lt_show_grid)

    def set_labels(self, a_label: str, b_label: str, c_label: str):
        """Replace axis labels."""
        self._lt_labels = (a_label, b_label, c_label)
        for art in self._lt_label_artists:
            art.remove()
        self._lt_label_artists = []
        self._draw_labels()

    # ------------------------------------------------------------------
    # Internal drawing helpers
    # ------------------------------------------------------------------

    def _setup_view(self):
        self.set_aspect("equal")
        self.axis("off")
        pad = 0.6
        s3h = np.sqrt(3) / 2
        ml = self._lt_max_level
        self.set_xlim(-s3h * ml - pad, s3h * ml + pad)
        self.set_ylim(-ml - pad, ml + pad)

    def _draw_decorations(self):
        self._draw_grid()
        self._draw_axis_lines()
        self._draw_ticks()
        self._draw_labels()

    # ---- grid ----

    def _draw_grid(self):
        ml = self._lt_max_level
        s3 = np.sqrt(3)
        xlim = self.get_xlim()
        ylim = self.get_ylim()

        segments = []
        for k in range(-ml, ml + 1):
            # a-family: horizontal  y = k
            segments.append([(xlim[0], k), (xlim[1], k)])
            # b-family: y = √3·x − 2k
            seg = self._clip_line(s3, -2 * k, xlim, ylim)
            if seg is not None:
                segments.append(seg)
            # c-family: y = −√3·x − 2k
            seg = self._clip_line(-s3, -2 * k, xlim, ylim)
            if seg is not None:
                segments.append(seg)

        lc = LineCollection(
            segments,
            colors=_GRID_COLOR,
            linewidths=_GRID_LW,
            alpha=_GRID_ALPHA,
            zorder=1,
        )
        lc.set_visible(self._lt_show_grid)
        self.add_collection(lc)
        self._lt_grid_lc = lc

    @staticmethod
    def _clip_line(slope, intercept, xlim, ylim):
        """Clip ``y = slope * x + intercept`` to the bounding box."""
        pts = []
        for xb in xlim:
            yv = slope * xb + intercept
            if ylim[0] - 1e-9 <= yv <= ylim[1] + 1e-9:
                pts.append((xb, yv))
        if slope != 0:
            for yb in ylim:
                xv = (yb - intercept) / slope
                if xlim[0] - 1e-9 <= xv <= xlim[1] + 1e-9:
                    pts.append((xv, yb))
        # Deduplicate
        unique = []
        for p in pts:
            if not any(abs(p[0] - q[0]) < 1e-9 and abs(p[1] - q[1]) < 1e-9 for q in unique):
                unique.append(p)
        return unique[:2] if len(unique) >= 2 else None

    # ---- axes ----

    def _draw_axis_lines(self):
        ml = self._lt_max_level
        arrow_len = ml + 0.3

        for i in range(3):
            ex, ey = AXIS_VECTORS[i]
            # Positive direction arrow
            ann = Axes.annotate(
                self,
                "",
                xy=(ex * arrow_len, ey * arrow_len),
                xytext=(0, 0),
                arrowprops=dict(
                    arrowstyle="->",
                    color=_AXIS_COLOR,
                    lw=_AXIS_LW,
                    shrinkA=0,
                    shrinkB=0,
                    mutation_scale=14,
                ),
                zorder=4,
            )
            self._lt_axis_artists.append(ann)
            # Negative direction (thin line)
            (line,) = Axes.plot(
                self,
                [0, -ex * arrow_len * 0.85],
                [0, -ey * arrow_len * 0.85],
                color=_AXIS_COLOR,
                lw=_AXIS_LW * 0.5,
                zorder=4,
                solid_capstyle="round",
            )
            self._lt_axis_artists.append(line)

    # ---- ticks ----

    def _draw_ticks(self):
        ml = self._lt_max_level
        # Perpendicular directions (rotated 90° clockwise)
        perp_dirs = [(AXIS_VECTORS[i][1], -AXIS_VECTORS[i][0]) for i in range(3)]

        for i in range(3):
            ex, ey = AXIS_VECTORS[i]
            perp_x, perp_y = perp_dirs[i]

            for k in range(-ml, ml + 1):
                if k == 0:
                    continue
                px, py = ex * k, ey * k

                # Small tick mark
                tick_len = 0.06
                (line,) = Axes.plot(
                    self,
                    [px - perp_x * tick_len, px + perp_x * tick_len],
                    [py - perp_y * tick_len, py + perp_y * tick_len],
                    color=_AXIS_COLOR,
                    lw=0.8,
                    zorder=4,
                )
                self._lt_tick_artists.append(line)

                # Label text
                offset = 0.22
                label = self._format_tick(k)
                color = "#444444" if k > 0 else "#888888"
                fontsize = _TICK_FONTSIZE if k > 0 else _TICK_FONTSIZE - 0.5
                txt = Axes.text(
                    self,
                    px + perp_x * offset,
                    py + perp_y * offset,
                    label,
                    fontsize=fontsize,
                    ha="center",
                    va="center",
                    color=color,
                    zorder=6,
                )
                self._lt_tick_artists.append(txt)

    def _format_tick(self, level):
        if callable(self._lt_tick_format):
            return self._lt_tick_format(level, self._lt_base)
        if self._lt_tick_format == "log":
            return str(level)
        # 'fold' format
        val = self._lt_base**level
        if val >= 1:
            if val == int(val):
                return f"{int(val)}×"
            return f"{val:.1f}×"
        else:
            inv = self._lt_base ** (-level)
            if inv == int(inv):
                return f"1/{int(inv)}×"
            return f"{val:.2g}×"

    # ---- axis labels ----

    def _draw_labels(self):
        ml = self._lt_max_level
        label_dist = ml + 0.65

        for i, label in enumerate(self._lt_labels):
            ex, ey = AXIS_VECTORS[i]
            txt = Axes.text(
                self,
                ex * label_dist,
                ey * label_dist,
                label,
                fontsize=_LABEL_FONTSIZE,
                fontweight="bold",
                ha="center",
                va="center",
                color=_AXIS_COLOR,
                zorder=6,
            )
            self._lt_label_artists.append(txt)


# ---------------------------------------------------------------------------
# Register so that  subplot_kw={'projection': 'logternary'}  works.
# ---------------------------------------------------------------------------
register_projection(LogTernaryAxes)
