"""Tests for logternary.projection (the matplotlib Axes subclass)."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")

import logternary  # noqa: F401  — registers projection
from logternary import LogTernaryAxes, to_xy


@pytest.fixture
def ltax():
    """Create a LogTernaryAxes and close the figure after the test."""
    fig, ax = plt.subplots(subplot_kw={"projection": "logternary"})
    yield ax
    plt.close(fig)


@pytest.fixture
def ltax_custom():
    """LogTernaryAxes with custom settings."""
    fig, ax = plt.subplots(
        subplot_kw={
            "projection": "logternary",
            "base": 10,
            "max_level": 2,
            "labels": ("X", "Y", "Z"),
        },
    )
    yield ax
    plt.close(fig)


class TestCreation:
    def test_axes_type(self, ltax):
        assert isinstance(ltax, LogTernaryAxes)

    def test_projection_name(self, ltax):
        assert ltax.name == "logternary"

    def test_default_base(self, ltax):
        assert ltax.base == 2.0

    def test_default_max_level(self, ltax):
        assert ltax.max_level == 3

    def test_custom_base(self, ltax_custom):
        assert ltax_custom.base == 10.0

    def test_custom_max_level(self, ltax_custom):
        assert ltax_custom.max_level == 2


class TestScatter:
    def test_three_args_transforms(self, ltax):
        """scatter(a, b, c) transforms to xy-space."""
        collection = ltax.scatter([4], [1], [1])
        offsets = collection.get_offsets()
        x_expected, y_expected = to_xy(4, 1, 1, base=2)
        assert offsets[0, 0] == pytest.approx(float(x_expected))
        assert offsets[0, 1] == pytest.approx(float(y_expected))

    def test_two_args_passthrough(self, ltax):
        """scatter(x, y) falls through to standard Axes.scatter."""
        collection = ltax.scatter([1.0], [2.0])
        offsets = collection.get_offsets()
        assert offsets[0, 0] == pytest.approx(1.0)
        assert offsets[0, 1] == pytest.approx(2.0)

    def test_array_input(self, ltax):
        a = np.array([1.0, 2.0, 4.0])
        b = np.ones(3)
        c = np.ones(3)
        collection = ltax.scatter(a, b, c)
        assert collection.get_offsets().shape == (3, 2)


class TestPlot:
    def test_three_args_transforms(self, ltax):
        """plot(a, b, c) transforms to xy-space."""
        lines = ltax.plot([1, 2, 4], [1, 1, 1], [1, 1, 1])
        xdata = lines[0].get_xdata()
        ydata = lines[0].get_ydata()
        x_expected, y_expected = to_xy([1, 2, 4], [1, 1, 1], [1, 1, 1], base=2)
        np.testing.assert_allclose(xdata, x_expected)
        np.testing.assert_allclose(ydata, y_expected)

    def test_format_string_passthrough(self, ltax):
        """plot(x, y, 'r--') falls through to standard plot."""
        lines = ltax.plot([0, 1], [0, 1], "r--")
        assert lines[0].get_color() == "r"


class TestAnnotate:
    def test_three_args_transforms(self, ltax):
        """annotate('text', a, b, c) places annotation at transformed position."""
        ann = ltax.annotate("test", 4, 1, 1)
        x_expected, y_expected = to_xy(4, 1, 1, base=2)
        # annotation xy is in data coords
        assert ann.xy[0] == pytest.approx(float(x_expected))
        assert ann.xy[1] == pytest.approx(float(y_expected))

    def test_tuple_passthrough(self, ltax):
        """annotate('text', (x, y)) falls through to standard annotate."""
        ann = ltax.annotate("test", (1.5, 2.5))
        assert ann.xy[0] == pytest.approx(1.5)
        assert ann.xy[1] == pytest.approx(2.5)


class TestSubplots:
    def test_multiple_subplots(self):
        """Two logternary axes in the same figure."""
        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            subplot_kw={"projection": "logternary", "base": 2, "max_level": 2},
        )
        assert isinstance(ax1, LogTernaryAxes)
        assert isinstance(ax2, LogTernaryAxes)
        ax1.scatter([1], [1], [1])
        ax2.scatter([2], [1], [1])
        plt.close(fig)

    def test_savefig(self, ltax, tmp_path):
        """Figure can be saved without errors."""
        ltax.scatter([1, 2, 4], [1, 1, 1], [1, 1, 1])
        path = tmp_path / "test.png"
        ltax.figure.savefig(path, dpi=50)
        assert path.exists()
        assert path.stat().st_size > 0
