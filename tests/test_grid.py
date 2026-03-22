"""Tests for grid toggling and axis label updates."""

import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")

import logternary  # noqa: F401


@pytest.fixture
def ltax():
    fig, ax = plt.subplots(subplot_kw={"projection": "logternary"})
    yield ax
    plt.close(fig)


class TestGrid:
    def test_grid_on_by_default(self, ltax):
        assert ltax._lt_show_grid is True
        assert ltax._lt_grid_lc is not None
        assert ltax._lt_grid_lc.get_visible() is True

    def test_grid_off(self, ltax):
        ltax.grid(False)
        assert ltax._lt_show_grid is False
        assert ltax._lt_grid_lc.get_visible() is False

    def test_grid_toggle(self, ltax):
        ltax.grid()  # None → toggle off
        assert ltax._lt_show_grid is False
        ltax.grid()  # toggle on
        assert ltax._lt_show_grid is True

    def test_grid_on_after_off(self, ltax):
        ltax.grid(False)
        ltax.grid(True)
        assert ltax._lt_show_grid is True
        assert ltax._lt_grid_lc.get_visible() is True


class TestLabels:
    def test_default_labels(self, ltax):
        assert ltax._lt_labels == ("a", "b", "c")

    def test_set_labels(self, ltax):
        ltax.set_labels("X", "Y", "Z")
        assert ltax._lt_labels == ("X", "Y", "Z")
        # Check that label artists were recreated
        assert len(ltax._lt_label_artists) == 3
        texts = [art.get_text() for art in ltax._lt_label_artists]
        assert texts == ["X", "Y", "Z"]

    def test_custom_labels_at_creation(self):
        fig, ax = plt.subplots(
            subplot_kw={
                "projection": "logternary",
                "labels": ("WT", "KO", "Rescue"),
            },
        )
        assert ax._lt_labels == ("WT", "KO", "Rescue")
        plt.close(fig)


class TestTickFormat:
    def test_fold_format(self):
        fig, ax = plt.subplots(
            subplot_kw={"projection": "logternary", "tick_format": "fold"},
        )
        assert ax._format_tick(1) == "2×"
        assert ax._format_tick(2) == "4×"
        assert ax._format_tick(-1) == "1/2×"
        assert ax._format_tick(-3) == "1/8×"
        plt.close(fig)

    def test_log_format(self):
        fig, ax = plt.subplots(
            subplot_kw={"projection": "logternary", "tick_format": "log"},
        )
        assert ax._format_tick(1) == "1"
        assert ax._format_tick(-2) == "-2"
        plt.close(fig)

    def test_custom_format(self):
        def my_fmt(level, base):
            return f"{base**level:.1f}"

        fig, ax = plt.subplots(
            subplot_kw={"projection": "logternary", "tick_format": my_fmt},
        )
        assert ax._format_tick(2) == "4.0"
        plt.close(fig)
