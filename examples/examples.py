"""
Examples demonstrating the log-ternary plot as a matplotlib projection.

Recreates elements from Figure 3 of the paper and shows additional features.
Saves all plots to docs/images/ for use in the README.

Usage::

    uv run python examples/examples.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import logternary  # noqa: F401  — registers 'logternary' projection

# Output directory (relative to repo root)
OUT = Path(__file__).resolve().parent.parent / "docs" / "images"


def example_figure3():
    """Recreate key elements from Figure 3: axes, mirrored points, collinear points."""
    fig, ax = plt.subplots(
        figsize=(7, 7),
        subplot_kw={
            "projection": "logternary",
            "base": 2,
            "max_level": 3,
            "labels": ("a", "b", "c"),
        },
    )

    # Origin: no fold-change
    ax.scatter([1], [1], [1], color="gray", s=60, zorder=10, marker="o")
    ax.annotate(
        "(1,1,1)", 1, 1, 1, xytext=(8, 4), textcoords="offset points", fontsize=8, color="gray"
    )

    # Orange mirrored pair
    ax.scatter([4, 1 / 4], [1, 1], [2, 1 / 2], color="#e67e22", s=60, zorder=10)
    ax.annotate(
        "(4,1,2)", 4, 1, 2, xytext=(5, 5), textcoords="offset points", fontsize=8, color="#e67e22"
    )
    ax.annotate(
        "(¼,1,½)",
        1 / 4,
        1,
        1 / 2,
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
        color="#e67e22",
    )

    # Green collinear points: (a·pⁿ, b·qⁿ, c·sⁿ) for different n
    a0, b0, c0 = 1, 1, 1
    p, q, s = 1.5, 2.5, 0.8
    ns = np.arange(-3, 5)
    a_vals = [a0 * p**n for n in ns]
    b_vals = [b0 * q**n for n in ns]
    c_vals = [c0 * s**n for n in ns]
    ax.plot(a_vals, b_vals, c_vals, color="#27ae60", lw=1, ls="--", zorder=4)
    ax.scatter(a_vals, b_vals, c_vals, color="#27ae60", s=40, zorder=10)

    # Blue transformation demo
    gray_a, gray_b, gray_c = 1.5, 0.8, 1.2
    ax.scatter([gray_a], [gray_b], [gray_c], color="gray", s=50, zorder=10)

    beta = 2
    transforms = [
        (gray_a * beta, gray_b, gray_c, "a→βa"),
        (gray_a / beta, gray_b, gray_c, "a→a/β"),
        (gray_a, gray_b * beta, gray_c, "b→βb"),
        (gray_a, gray_b / beta, gray_c, "b→b/β"),
        (gray_a, gray_b, gray_c * beta, "c→βc"),
        (gray_a, gray_b, gray_c / beta, "c→c/β"),
    ]
    for ta, tb, tc, lbl in transforms:
        ax.scatter([ta], [tb], [tc], color="#2980b9", s=40, zorder=10)
        ax.annotate(
            lbl, ta, tb, tc, xytext=(5, 5), textcoords="offset points", fontsize=7, color="#2980b9"
        )

    ax.set_title("Log-Ternary Plot — Key Properties", fontsize=13, pad=15)
    return fig


def example_transcriptome():
    """Show random transcriptome-like data across three conditions."""
    rng = np.random.default_rng(42)

    n_genes = 500
    log_fc = rng.normal(0, 0.5, size=(n_genes, 3))
    n_de = 40
    log_fc[:n_de, 0] += 1.5
    log_fc[:n_de, 1] -= 0.5
    vals = 2.0**log_fc

    fig, ax = plt.subplots(
        figsize=(7, 7),
        subplot_kw={
            "projection": "logternary",
            "base": 2,
            "max_level": 3,
            "labels": ("Wild type", "Knockout", "Rescue"),
        },
    )

    ax.scatter(
        vals[n_de:, 0],
        vals[n_de:, 1],
        vals[n_de:, 2],
        color="#95a5a6",
        s=8,
        alpha=0.5,
        label="Unchanged",
    )
    ax.scatter(
        vals[:n_de, 0],
        vals[:n_de, 1],
        vals[:n_de, 2],
        color="#e74c3c",
        s=15,
        alpha=0.8,
        label="DE genes",
    )

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_title("Simulated Transcriptome Data", fontsize=13, pad=15)
    return fig


def example_exchange_rates():
    """Show exchange-rate trajectories as an economics application."""
    rng = np.random.default_rng(123)
    n_months = 24

    log_a = np.cumsum(rng.normal(0, 0.03, n_months))
    log_b = np.cumsum(rng.normal(0.01, 0.04, n_months))
    log_c = np.cumsum(rng.normal(-0.005, 0.025, n_months))
    a, b, c = np.exp(log_a), np.exp(log_b), np.exp(log_c)

    fig, ax = plt.subplots(
        figsize=(7, 7),
        subplot_kw={
            "projection": "logternary",
            "base": 2,
            "max_level": 2,
            "labels": ("USD", "EUR", "GBP"),
        },
    )

    ax.plot(a, b, c, color="#3498db", lw=1.5, alpha=0.7)
    ax.scatter([a[0]], [b[0]], [c[0]], color="#27ae60", s=80, zorder=10, marker="s", label="Start")
    ax.scatter(
        [a[-1]], [b[-1]], [c[-1]], color="#e74c3c", s=80, zorder=10, marker="D", label="End"
    )

    for i in [6, 12, 18]:
        ax.scatter([a[i]], [b[i]], [c[i]], color="#f39c12", s=50, zorder=10)
        ax.annotate(
            f"Month {i}", a[i], b[i], c[i], xytext=(7, 5), textcoords="offset points", fontsize=8
        )

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_title("Exchange Rate Trajectory (simulated)", fontsize=13, pad=15)
    return fig


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)

    save_kw = dict(bbox_inches="tight", dpi=150)

    fig = example_figure3()
    fig.savefig(OUT / "example_figure3.png", **save_kw)
    print(f"Saved {OUT / 'example_figure3.png'}")

    fig = example_transcriptome()
    fig.savefig(OUT / "example_transcriptome.png", **save_kw)
    print(f"Saved {OUT / 'example_transcriptome.png'}")

    fig = example_exchange_rates()
    fig.savefig(OUT / "example_exchange_rates.png", **save_kw)
    print(f"Saved {OUT / 'example_exchange_rates.png'}")

    plt.close("all")
    print("\nAll examples saved.")
