"""Tests for logternary.transforms."""

import numpy as np
import pytest

from logternary import AXIS_VECTORS, from_xy, project_onto_axis, to_xy


class TestToXy:
    """Forward transform (a, b, c) → (x, y)."""

    def test_origin(self):
        """(1, 1, 1) maps to the origin."""
        x, y = to_xy(1, 1, 1)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)

    def test_projective_invariance(self):
        """(ka, kb, kc) maps to the same point as (a, b, c)."""
        a, b, c = 2.0, 3.0, 5.0
        x1, y1 = to_xy(a, b, c)
        for k in [0.1, 7, 100]:
            x2, y2 = to_xy(k * a, k * b, k * c)
            assert x2 == pytest.approx(x1)
            assert y2 == pytest.approx(y1)

    def test_mirror_symmetry(self):
        """r(1/a, 1/b, 1/c) = -r(a, b, c)."""
        a, b, c = 4.0, 1.0, 2.0
        x1, y1 = to_xy(a, b, c)
        x2, y2 = to_xy(1 / a, 1 / b, 1 / c)
        assert x2 == pytest.approx(-x1)
        assert y2 == pytest.approx(-y1)

    def test_a_axis_points_up(self):
        """(p, 1, 1) lies on the positive y-axis."""
        x, y = to_xy(4, 1, 1)
        assert x == pytest.approx(0.0)
        assert y > 0

    def test_b_axis_direction(self):
        """(1, p, 1) lies along the b-axis direction."""
        x, y = to_xy(1, 4, 1)
        # Should be proportional to AXIS_VECTORS[1]
        norm = np.sqrt(x**2 + y**2)
        assert x / norm == pytest.approx(AXIS_VECTORS[1, 0], abs=1e-10)
        assert y / norm == pytest.approx(AXIS_VECTORS[1, 1], abs=1e-10)

    def test_c_axis_direction(self):
        """(1, 1, p) lies along the c-axis direction."""
        x, y = to_xy(1, 1, 4)
        norm = np.sqrt(x**2 + y**2)
        assert x / norm == pytest.approx(AXIS_VECTORS[2, 0], abs=1e-10)
        assert y / norm == pytest.approx(AXIS_VECTORS[2, 1], abs=1e-10)

    def test_array_input(self):
        """Accepts array input and returns arrays."""
        a = np.array([1.0, 2.0, 4.0])
        b = np.ones(3)
        c = np.ones(3)
        x, y = to_xy(a, b, c)
        assert x.shape == (3,)
        assert y.shape == (3,)

    def test_base10(self):
        """Base 10: fold-change of 10 = 1 unit on the a-axis."""
        x, y = to_xy(10, 1, 1, base=10)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(1.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="positive"):
            to_xy(-1, 1, 1)

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            to_xy(0, 1, 1)


class TestRoundtrip:
    """Forward then inverse recovers the original ratios."""

    @pytest.mark.parametrize("base", [2.0, np.e, 10.0])
    def test_roundtrip(self, base):
        rng = np.random.default_rng(42)
        a = rng.uniform(0.1, 10, 50)
        b = rng.uniform(0.1, 10, 50)
        c = rng.uniform(0.1, 10, 50)

        x, y = to_xy(a, b, c, base)
        a2, b2, c2 = from_xy(x, y, base)

        # Ratios must match (c2 == 1, so normalise originals by c)
        assert (a2 / b2) == pytest.approx(a / b, rel=1e-10)
        assert (a2 / c2) == pytest.approx(a / c, rel=1e-10)


class TestCollinearity:
    """Points (a·pⁿ, b·qⁿ, c·sⁿ) are collinear in xy-space."""

    def test_collinear(self):
        a0, b0, c0 = 1.0, 1.0, 1.0
        p, q, s = 1.5, 2.5, 0.8
        ns = np.arange(-5, 6)
        a_vals = np.array([a0 * p**n for n in ns])
        b_vals = np.array([b0 * q**n for n in ns])
        c_vals = np.array([c0 * s**n for n in ns])

        x, y = to_xy(a_vals, b_vals, c_vals)

        # All points lie on a line through the origin.
        # Check via cross product: (x1*y2 - x2*y1) ≈ 0 for all pairs with [0].
        if np.abs(x[0]) + np.abs(y[0]) > 1e-12:
            cross = x * y[0] - y * x[0]
        else:
            cross = x * y[1] - y * x[1]
        np.testing.assert_allclose(cross, 0, atol=1e-10)


class TestProjectOntoAxis:
    def test_a_axis_projection(self):
        """A point on the a-axis projects its full magnitude onto 'a'."""
        x, y = to_xy(4, 1, 1, base=2)
        k = project_onto_axis(x, y, "a")
        assert k == pytest.approx(2.0)  # log2(4) = 2

    def test_orthogonal_projection_symmetric(self):
        """A point on the a-axis projects equally (and negatively) onto b and c."""
        x, y = to_xy(8, 1, 1, base=2)
        proj_b = project_onto_axis(x, y, "b")
        proj_c = project_onto_axis(x, y, "c")
        assert proj_b == pytest.approx(proj_c)
        assert proj_b < 0

    def test_projections_sum_to_zero(self):
        """The three axis projections always sum to zero."""
        rng = np.random.default_rng(99)
        for _ in range(20):
            a, b, c = rng.uniform(0.1, 10, 3)
            x, y = to_xy(a, b, c, base=2)
            total = (
                project_onto_axis(x, y, "a")
                + project_onto_axis(x, y, "b")
                + project_onto_axis(x, y, "c")
            )
            assert total == pytest.approx(0.0, abs=1e-10)
