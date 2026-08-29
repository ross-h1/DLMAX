"""sweep primitive tests."""

import pytest
from DLMAX.ffs.sweep import SweepDim, sweep, default_keyfn


def test_basic_cartesian_product():
    dims = [
        SweepDim("trend", "disc_rate", [1.0, 0.97], short="D"),
        SweepDim("error", "power", [0, 1], short="P"),
    ]
    cells = list(sweep(dims))
    assert len(cells) == 4
    keys = [k for k, _, _ in cells]
    assert "trend.D0|error.P0" in keys
    assert "trend.D1|error.P1" in keys


def test_constraint_filters_cells():
    dims = [
        SweepDim("a", "v", [1, 2, 3]),
        SweepDim("b", "v", [10, 20, 30]),
    ]
    constraint = lambda cell: cell["a"]["v"] <= cell["b"]["v"] / 10
    cells = list(sweep(dims, constraint=constraint))
    # Accepts: (1,10), (1,20), (1,30), (2,20), (2,30), (3,30) = 6 cells
    assert len(cells) == 6


def test_length_one_dim_yields_one_cell():
    dims = [
        SweepDim("a", "v", [42]),
        SweepDim("b", "v", [1, 2]),
    ]
    cells = list(sweep(dims))
    assert len(cells) == 2
    assert all(c["a"]["v"] == 42 for _, c, _ in cells)


def test_empty_values_rejected():
    with pytest.raises(ValueError, match="empty"):
        SweepDim("a", "v", [])


def test_non_list_values_rejected():
    with pytest.raises(TypeError, match="must be a list"):
        SweepDim("a", "v", (1, 2))


def test_cell_structure():
    dims = [
        SweepDim("trend", "disc_rate", [1.0]),
        SweepDim("trend", "damping", [0.99]),
        SweepDim("error", "disc_rate", [0.99]),
    ]
    ((key, cell, idx),) = list(sweep(dims))
    assert cell == {
        "trend": {"disc_rate": 1.0, "damping": 0.99},
        "error": {"disc_rate": 0.99},
    }
    assert idx == (0, 0, 0)


def test_constraint_sees_full_cell():
    """Constraint must receive the cell dict with all dims resolved."""
    dims = [
        SweepDim("seasonal", "disc_rate", [None, 0.95, 0.9]),
        SweepDim("trend", "disc_rate", [1.0, 0.97, 0.9]),
    ]
    seen = []

    def cstr(cell):
        seen.append(dict((k, dict(v)) for k, v in cell.items()))
        return True

    list(sweep(dims, constraint=cstr))
    assert len(seen) == 9
    assert all("seasonal" in c and "trend" in c for c in seen)


def test_indices_returned():
    dims = [
        SweepDim("a", "v", [10, 20]),
        SweepDim("b", "v", [100, 200, 300]),
    ]
    cells = list(sweep(dims))
    assert len(cells) == 6
    seen_idx = {idx for _, _, idx in cells}
    assert seen_idx == {(i, j) for i in range(2) for j in range(3)}
