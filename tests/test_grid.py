"""Unit tests for the initial 2D structured BackgroundGrid."""

import numpy as np
import pytest

from geopympm.core import BackgroundGrid, ParticleSet


def test_grid_geometry_and_flattening():
    grid = BackgroundGrid(origin=[1.0, -1.0], cell_size=[0.5, 0.25], n_cells=[3, 2])
    assert grid.ndim == 2
    assert grid.n_cells == (3, 2)
    assert grid.node_shape == (4, 3)
    assert grid.n_nodes == 12
    assert grid.n_elements == 6
    assert grid.node_coordinates.shape == (12, 2)
    assert np.allclose(grid.node_coordinates[0], [1.0, -1.0])
    assert np.allclose(grid.node_coordinates[3], [2.5, -1.0])
    assert np.allclose(grid.node_coordinates[4], [1.0, -0.75])
    assert np.allclose(grid.node_coordinates[-1], [2.5, -0.5])
    assert grid.node_coordinates.flags.c_contiguous


def test_node_id_and_cell_corners():
    grid = BackgroundGrid(origin=(0, 0), cell_size=(0.5, 0.5), n_cells=(2, 1))
    assert grid.node_id(0, 0) == 0
    assert grid.node_id(2, 0) == 2
    assert grid.node_id(0, 1) == 3
    assert grid.node_id(2, 1) == 5
    assert np.array_equal(grid.cell_node_ids(0, 0), [0, 1, 4, 3])
    assert np.array_equal(grid.cell_node_ids(1, 0), [1, 2, 5, 4])


def test_nodal_state_shapes_and_defaults():
    grid = BackgroundGrid((0, 0), (1, 2), (2, 3))
    assert grid.mass.shape == (12,)
    for name in ("momentum", "velocity", "acceleration", "force_internal", "force_external"):
        array = getattr(grid, name)
        assert array.shape == (12, 2)
        assert array.dtype == np.float64
        assert array.flags.c_contiguous
        assert np.all(array == 0)
    assert grid.velocity_bc_mask.shape == (12, 2)
    assert grid.velocity_bc_mask.dtype == np.bool_
    assert not np.any(grid.velocity_bc_mask)
    assert np.all(grid.velocity_bc_value == 0)


def test_reset_keeps_geometry_and_constraints():
    grid = BackgroundGrid((0, 0), (1, 1), (2, 1))
    geometry_before = grid.node_coordinates.copy()
    grid.velocity_bc_mask[[0, 1, 2], 1] = True
    grid.velocity_bc_value[[0, 1, 2], 1] = 0.1
    for name in ("mass", "momentum", "velocity", "acceleration", "force_internal", "force_external"):
        getattr(grid, name).fill(5.0)
    grid.reset_nodal_state()
    for name in ("mass", "momentum", "velocity", "acceleration", "force_internal", "force_external"):
        assert np.all(getattr(grid, name) == 0)
    assert np.array_equal(grid.node_coordinates, geometry_before)
    assert np.all(grid.velocity_bc_mask[[0, 1, 2], 1])
    assert np.allclose(grid.velocity_bc_value[[0, 1, 2], 1], 0.1)


def test_input_arrays_independently_copied():
    origin = np.array([1.0, 2.0])
    spacing = np.array([0.5, 0.5])
    grid = BackgroundGrid(origin, spacing, (1, 1))
    origin[0] = 77
    spacing[0] = 99
    assert grid.origin[0] == 1.0
    assert grid.cell_size[0] == 0.5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"origin": [0.0], "cell_size": [1.0, 1.0], "n_cells": (2, 2)},
        {"origin": [0.0, np.nan], "cell_size": [1.0, 1.0], "n_cells": (2, 2)},
        {"origin": [0.0, 0.0], "cell_size": [0.0, 1.0], "n_cells": (2, 2)},
        {"origin": [0.0, 0.0], "cell_size": [np.inf, 1.0], "n_cells": (2, 2)},
        {"origin": [0.0, 0.0], "cell_size": [1.0, 1.0], "n_cells": (0, 2)},
        {"origin": [0.0, 0.0], "cell_size": [1.0, 1.0], "n_cells": (2, -1)},
        {"origin": [0.0, 0.0], "cell_size": [1.0, 1.0], "n_cells": (2.0, 2.0)},
        {"origin": [0.0, 0.0], "cell_size": [1.0, 1.0], "n_cells": (2, 2, 2)},
    ],
)
def test_invalid_grid_raises_value_error(kwargs):
    with pytest.raises(ValueError):
        BackgroundGrid(**kwargs)


def test_invalid_node_and_cell_indices():
    grid = BackgroundGrid((0, 0), (1, 1), (2, 2))
    with pytest.raises(IndexError):
        grid.node_id(3, 0)
    with pytest.raises(IndexError):
        grid.node_id(-1, 0)
    with pytest.raises(IndexError):
        grid.cell_node_ids(2, 1)
    with pytest.raises(IndexError):
        grid.cell_node_ids(0, -1)
    with pytest.raises(TypeError):
        grid.node_id(0.0, 1)
    with pytest.raises(TypeError):
        grid.cell_node_ids(True, 0)


def test_particle_grid_dims_match():
    particles = ParticleSet(position=[[0.25, 0.25]], mass=[1.0], volume=[1.0])
    grid = BackgroundGrid((0.0, 0.0), (0.5, 0.5), (2, 2))
    assert particles.ndim == grid.ndim
    assert np.all(particles.position[0] >= grid.origin)
