"""Numerical and API checks for Stage 2 conventional-MPM Quad4 interpolation."""

import numpy as np
import pytest

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.interpolation import Quad4Interpolation, evaluate_quad4
from geopympm.kernels.interpolation_quad4 import quad4_kernel


def particles_at(points, ndim=2):
    coords = np.asarray(points, dtype=np.float64)
    assert coords.shape[1] == ndim
    n = coords.shape[0]
    return ParticleSet(position=coords, mass=np.ones(n), volume=np.ones(n))


def unit_grid():
    return BackgroundGrid((0.0, 0.0), (1.0, 1.0), (2, 2))


def test_center_of_first_cell_exact_weights_and_gradients():
    grid = unit_grid()
    result = evaluate_quad4(particles_at([[0.5, 0.5]]), grid)
    assert isinstance(result, Quad4Interpolation)
    np.testing.assert_array_equal(result.cell_indices[0], [0, 0])
    np.testing.assert_array_equal(result.local_coordinates[0], [0.5, 0.5])
    np.testing.assert_array_equal(result.node_ids[0], [0, 1, 4, 3])
    np.testing.assert_allclose(result.weights[0], [0.25] * 4)
    np.testing.assert_allclose(result.gradients[0], [
        [-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]
    ])


def test_non_square_cell_with_nonzero_origin():
    grid = BackgroundGrid(origin=(10, -2), cell_size=(2, 4), n_cells=(3, 2))
    result = evaluate_quad4(particles_at([[10.5, -1.0]]), grid)
    np.testing.assert_array_equal(result.cell_indices[0], [0, 0])
    np.testing.assert_allclose(result.local_coordinates[0], [0.25, 0.25])
    np.testing.assert_allclose(result.weights[0], [0.5625, 0.1875, 0.0625, 0.1875])
    np.testing.assert_allclose(result.gradients[0], [
        [-0.375, -0.1875],
        [0.375, -0.0625],
        [0.125, 0.0625],
        [-0.125, 0.1875],
    ])


def test_second_row_and_x_fast_numbering():
    grid = BackgroundGrid((0, 0), (1, 1), (3, 2))
    result = evaluate_quad4(particles_at([[1.4, 1.6]]), grid)
    np.testing.assert_array_equal(result.cell_indices[0], [1, 1])
    np.testing.assert_array_equal(result.node_ids[0], grid.cell_node_ids(1, 1))
    np.testing.assert_array_equal(result.node_ids[0], [5, 6, 10, 9])


@pytest.mark.parametrize("position,cell,weights", [
    ([0.0, 0.0], [0, 0], [1, 0, 0, 0]),
    ([1.0, 0.0], [1, 0], [1, 0, 0, 0]),
    ([0.0, 1.0], [0, 1], [1, 0, 0, 0]),
    ([2.0, 0.0], [1, 0], [0, 1, 0, 0]),
    ([0.0, 2.0], [0, 1], [0, 0, 0, 1]),
    ([2.0, 2.0], [1, 1], [0, 0, 1, 0]),
    ([1.0, 1.0], [1, 1], [1, 0, 0, 0]),
])
def test_points_on_grid_lines_and_domain_edges(position, cell, weights):
    result = evaluate_quad4(particles_at([position]), unit_grid())
    np.testing.assert_array_equal(result.cell_indices[0], cell)
    np.testing.assert_allclose(result.weights[0], weights)


def test_partition_of_unity_zero_gradient_sum_and_linear_field_reproduction():
    rng = np.random.default_rng(20261023)
    grid = BackgroundGrid(origin=(5.0, -3.0), cell_size=(0.7, 1.4), n_cells=(9, 6))
    positions = grid.origin + rng.random((300, 2)) * (grid.cell_size * grid.n_cells)
    result = evaluate_quad4(particles_at(positions), grid)

    np.testing.assert_allclose(result.weights.sum(axis=1), 1, atol=3e-15)
    np.testing.assert_allclose(result.gradients.sum(axis=1), 0, atol=3e-15)
    assert np.all(result.weights >= 0.0)
    assert np.all(result.weights <= 1.0)

    node_xy = grid.node_coordinates[result.node_ids]
    reconstructed = np.sum(result.weights[:, :, None] * node_xy, axis=1)
    np.testing.assert_allclose(reconstructed, positions, atol=2e-14, rtol=0)

    # The derivative of the linearly reproduced coordinate field is identity.
    jacobians = np.einsum('pni,pnj->pij', node_xy, result.gradients)
    np.testing.assert_allclose(jacobians, np.broadcast_to(np.eye(2), jacobians.shape), atol=2e-14, rtol=0)


def test_exact_linear_velocity_gradient():
    grid = BackgroundGrid((0, 0), (0.5, 2.0), (3, 2))
    particles = particles_at([[0.13, 0.45], [0.8, 2.7], [1.4, 3.9]])
    result = evaluate_quad4(particles, grid)
    # v(x) = A x + b. Then L = sum_i v_i ⊗ grad N_i == A.
    A = np.array([[2.5, -0.4], [1.1, -3.2]])
    b = np.array([-0.2, 10.0])
    v_i = grid.node_coordinates @ A.T + b
    v_support = v_i[result.node_ids]
    L = np.einsum('pni,pnj->pij', v_support, result.gradients)
    np.testing.assert_allclose(L, np.broadcast_to(A, L.shape), atol=1e-14, rtol=0)


def test_valid_c_contiguous_typed_arrays():
    grid = unit_grid()
    result = evaluate_quad4(particles_at([[0.5, 0.5], [1.5, 0.5]]), grid)
    for name, shape, dtype in (
        ("cell_indices", (2, 2), np.int64),
        ("local_coordinates", (2, 2), np.float64),
        ("node_ids", (2, 4), np.int64),
        ("weights", (2, 4), np.float64),
        ("gradients", (2, 4, 2), np.float64),
    ):
        value = getattr(result, name)
        assert value.shape == shape
        assert value.dtype == dtype
        assert value.flags.c_contiguous


@pytest.mark.parametrize("position", [
    [-1e-4, 0.2], [2.0001, 0.3], [0.2, -0.01], [0.2, 2.01],
])
def test_out_of_grid_particle_rejected(position):
    with pytest.raises(ValueError, match="Particle 0.*outside"):
        evaluate_quad4(particles_at([position]), unit_grid())


def test_error_identifies_first_outside_particle():
    positions = [[0.2, 0.2], [1.2, 0.2], [-0.01, 0.2], [4.0, 4.0]]
    with pytest.raises(ValueError, match="Particle 2"):
        evaluate_quad4(particles_at(positions), unit_grid())


def test_roundoff_tolerance_on_outer_boundary_but_not_extrapolation():
    particles = particles_at([[2.0 + 2e-15, 1.0]])
    result = evaluate_quad4(particles, unit_grid())
    np.testing.assert_allclose(result.weights[0], [0, 1, 0, 0], atol=1e-14)
    # Input particle position is not mutated by snapping to the grid boundary.
    assert particles.position[0, 0] > 2.0
    particles.position[0, 0] = 2.0 + 1e-5
    with pytest.raises(ValueError, match="outside"):
        evaluate_quad4(particles, unit_grid())


def test_particle_position_update_requires_new_evaluation():
    particles = particles_at([[0.25, 0.5]])
    first = evaluate_quad4(particles, unit_grid())
    particles.position[0, 0] = 1.25
    second = evaluate_quad4(particles, unit_grid())
    np.testing.assert_array_equal(first.node_ids[0], [0, 1, 4, 3])
    np.testing.assert_array_equal(second.node_ids[0], [1, 2, 5, 4])
    np.testing.assert_allclose(first.weights[0], second.weights[0])


def test_3d_particle_set_not_supported():
    particles = particles_at([[0.25, 0.25, 0.25]], ndim=3)
    with pytest.raises(ValueError, match="requires 2D"):
        evaluate_quad4(particles, unit_grid())


def test_non_finite_coordinates_after_particle_mutation_rejected():
    particles = particles_at([[0.1, 0.3]])
    particles.position[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_quad4(particles, unit_grid())


def test_invalid_types_rejected():
    with pytest.raises(TypeError, match="ParticleSet"):
        evaluate_quad4(np.array([[0.5, 0.5]]), unit_grid())
    with pytest.raises(TypeError, match="BackgroundGrid"):
        evaluate_quad4(particles_at([[0.5, 0.5]]), None)


def test_gimp_domain_is_deliberately_unused():
    grid = unit_grid()
    positions = [[0.4, 0.5]]
    no_domain = particles_at(positions)
    with_domain = ParticleSet(
        position=positions, mass=[1], volume=[1], domain_size=[[0.8, 0.8]]
    )
    a = evaluate_quad4(no_domain, grid)
    b = evaluate_quad4(with_domain, grid)
    np.testing.assert_array_equal(a.node_ids, b.node_ids)
    np.testing.assert_array_equal(a.weights, b.weights)
    np.testing.assert_array_equal(a.gradients, b.gradients)


def test_numba_kernel_compiles_in_nopython_mode():
    evaluate_quad4(particles_at([[0.5, 0.5]]), unit_grid())
    assert quad4_kernel.nopython_signatures
