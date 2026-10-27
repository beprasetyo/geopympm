"""Stage 3A P2G: mass/momentum projection and API/physics checks."""

import numpy as np
import pytest

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.interpolation import evaluate_quad4
from geopympm.kernels.p2g import (
    p2g_mass_momentum_kernel, recover_nodal_velocity_kernel,
)
from geopympm.solvers import particle_to_grid


def grid2d():
    return BackgroundGrid(origin=(0., 0.), cell_size=(1., 1.), n_cells=(2, 2))


def particles_at(positions, masses, velocities):
    return ParticleSet(position=positions, mass=masses,
                       volume=np.ones(len(masses)), velocity=velocities)


def run_p2g(p, g):
    interp = evaluate_quad4(p, g)
    particle_to_grid(p, g, interp)
    return interp


def test_single_particle_at_cell_center_exact_mass_momentum_and_velocity():
    grid = grid2d()
    p = particles_at([[.5, .5]], [4.], [[3., -2.]])
    interp = run_p2g(p, grid)
    np.testing.assert_array_equal(interp.node_ids[0], [0, 1, 4, 3])
    for i in [0, 1, 4, 3]:
        assert grid.mass[i] == 1.
        np.testing.assert_array_equal(grid.momentum[i], [3., -2.])
        np.testing.assert_array_equal(grid.velocity[i], [3., -2.])
    for i in [2, 5, 6, 7, 8]:
        assert grid.mass[i] == 0.
        np.testing.assert_array_equal(grid.momentum[i], [0., 0.])
        np.testing.assert_array_equal(grid.velocity[i], [0., 0.])


def test_two_particles_same_cell_mass_weighted_nodal_velocities():
    grid = grid2d()
    p = particles_at([[.5, .5], [.5, .5]], [2., 6.], [[4., 0.], [0., 2.]])
    run_p2g(p, grid)
    for i in [0, 1, 3, 4]:
        assert grid.mass[i] == 2.
        np.testing.assert_array_equal(grid.momentum[i], [2., 3.])
        np.testing.assert_array_equal(grid.velocity[i], [1., 1.5])


def test_multiple_particles_share_grid_nodes_between_adjacent_cells():
    grid = BackgroundGrid((0, 0), (1, 1), (2, 1))
    p = particles_at([[.5, .5], [1.5, .5]], [2., 4.], [[3., 1.], [0., -2.]])
    run_p2g(p, grid)
    np.testing.assert_allclose(grid.mass, [.5, 1.5, 1., .5, 1.5, 1.])
    np.testing.assert_allclose(grid.momentum[1], [.5 * 3., .5 * 1. + 1. * -2.])
    np.testing.assert_allclose(grid.velocity[1], [1., -1.])
    np.testing.assert_allclose(grid.mass.sum(), 6.)
    np.testing.assert_allclose(grid.momentum.sum(axis=0), [6., -6.])


def test_conservation_mass_momentum_on_random_particles_and_nonsquare_grid():
    rng = np.random.default_rng(777)
    grid = BackgroundGrid(origin=(-3., 1.2), cell_size=(.25, .7), n_cells=(19, 13))
    xyz = grid.origin + rng.random((800, 2)) * (grid.cell_size * grid.n_cells)
    mass = 0.1 + 10. * rng.random(800)
    velocity = rng.standard_normal((800, 2)) * 20.
    p = particles_at(xyz, mass, velocity)
    run_p2g(p, grid)
    np.testing.assert_allclose(np.sum(grid.mass), np.sum(p.mass), rtol=3e-14)
    np.testing.assert_allclose(np.sum(grid.momentum, axis=0),
                               np.sum(p.mass[:, None] * p.velocity, axis=0),
                               rtol=1e-12, atol=1e-10)


def test_uniform_particle_velocity_is_reproduced_on_active_nodes():
    grid = grid2d()
    rng = np.random.default_rng(100)
    positions = rng.random((90, 2)) * 2.
    vel = np.tile([1.2, -3.4], (90, 1))
    p = particles_at(positions, np.ones(90), vel)
    run_p2g(p, grid)
    np.testing.assert_allclose(
        grid.velocity[grid.mass > 0],
        np.broadcast_to(vel[0], grid.velocity[grid.mass > 0].shape),
        rtol=0, atol=1e-14
    )


def test_boundary_particle_contributes_only_to_corner_node():
    grid = grid2d()
    p = particles_at([[2., 2.]], [5.], [[-1., 2.]])
    run_p2g(p, grid)
    assert grid.mass[8] == 5.
    np.testing.assert_array_equal(grid.momentum[8], [-5., 10.])
    assert np.count_nonzero(grid.mass) == 1


def test_grid_is_reset_every_call_without_affecting_boundaries():
    grid = grid2d()
    grid.velocity_bc_mask[0, 0] = True
    grid.velocity_bc_value[0, 0] = 2.5
    grid.force_internal[:] = 10.
    grid.force_external[:] = 20.
    grid.acceleration[:] = 30.
    p = particles_at([[.5, .5]], [4.], [[3., -2.]])
    interp = evaluate_quad4(p, grid)
    particle_to_grid(p, grid, interp)
    assert grid.mass.sum() == 4.
    np.testing.assert_array_equal(grid.force_internal, 0.)
    np.testing.assert_array_equal(grid.force_external, 0.)
    np.testing.assert_array_equal(grid.acceleration, 0.)
    particle_to_grid(p, grid, interp)
    assert grid.mass.sum() == 4.  # no double counting
    assert grid.velocity_bc_mask[0, 0]
    assert grid.velocity_bc_value[0, 0] == 2.5
    assert grid.velocity[0, 0] == 3.  # BC not applied in Stage 3A


def test_nonphysical_stress_volume_dont_affect_mass_momentum_p2g():
    grid = grid2d()
    p = particles_at([[.5, .5]], [4.], [[3., -2.]])
    p.stress[0, 0] = -1e8
    p.volume[0] = .001
    run_p2g(p, grid)
    assert grid.mass.sum() == 4.
    assert grid.momentum.sum(axis=0)[0] == 12.


def test_stale_interpolation_rejected_before_grid_reset():
    grid = grid2d()
    p = particles_at([[.5, .5]], [4.], [[3., -2.]])
    interp = evaluate_quad4(p, grid)
    grid.mass[0] = 1234.0
    p.position[0, 0] = 1.5
    with pytest.raises(ValueError, match="stale"):
        particle_to_grid(p, grid, interp)
    assert grid.mass[0] == 1234.0
    run_p2g(p, grid)
    assert grid.mass.sum() == 4.


@pytest.mark.parametrize("field, bad_value, message", [
    ("mass", 0.0, "Particle mass"),
    ("mass", -1.0, "Particle mass"),
    ("mass", np.nan, "Particle mass"),
    ("velocity", np.inf, "Particle velocity"),
])
def test_mutated_bad_particle_data_rejected_before_grid_reset(field, bad_value, message):
    grid = grid2d()
    p = particles_at([[.5, .5]], [4.], [[3., -2.]])
    interp = evaluate_quad4(p, grid)
    grid.mass[0] = 56.
    if field == "mass":
        p.mass[0] = bad_value
    else:
        p.velocity[0, 0] = bad_value
    with pytest.raises(ValueError, match=message):
        particle_to_grid(p, grid, interp)
    assert grid.mass[0] == 56.


def test_mutated_invalid_node_ids_rejected():
    grid = grid2d()
    p = particles_at([[.5, .5]], [4.], [[3., -2.]])
    interp = evaluate_quad4(p, grid)
    interp.node_ids[0, 0] = 100
    with pytest.raises(ValueError, match="node_ids"):
        particle_to_grid(p, grid, interp)


def test_mutated_invalid_weights_rejected():
    grid = grid2d()
    p = particles_at([[.5, .5]], [4.], [[3., -2.]])
    interp = evaluate_quad4(p, grid)
    interp.weights[0, 0] = -1.
    with pytest.raises(ValueError, match="weights"):
        particle_to_grid(p, grid, interp)


def test_invalid_dataclass_and_inputs_rejected():
    grid = grid2d()
    p = particles_at([[.5, .5]], [4.], [[3., -2.]])
    interp = evaluate_quad4(p, grid)
    with pytest.raises(TypeError, match="ParticleSet"):
        particle_to_grid(None, grid, interp)
    with pytest.raises(TypeError, match="BackgroundGrid"):
        particle_to_grid(p, None, interp)
    with pytest.raises(TypeError, match="Quad4Interpolation"):
        particle_to_grid(p, grid, None)


def test_three_dimensional_particle_not_allowed():
    grid = grid2d()
    p = ParticleSet(position=[[.5, .5, .5]], mass=[1.], volume=[1.])
    interp = evaluate_quad4(particles_at([[.5, .5]], [1.], [[0., 0.]]), grid)
    with pytest.raises(ValueError, match="requires 2D"):
        particle_to_grid(p, grid, interp)


def test_numba_p2g_kernels_compile_nopython():
    grid = grid2d()
    p = particles_at([[.5, .5]], [4.], [[3., -2.]])
    run_p2g(p, grid)
    assert p2g_mass_momentum_kernel.nopython_signatures
    assert recover_nodal_velocity_kernel.nopython_signatures


def test_explicit_reference_matches_numba_accumulation():
    grid = grid2d()
    p = particles_at([[.33, .24], [1.2, .8], [.15, 1.9], [1.75, 1.75]],
                     [.7, 1.4, 3.1, .6], [[1., 2.], [-3., 1.], [2., -1.], [2.5, 4.]])
    interp = run_p2g(p, grid)
    expected_mass = np.zeros(grid.n_nodes)
    expected_momentum = np.zeros((grid.n_nodes, 2))
    for part in range(p.n_particles):
        for a in range(4):
            inode = interp.node_ids[part, a]
            projected_mass = interp.weights[part, a] * p.mass[part]
            expected_mass[inode] += projected_mass
            expected_momentum[inode] += projected_mass * p.velocity[part]
    np.testing.assert_array_equal(grid.mass, expected_mass)
    np.testing.assert_array_equal(grid.momentum, expected_momentum)


def test_interpolation_from_different_grid_rejected_even_when_ids_in_range():
    grid = BackgroundGrid((0., 0.), (1., 1.), (3, 2))
    other = BackgroundGrid((0., 0.), (1., 1.), (2, 2))
    p = particles_at([[.5, .5]], [1.], [[1., 0.]])
    interp = evaluate_quad4(p, other)
    with pytest.raises(ValueError, match="connectivity"):
        particle_to_grid(p, grid, interp)
