"""Stage 3B: force assembly, gravity, velocity constraints and nodal update."""

import numpy as np
import pytest

from geopympm.core import ParticleSet, BackgroundGrid
from geopympm.interpolation import evaluate_quad4
from geopympm.kernels.grid_update import (
    assemble_internal_force_quad4_kernel, explicit_grid_momentum_kernel,
)
from geopympm.solvers import particle_to_grid, update_grid_momentum


def setup_case(*, position=((0.5, 0.5),), mass=(4.0,), volume=(1.0,),
               velocity=((0.0, 0.0),), stress=None, n_cells=(1, 1)):
    grid = BackgroundGrid(origin=(0., 0.), cell_size=(1., 1.), n_cells=n_cells)
    p = ParticleSet(position=position, mass=mass, volume=volume,
                    velocity=velocity, stress=stress)
    ip = evaluate_quad4(p, grid)
    particle_to_grid(p, grid, ip)
    return p, grid, ip


def test_zero_stress_zero_gravity_preserves_projected_velocity_and_momentum():
    p, g, ip = setup_case(velocity=((3., -1.),))
    old_p = g.momentum.copy()
    update_grid_momentum(p, g, ip, .01)
    np.testing.assert_array_equal(g.momentum, old_p)
    np.testing.assert_array_equal(g.force_internal, 0.)
    np.testing.assert_array_equal(g.force_external, 0.)
    np.testing.assert_array_equal(g.acceleration, 0.)
    for i in ip.node_ids[0]:
        np.testing.assert_allclose(g.velocity[i], [3., -1.])


def test_uniform_gravity_force_mass_times_gravity_and_velocity_step():
    p, g, ip = setup_case(velocity=((2., 3.),))
    update_grid_momentum(p, g, ip, .1, gravity=(1., -9.81))
    for i in ip.node_ids[0]:
        np.testing.assert_allclose(g.mass[i], 1.)
        np.testing.assert_allclose(g.force_external[i], [1., -9.81])
        np.testing.assert_allclose(g.acceleration[i], [1., -9.81])
        np.testing.assert_allclose(g.velocity[i], [2.1, 2.019])
    np.testing.assert_allclose(g.force_external.sum(axis=0), [4., -39.24])
    np.testing.assert_allclose(g.momentum.sum(axis=0), [8.4, 8.076])


def test_internal_force_from_pure_normal_stress_exact_and_zero_net():
    stress = np.array([[-100., -200., -50., 0., 0., 0.]])
    p, g, ip = setup_case(stress=stress)
    update_grid_momentum(p, g, ip, .01)
    expected = {
        0: [-50., -100.],
        1: [50., -100.],
        2: [-50., 100.],
        3: [50., 100.],
    }
    for node, force in expected.items():
        np.testing.assert_allclose(g.force_internal[node], force)
    np.testing.assert_allclose(g.force_internal.sum(axis=0), [0., 0.], atol=1e-13)
    np.testing.assert_allclose(g.force_external, 0.)
    # Force has now advanced the momentum and velocity, not the particles.
    np.testing.assert_allclose(g.momentum[0], [-0.5, -1.])
    np.testing.assert_array_equal(p.velocity, [[0., 0.]])


def test_shear_and_sigma_zz_plane_strain_behavior():
    p, g, ip = setup_case(stress=[[0., 0., 9e9, 12., 0., 0.]])
    update_grid_momentum(p, g, ip, .01)
    np.testing.assert_allclose(g.force_internal[0], [6., 6.])
    np.testing.assert_allclose(g.force_internal[1], [6., -6.])
    np.testing.assert_allclose(g.force_internal[2], [-6., 6.])
    np.testing.assert_allclose(g.force_internal[3], [-6., -6.])


def test_componentwise_fixed_and_prescribed_velocity_and_acceleration():
    p, g, ip = setup_case(velocity=((2., 3.),))
    g.velocity_bc_mask[0, 0] = True
    g.velocity_bc_value[0, 0] = 0.
    g.velocity_bc_mask[1, 1] = True
    g.velocity_bc_value[1, 1] = -4.
    update_grid_momentum(p, g, ip, .2, gravity=(1., -10.))
    np.testing.assert_allclose(g.velocity[0], [0., 1.])
    np.testing.assert_allclose(g.velocity[1], [2.2, -4.])
    np.testing.assert_allclose(g.acceleration[0], [-10., -10.])
    np.testing.assert_allclose(g.acceleration[1], [1., -35.])
    np.testing.assert_allclose(g.momentum[0], [0., 1.])
    np.testing.assert_allclose(g.momentum[1], [2.2, -4.])
    assert g.velocity_bc_mask[0, 0]
    assert g.velocity_bc_value[1, 1] == -4.


def test_external_nodal_force_combines_with_gravity():
    p, g, ip = setup_case()
    force = np.zeros((g.n_nodes, 2))
    force[0] = [4., -2.]
    update_grid_momentum(p, g, ip, .5, gravity=(0., -2.),
                         nodal_external_force=force)
    np.testing.assert_allclose(g.force_external[0], [4., -4.])
    np.testing.assert_allclose(g.force_external[1], [0., -2.])
    np.testing.assert_allclose(g.velocity[0], [2., -2.])
    np.testing.assert_allclose(g.acceleration[0], [4., -4.])
    np.testing.assert_allclose(force[0], [4., -2.])  # caller's data not mutated


def test_many_particles_internal_force_has_zero_global_resultant():
    rng = np.random.default_rng(456)
    x = rng.uniform(.1, 3.9, (150, 2))
    masses = rng.uniform(.1, 4., 150)
    volumes = rng.uniform(.001, .09, 150)
    stresses = rng.normal(0., 50., (150, 6))
    velocities = rng.normal(0., 1., (150, 2))
    p, g, ip = setup_case(position=x, mass=masses, volume=volumes,
                          stress=stresses, velocity=velocities,
                          n_cells=(4, 4))
    old_sum = g.momentum.sum(axis=0).copy()
    dt = .0005
    update_grid_momentum(p, g, ip, dt)
    np.testing.assert_allclose(g.force_internal.sum(axis=0), [0., 0.], atol=1e-11)
    np.testing.assert_allclose(g.momentum.sum(axis=0), old_sum, atol=1e-11)


def test_inactive_nodes_remain_zero_and_bc_persists():
    p, g, ip = setup_case(n_cells=(3, 3))
    inactive = np.where(g.mass == 0)[0]
    assert len(inactive) > 0
    g.velocity_bc_mask[inactive[0], 0] = True
    g.velocity_bc_value[inactive[0], 0] = 3.
    update_grid_momentum(p, g, ip, .001)
    np.testing.assert_array_equal(g.velocity[inactive], 0.)
    np.testing.assert_array_equal(g.acceleration[inactive], 0.)
    np.testing.assert_array_equal(g.momentum[inactive], 0.)
    assert g.velocity_bc_value[inactive[0], 0] == 3.


def test_repeated_new_p2g_step_reassembles_forces_not_accumulates():
    p, g, ip = setup_case(stress=[[-10., -10., 0., 0., 0., 0.]])
    update_grid_momentum(p, g, ip, .01, gravity=(0., -1.))
    force_first = g.force_internal.copy()
    ext_first = g.force_external.copy()
    # An actual next step would perform G2P before this P2G. The purpose
    # here is to verify the grid's force reset + repeatable assembly.
    particle_to_grid(p, g, ip)
    np.testing.assert_array_equal(g.force_internal, 0.)
    update_grid_momentum(p, g, ip, .01, gravity=(0., -1.))
    np.testing.assert_array_equal(g.force_internal, force_first)
    np.testing.assert_array_equal(g.force_external, ext_first)


@pytest.mark.parametrize('bad_dt', [0., -1., float('nan'), float('inf'), 'oops'])
def test_bad_timestep_fails_before_mutating_grid(bad_dt):
    p, g, ip = setup_case()
    old = g.momentum.copy()
    with pytest.raises(ValueError, match='dt'):
        update_grid_momentum(p, g, ip, bad_dt)
    np.testing.assert_array_equal(g.momentum, old)


@pytest.mark.parametrize('gravity', [[0., 0., 1.], [0., float('nan')], [float('inf'), 0.]])
def test_bad_gravity_rejected(gravity):
    p, g, ip = setup_case()
    with pytest.raises(ValueError, match='gravity'):
        update_grid_momentum(p, g, ip, .1, gravity=gravity)


def test_uninitialized_p2g_rejected():
    g = BackgroundGrid((0, 0), (1, 1), (1, 1))
    p = ParticleSet(position=[[.5, .5]], mass=[1.], volume=[1.])
    ip = evaluate_quad4(p, g)
    with pytest.raises(ValueError, match='particle_to_grid'):
        update_grid_momentum(p, g, ip, .1)


def test_invalid_particle_stress_or_volume_rejected_without_mutation():
    p, g, ip = setup_case()
    g.force_internal[:] = 101.
    g.force_external[:] = 102.
    g.acceleration[:] = 103.
    p.stress[0, 0] = float('nan')
    with pytest.raises(ValueError, match='stresses'):
        update_grid_momentum(p, g, ip, .1)
    np.testing.assert_array_equal(g.force_internal, 101.)
    p.stress[0, 0] = 0.
    p.volume[0] = -1.
    with pytest.raises(ValueError, match='volumes'):
        update_grid_momentum(p, g, ip, .1)
    np.testing.assert_array_equal(g.force_external, 102.)
    np.testing.assert_array_equal(g.acceleration, 103.)


def test_mutated_grid_mass_rejected():
    p, g, ip = setup_case()
    g.mass[0] = 0.
    with pytest.raises(ValueError, match='mass is inconsistent'):
        update_grid_momentum(p, g, ip, .1)


def test_stale_interpolation_rejected():
    p, g, ip = setup_case()
    p.position[0, 0] += .05
    with pytest.raises(ValueError, match='stale'):
        update_grid_momentum(p, g, ip, .1)


def test_mutated_gradients_rejected():
    p, g, ip = setup_case()
    ip.gradients[0, 0] = [-10., 5.]
    with pytest.raises(ValueError, match='gradients'):
        update_grid_momentum(p, g, ip, .1)


def test_nodal_load_on_massless_node_rejected():
    p, g, ip = setup_case(n_cells=(3, 3))
    nodal = np.zeros((g.n_nodes, 2))
    nodal[-1] = [1., 0.]
    with pytest.raises(ValueError, match='massless'):
        update_grid_momentum(p, g, ip, .1, nodal_external_force=nodal)


def test_invalid_nodal_load_shape_rejected():
    p, g, ip = setup_case()
    with pytest.raises(ValueError, match='nodal_external_force'):
        update_grid_momentum(p, g, ip, .1, nodal_external_force=np.ones((3, 2)))


def test_p2g_particles_not_mutated_by_grid_update():
    p, g, ip = setup_case(velocity=((3., 2.),),
                          stress=[[-25., -10., -4., 2., 0., 0.]])
    snapshots = {name: getattr(p, name).copy() for name in
                 ('position', 'velocity', 'stress', 'strain', 'volume',
                  'deformation_gradient')}
    update_grid_momentum(p, g, ip, .01, gravity=(0., -9.81))
    for key, old in snapshots.items():
        np.testing.assert_array_equal(getattr(p, key), old)


def test_numba_compiles_both_new_kernels():
    p, g, ip = setup_case()
    update_grid_momentum(p, g, ip, .1)
    assert assemble_internal_force_quad4_kernel.nopython_signatures
    assert explicit_grid_momentum_kernel.nopython_signatures
