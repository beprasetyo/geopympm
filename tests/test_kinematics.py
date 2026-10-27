"""Stage 3D kinematics tests: analytic affine velocity gradients and safety."""

import numpy as np
import pytest

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.interpolation import evaluate_quad4
from geopympm.kernels.kinematics import particle_kinematics_quad4_kernel
from geopympm.solvers import (
    particle_to_grid, update_grid_momentum, grid_to_particle,
    update_particle_kinematics, KinematicsResult,
)


def case(position=((0.5, 0.5),), volumes=(0.25,), masses=(450.0,)):
    p = ParticleSet(position=position, mass=masses, volume=volumes)
    g = BackgroundGrid(origin=(0.0, 0.0), cell_size=(1.0, 1.0), n_cells=(1, 1))
    ip = evaluate_quad4(p, g)
    particle_to_grid(p, g, ip)
    return p, g, ip


def affine_velocity(grid, A, offset=(0.0, 0.0)):
    """v(x) = A @ x + offset, exactly reproducible by Quad4 shape functions."""
    grid.velocity[:] = grid.node_coordinates @ np.asarray(A).T + np.asarray(offset)


def test_zero_field_initial_F_and_volume_unchanged():
    p, g, ip = case()
    old_F = p.deformation_gradient.copy()
    old_volume = p.volume.copy()
    result = update_particle_kinematics(p, g, ip, dt=0.1)
    assert isinstance(result, KinematicsResult)
    np.testing.assert_array_equal(result.velocity_gradient, np.zeros((1, 2, 2)))
    np.testing.assert_array_equal(result.strain_increment, np.zeros((1, 6)))
    np.testing.assert_array_equal(p.deformation_gradient, old_F)
    np.testing.assert_array_equal(p.volume, old_volume)
    np.testing.assert_array_equal(p.strain, np.zeros((1, 6)))
    np.testing.assert_array_equal(result.jacobian, [1.0])


def test_rigid_translation_zero_velocity_gradient():
    p, g, ip = case()
    affine_velocity(g, [[0, 0], [0, 0]], offset=(2, -3))
    result = update_particle_kinematics(p, g, ip, dt=0.15)
    np.testing.assert_allclose(result.velocity_gradient, 0.0, atol=1e-14)
    np.testing.assert_allclose(p.volume, [0.25])
    np.testing.assert_allclose(p.deformation_gradient, [np.eye(2)])
    np.testing.assert_allclose(p.strain, 0.0)


def test_uniaxial_vertical_compression():
    p, g, ip = case(volumes=(0.5,))
    affine_velocity(g, [[0, 0], [0, -0.2]])
    result = update_particle_kinematics(p, g, ip, dt=0.1)
    np.testing.assert_allclose(result.velocity_gradient, [[[0, 0], [0, -0.2]]], atol=1e-14)
    np.testing.assert_allclose(p.deformation_gradient, [[[1, 0], [0, .98]]], atol=1e-14)
    np.testing.assert_allclose(result.jacobian, [.98], atol=1e-14)
    np.testing.assert_allclose(p.volume, [.49], atol=1e-14)
    np.testing.assert_allclose(p.density, [450.0 / .49], rtol=1e-14)
    np.testing.assert_allclose(result.strain_increment, [[0, -.02, 0, 0, 0, 0]], atol=1e-14)
    np.testing.assert_allclose(p.strain, [[0, -.02, 0, 0, 0, 0]], atol=1e-14)


def test_biaxial_extension_and_incremental_volume_is_product_of_stretches():
    p, g, ip = case()
    affine_velocity(g, [[1, 0], [0, 2]])
    result = update_particle_kinematics(p, g, ip, dt=0.1)
    np.testing.assert_allclose(p.deformation_gradient, [[[1.1, 0], [0, 1.2]]])
    np.testing.assert_allclose(result.jacobian, [1.32])
    np.testing.assert_allclose(p.volume, [.33])
    np.testing.assert_allclose(result.strain_increment, [[.1, .2, 0, 0, 0, 0]])


def test_simple_shear_increment_uses_engineering_gamma():
    p, g, ip = case()
    affine_velocity(g, [[0, 2], [0, 0]])
    result = update_particle_kinematics(p, g, ip, dt=0.1)
    np.testing.assert_allclose(result.velocity_gradient, [[[0, 2], [0, 0]]])
    np.testing.assert_allclose(p.deformation_gradient, [[[1, .2], [0, 1]]])
    np.testing.assert_allclose(p.volume, [.25])
    np.testing.assert_allclose(result.strain_increment, [[0, 0, 0, .2, 0, 0]])
    np.testing.assert_allclose(p.strain, [[0, 0, 0, .2, 0, 0]])


def test_two_particles_obtain_same_gradient_for_affine_nodal_velocity():
    p, g, ip = case(position=((.2, .3), (.8, .9)), volumes=(.25, .5), masses=(2, 3))
    affine_velocity(g, [[.1, .4], [-.2, -.3]], offset=(3, 1))
    result = update_particle_kinematics(p, g, ip, dt=.2)
    np.testing.assert_allclose(result.velocity_gradient, [ [[.1, .4], [-.2, -.3]]] * 2, atol=1e-14)
    np.testing.assert_allclose(result.strain_increment, [[.02, -.06, 0, .04, 0, 0]] * 2, atol=1e-14)
    np.testing.assert_allclose(p.mass, [2, 3])


def test_previous_F_left_multiplied_and_existing_strain_accumulated():
    p, g, ip = case()
    p.deformation_gradient[0] = [[1.3, .2], [0, .8]]
    p.volume[:] = 0.25 * 1.04
    p.strain[0] = [0.02, -.03, 0, .04, 0, 0]
    affine_velocity(g, [[0, 0], [0, -.2]])
    result = update_particle_kinematics(p, g, ip, dt=.1)
    np.testing.assert_allclose(p.deformation_gradient, [[[1.3, .2], [0, .784]]])
    np.testing.assert_allclose(p.volume, [.25 * 1.3 * .784])
    np.testing.assert_allclose(p.strain, [[.02, -.05, 0, .04, 0, 0]])
    assert result.strain_increment.shape == (1, 6)


def test_rigid_rotation_rate_has_zero_symmetric_strain_but_euler_J_drift():
    p, g, ip = case()
    affine_velocity(g, [[0, -1], [1, 0]])
    result = update_particle_kinematics(p, g, ip, dt=.01)
    np.testing.assert_allclose(result.strain_increment, 0.0, atol=1e-14)
    # First-order Euler F update is not exactly volume preserving for rotation:
    np.testing.assert_allclose(result.jacobian, [1.0001], atol=1e-14)


def test_no_partial_commit_if_negative_jacobian():
    p, g, ip = case(position=((.25, .5), (.75, .5)), volumes=(.25, .25), masses=(1, 1))
    # Vertical large compression: Fyy = 1 + dt * (-2) <= 0 for dt=0.6.
    affine_velocity(g, [[0, 0], [0, -2]])
    original = [p.deformation_gradient.copy(), p.volume.copy(), p.strain.copy()]
    with pytest.raises(ValueError, match="nonpositive deformation Jacobian"):
        update_particle_kinematics(p, g, ip, dt=.6)
    for actual, old in zip((p.deformation_gradient, p.volume, p.strain), original):
        np.testing.assert_array_equal(actual, old)


def test_no_partial_commit_if_second_particle_has_invalid_F():
    p, g, ip = case(position=((.25, .5), (.75, .5)), volumes=(.25, .25), masses=(1, 1))
    # Fake an invalid pre-state for second particle.
    p.deformation_gradient[1] = [[1, 0], [0, -1]]
    originals = [p.deformation_gradient.copy(), p.volume.copy(), p.strain.copy()]
    with pytest.raises(ValueError, match="positive Jacobians"):
        update_particle_kinematics(p, g, ip, dt=.01)
    for actual, old in zip((p.deformation_gradient, p.volume, p.strain), originals):
        np.testing.assert_array_equal(actual, old)


def test_detect_stale_interpolation_after_G2P():
    p, g, ip = case()
    g.velocity[:] = [0.2, 0.0]
    g.acceleration[:] = [0.0, 0.0]
    grid_to_particle(p, g, ip, dt=.01)
    with pytest.raises(ValueError, match="stale"):
        update_particle_kinematics(p, g, ip, dt=.01)


def test_bad_gradient_rejected_without_commit():
    p, g, ip = case()
    ip.gradients[0, 0, 0] += .2
    state = p.deformation_gradient.copy()
    with pytest.raises(ValueError, match="inconsistent"):
        update_particle_kinematics(p, g, ip, dt=.01)
    np.testing.assert_array_equal(p.deformation_gradient, state)


def test_grid_mass_missing():
    p, g, ip = case()
    g.reset_nodal_state()
    with pytest.raises(ValueError, match="Grid mass"):
        update_particle_kinematics(p, g, ip, dt=.01)


@pytest.mark.parametrize('dt', [0.0, -1.0, float('nan'), float('inf'), True])
def test_bad_dt(dt):
    p, g, ip = case()
    with pytest.raises(ValueError, match="dt"):
        update_particle_kinematics(p, g, ip, dt=dt)


def test_combined_p2g_grid_update_kinematics_then_g2p_sequence():
    p = ParticleSet(position=[[.5, .5]], mass=[4.0], volume=[.25])
    g = BackgroundGrid((0, 0), (1, 1), (1, 1))
    g.velocity_bc_mask[[0, 1], 1] = True
    ip = evaluate_quad4(p, g)
    particle_to_grid(p, g, ip)
    update_grid_momentum(p, g, ip, dt=.01, gravity=(0, -9.81))
    result = update_particle_kinematics(p, g, ip, dt=.01)
    np.testing.assert_allclose(result.velocity_gradient, [[[0, 0], [0, -.0981]]])
    np.testing.assert_allclose(p.deformation_gradient, [[[1, 0], [0, .999019]]])
    np.testing.assert_allclose(p.volume, [.25 * .999019])
    np.testing.assert_allclose(p.strain, [[0, -.000981, 0, 0, 0, 0]])
    grid_to_particle(p, g, ip, dt=.01, pic_fraction=1.0)
    np.testing.assert_allclose(p.position, [[.5, .4995095]])
    np.testing.assert_allclose(p.velocity, [[0, -.04905]])


def test_kernel_compiled_numba():
    p, g, ip = case()
    update_particle_kinematics(p, g, ip, dt=.01)
    assert particle_kinematics_quad4_kernel.signatures
