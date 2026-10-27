"""Stage 3C: PIC/FLIP G2P transfer and no-partial-commit validation."""

import numpy as np
import pytest

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.interpolation import evaluate_quad4
from geopympm.kernels.g2p import g2p_quad4_kernel
from geopympm.solvers import particle_to_grid, update_grid_momentum, grid_to_particle


def setup_case(*, position=((0.5, 0.5),), velocity=((0., 0.),),
               mass=(4.,), volume=(1.,), n_cells=(1, 1),
               gravity=(0., 0.), dt=0.01, fixed_bottom=False):
    p = ParticleSet(position=position, velocity=velocity, mass=mass, volume=volume)
    g = BackgroundGrid((0., 0.), (1., 1.), n_cells)
    if fixed_bottom:
        g.velocity_bc_mask[[0, 1], 1] = True
    ip = evaluate_quad4(p, g)
    particle_to_grid(p, g, ip)
    update_grid_momentum(p, g, ip, dt, gravity=gravity)
    return p, g, ip


@pytest.mark.parametrize('alpha', [0., 0.25, 0.5, 1.])
def test_constant_velocity_exact_for_pic_flip_and_blends(alpha):
    p, g, ip = setup_case(velocity=((2., -3.),))
    grid_to_particle(p, g, ip, .01, pic_fraction=alpha)
    np.testing.assert_allclose(p.velocity, [[2., -3.]])
    np.testing.assert_allclose(p.position, [[.52, .47]])
    np.testing.assert_allclose(p.displacement, [[.02, -.03]])


@pytest.mark.parametrize('alpha', [0., 0.4, 1.])
def test_uniform_gravity_exact_velocity_and_position(alpha):
    p, g, ip = setup_case(velocity=((1., 2.),), dt=.1, gravity=(2., -4.))
    grid_to_particle(p, g, ip, .1, pic_fraction=alpha)
    np.testing.assert_allclose(p.velocity, [[1.2, 1.6]])
    np.testing.assert_allclose(p.position, [[.62, .66]])
    np.testing.assert_allclose(g.acceleration[g.mass > 0], [[2., -4.]] * 4)


def test_pic_flip_blending_different_with_opposite_particle_velocities():
    # Two particles at identical location, opposing velocities: grid PIC = 0,
    # grid acceleration = 0, so FLIP preserves each velocity.
    p0 = ((.5, .5), (.5, .5))
    v0 = ((2., 0.), (-2., 0.))
    for alpha, expected in ((1., 0.), (0., 2.), (.25, 1.5)):
        p, g, ip = setup_case(position=p0, velocity=v0,
                              mass=(1., 1.), volume=(.25, .25))
        grid_to_particle(p, g, ip, .1, pic_fraction=alpha)
        np.testing.assert_allclose(p.velocity[:, 0], [expected, -expected])
        np.testing.assert_allclose(p.position, p0)


def test_pic_and_flip_use_post_bc_grid_accelerations():
    p, g, ip = setup_case(velocity=((0., -1.),), dt=.1, fixed_bottom=True)
    np.testing.assert_allclose(g.velocity[[0, 1], 1], 0.)
    np.testing.assert_allclose(g.velocity[[2, 3], 1], -1.)
    np.testing.assert_allclose(g.acceleration[[0, 1], 1], 10.)
    grid_to_particle(p, g, ip, .1, pic_fraction=0.)
    np.testing.assert_allclose(p.velocity, [[0., -.5]])
    np.testing.assert_allclose(p.position, [[.5, .45]])


def test_first_gravity_step_with_bottom_fixed_nodes():
    p, g, ip = setup_case(gravity=(0., -9.81), dt=.01, fixed_bottom=True)
    grid_to_particle(p, g, ip, .01)
    np.testing.assert_allclose(p.velocity, [[0., -.04905]])
    np.testing.assert_allclose(p.position, [[.5, .4995095]])


def test_g2p_changes_only_particle_position_and_velocity():
    p, g, ip = setup_case(velocity=((1., 0.),))
    unchanged = {name: getattr(p, name).copy() for name in (
        'position0', 'mass', 'volume', 'volume0', 'stress', 'strain',
        'deformation_gradient', 'material_id', 'domain_size', 'domain_size0'
    )}
    position_array = p.position
    velocity_array = p.velocity
    grid_before = (g.mass.copy(), g.velocity.copy(), g.acceleration.copy())
    grid_to_particle(p, g, ip, .01)
    for name, state in unchanged.items():
        np.testing.assert_array_equal(getattr(p, name), state)
    assert p.position is position_array
    assert p.velocity is velocity_array
    for state, before in zip((g.mass, g.velocity, g.acceleration), grid_before):
        np.testing.assert_array_equal(state, before)


def test_recompute_interpolation_after_g2p_and_second_step():
    p, g, ip = setup_case(velocity=((.1, .0),), dt=.1)
    grid_to_particle(p, g, ip, .1)
    with pytest.raises(ValueError, match='stale'):
        grid_to_particle(p, g, ip, .1)
    ip2 = evaluate_quad4(p, g)
    particle_to_grid(p, g, ip2)
    update_grid_momentum(p, g, ip2, .1)
    grid_to_particle(p, g, ip2, .1)
    np.testing.assert_allclose(p.position, [[.52, .5]])


def test_particle_reaching_boundary_exactly_is_valid():
    p, g, ip = setup_case(position=((.9, .5),), velocity=((1., 0.),), dt=.1)
    grid_to_particle(p, g, ip, .1)
    np.testing.assert_allclose(p.position, [[1., .5]])
    evaluate_quad4(p, g)  # Stage 2 accepts points on upper/right boundary


def test_particle_outside_grid_rejected_without_partial_state_changes():
    p, g, ip = setup_case(position=((.95, .5),), velocity=((1., 0.),), dt=.2)
    old_x = p.position.copy()
    old_v = p.velocity.copy()
    with pytest.raises(ValueError, match='outside background grid'):
        grid_to_particle(p, g, ip, .2)
    np.testing.assert_array_equal(p.position, old_x)
    np.testing.assert_array_equal(p.velocity, old_v)


def test_if_any_particle_exits_grid_none_are_committed():
    p, g, ip = setup_case(position=((.5, .5), (.95, .5)),
                          velocity=((.5, 0.), (2., 0.)), mass=(2., 2.),
                          volume=(.3, .3), dt=.1)
    before_pos, before_vel = p.position.copy(), p.velocity.copy()
    with pytest.raises(ValueError, match='outside background grid'):
        grid_to_particle(p, g, ip, .1)
    np.testing.assert_array_equal(p.position, before_pos)
    np.testing.assert_array_equal(p.velocity, before_vel)


@pytest.mark.parametrize('bad_dt', [0., -1., float('nan'), float('inf'), 'text'])
def test_invalid_dt_rejected_without_state_change(bad_dt):
    p, g, ip = setup_case()
    before = p.position.copy()
    with pytest.raises(ValueError, match='dt'):
        grid_to_particle(p, g, ip, bad_dt)
    np.testing.assert_array_equal(p.position, before)


@pytest.mark.parametrize('invalid', [-0.1, 1.01, float('nan'), float('inf'), 'bad', True])
def test_invalid_pic_fraction_rejected(invalid):
    p, g, ip = setup_case()
    with pytest.raises(ValueError, match='pic_fraction'):
        grid_to_particle(p, g, ip, .01, pic_fraction=invalid)


def test_missing_p2g_state_rejected():
    p = ParticleSet(position=((.5, .5),), mass=(1.,), volume=(1.,))
    g = BackgroundGrid((0., 0.), (1., 1.), (1, 1))
    ip = evaluate_quad4(p, g)
    with pytest.raises(ValueError, match='particle_to_grid'):
        grid_to_particle(p, g, ip, .01)


def test_stale_interpolation_rejected_without_mutation():
    p, g, ip = setup_case()
    p.position[0, 0] += .1
    before_v = p.velocity.copy()
    with pytest.raises(ValueError, match='stale'):
        grid_to_particle(p, g, ip, .01)
    np.testing.assert_array_equal(p.velocity, before_v)


def test_invalid_weights_rejected():
    p, g, ip = setup_case()
    ip.weights[0] = [0.5, 0.5, 0., 0.]
    with pytest.raises(ValueError, match='weights'):
        grid_to_particle(p, g, ip, .01)


def test_invalid_node_ids_rejected():
    p, g, ip = setup_case()
    ip.node_ids[0, 0] = g.n_nodes
    with pytest.raises(ValueError, match='node_ids'):
        grid_to_particle(p, g, ip, .01)


def test_nan_grid_velocity_rejected():
    p, g, ip = setup_case()
    g.velocity[0, 0] = np.nan
    with pytest.raises(ValueError, match='velocity'):
        grid_to_particle(p, g, ip, .01)


def test_positive_weight_on_massless_node_rejected():
    p, g, ip = setup_case()
    g.mass[0] = 0.
    # Check mass sum happens before positive-weight check, but both prevent
    # silently interpolating invalid nodal states.
    with pytest.raises(ValueError, match='mass'):
        grid_to_particle(p, g, ip, .01)


def test_zero_weight_nodes_need_not_be_active():
    p, g, ip = setup_case(position=((0., 0.),), velocity=((.1, 0.),))
    assert (g.mass > 0).sum() == 1
    grid_to_particle(p, g, ip, .1)
    np.testing.assert_allclose(p.position, [[.01, 0.]])


def test_dimension_guard_for_3d_particles():
    p = ParticleSet(position=((.1, .1, .1),), mass=(1.,), volume=(1.,))
    g = BackgroundGrid((0., 0.), (1., 1.), (1, 1))
    ip = evaluate_quad4(
        ParticleSet(position=((.1, .1),), mass=(1.,), volume=(1.,)), g
    )
    with pytest.raises(ValueError, match='2D'):
        grid_to_particle(p, g, ip, .01)


def test_numba_g2p_kernel_compiles():
    p, g, ip = setup_case()
    grid_to_particle(p, g, ip, .01)
    assert g2p_quad4_kernel.nopython_signatures


def test_pic_and_flip_global_momentum_without_forces():
    rng = np.random.default_rng(1842)
    pos = rng.uniform(.1, .9, (30, 2))
    velocity = rng.uniform(-1, 1, (30, 2))
    mass = rng.uniform(.1, 1., 30)
    volume = np.full(30, .01)
    initial = (mass[:, None] * velocity).sum(axis=0)
    for alpha in (0., .25, 1.):
        p, g, ip = setup_case(position=pos, velocity=velocity, mass=mass,
                              volume=volume, dt=.001)
        grid_to_particle(p, g, ip, .001, pic_fraction=alpha)
        total = (p.mass[:, None] * p.velocity).sum(axis=0)
        np.testing.assert_allclose(total, initial, rtol=1e-12, atol=1e-12)
