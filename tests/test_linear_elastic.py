"""Stage 4: incremental isotropic elasticity, 2D plane strain, Numba."""
import numpy as np
import pytest

from geopympm.core import ParticleSet, BackgroundGrid
from geopympm.interpolation import evaluate_quad4
from geopympm.solvers import particle_to_grid, update_grid_momentum, update_particle_kinematics, grid_to_particle
from geopympm.materials import LinearElastic, update_linear_elastic_stress
from geopympm.kernels.linear_elastic import linear_elastic_plane_strain_kernel


def particles(n=1, material_id=None, stress=None):
    positions = [[(i + 1) / (n + 1), .5] for i in range(n)]
    return ParticleSet(
        position=positions, mass=np.full(n, 1.0), volume=np.full(n, 0.25),
        material_id=material_id, stress=stress,
    )


def inc(p, exx=0.0, eyy=0.0, gxy=0.0):
    output = np.zeros((p.n_particles, 6), dtype=np.float64)
    output[:, 0] = exx
    output[:, 1] = eyy
    output[:, 3] = gxy
    return output


def test_material_moduli_and_readonly_parameters():
    m = LinearElastic(20e6, .25)
    assert m.shear_modulus == pytest.approx(8e6)
    assert m.lame_lambda == pytest.approx(8e6)
    assert m.bulk_modulus == pytest.approx(20e6 / 1.5)
    with pytest.raises(AttributeError):
        m.poisson_ratio = .3


@pytest.mark.parametrize('E, nu', [(0, .3), (-1e6, .3), (10., -.99999999999999999),
                                    (100, .5), (100, -.999), (100, 1),
                                    (float('nan'), .3), (1.0, float('inf')),
                                    (True, .2), (100, False)])
def test_bad_elastic_parameter_values(E, nu):
    # Extremely close to -1 rounds to -1.0 in float64.
    if nu == -.999 and E == 100:
        assert LinearElastic(E, nu).shear_modulus > 0
    else:
        with pytest.raises(ValueError):
            LinearElastic(E, nu)


def test_plane_strain_exx_has_nonzero_out_of_plane_stress():
    p = particles()
    model = LinearElastic(20e6, .25)
    update_linear_elastic_stress(p, inc(p, exx=0.001), model)
    # lambda = G = 8 MPa, sigma_xx = (lambda + 2G)exx
    np.testing.assert_allclose(p.stress[0], [24000, 8000, 8000, 0, 0, 0])


def test_plane_strain_uniaxial_vertical_compression_tension_positive():
    p = particles()
    model = LinearElastic(20e6, .25)
    update_linear_elastic_stress(p, inc(p, eyy=-.002), model)
    np.testing.assert_allclose(p.stress[0], [-16000, -48000, -16000, 0, 0, 0])


def test_pure_shear_uses_engineering_shear_gamma():
    p = particles()
    m = LinearElastic(20e6, .25)  # G = 8e6
    update_linear_elastic_stress(p, inc(p, gxy=.003), m)
    np.testing.assert_allclose(p.stress[0], [0, 0, 0, 24000, 0, 0])


def test_two_increments_accumulate_and_strain_is_not_modified():
    p = particles(stress=[[110.0, 220.0, 330.0, 440.0, 0.0, 0.0]])
    m = LinearElastic(30e6, .2)
    one = inc(p, exx=.0001, eyy=-.0001, gxy=.0002)
    old_strain = p.strain.copy()
    update_linear_elastic_stress(p, one, m)
    first = p.stress.copy()
    update_linear_elastic_stress(p, one, m)
    np.testing.assert_allclose(p.stress, 2*first - [[110,220,330,440,0,0]])
    np.testing.assert_array_equal(p.strain, old_strain)
    np.testing.assert_array_equal(p.volume, [.25])


def test_two_material_ids_different_stiffness_while_same_increment():
    p = particles(n=2, material_id=[3, 11])
    models = {3: LinearElastic(10e6, .25), 11: LinearElastic(40e6, .25)}
    update_linear_elastic_stress(p, inc(p, exx=.001), models)
    np.testing.assert_allclose(p.stress[:, 0], [12000, 48000])
    np.testing.assert_allclose(p.stress[:, 2], [4000, 16000])


def test_single_material_works_with_arbitrary_ids():
    p = particles(n=2, material_id=[8, 45])
    update_linear_elastic_stress(p, inc(p, exx=.001), LinearElastic(10e6, 0.25))
    np.testing.assert_allclose(p.stress[:, 0], [12000, 12000])


def test_missing_material_is_atomic():
    p = particles(n=2, material_id=[0, 5])
    before = p.stress.copy()
    with pytest.raises(ValueError, match='material_id=5'):
        update_linear_elastic_stress(p, inc(p, exx=.01), {0: LinearElastic(10e6, .3)})
    np.testing.assert_array_equal(p.stress, before)


def test_invalid_material_mapping_is_atomic():
    p = particles()
    before = p.stress.copy()
    with pytest.raises(TypeError, match='LinearElastic instances'):
        update_linear_elastic_stress(p, inc(p, exx=.01), {0: 1000.0})
    with pytest.raises(ValueError, match='int32'):
        update_linear_elastic_stress(p, inc(p, exx=.01), {True: LinearElastic(1e6, .3)})
    np.testing.assert_array_equal(p.stress, before)


@pytest.mark.parametrize('bad', [
    np.zeros((1, 3)),
    np.array([[float('nan'), 0, 0, 0, 0, 0]]),
    np.array([[0, 0, .01, 0, 0, 0]]),
    np.array([[0, 0, 0, 0, 0, .01]]),
])
def test_invalid_strain_increment_rejected_without_commit(bad):
    p = particles()
    before = p.stress.copy()
    with pytest.raises(ValueError):
        update_linear_elastic_stress(p, bad, LinearElastic(10e6, .3))
    np.testing.assert_array_equal(p.stress, before)


def test_plane_strain_rejects_out_of_plane_initial_shear():
    p = particles(stress=[[0, 0, 0, 0, 1., 0]])
    before = p.stress.copy()
    with pytest.raises(ValueError, match='yz and xz'):
        update_linear_elastic_stress(p, inc(p), LinearElastic(10e6, .3))
    np.testing.assert_array_equal(p.stress, before)


def test_overflow_does_not_commit_any_particle():
    p = particles(n=2)
    before = p.stress.copy()
    values = inc(p, exx=.001)
    values[1, 0] = 1e300
    with np.errstate(over='ignore', invalid='ignore'):
        with pytest.raises(ValueError, match='non-finite'):
            update_linear_elastic_stress(p, values, LinearElastic(1e200, 0.25))
    np.testing.assert_array_equal(p.stress, before)


def test_rigid_body_rotation_is_known_limit_of_nonobjective_update():
    p = particles(stress=[[100., 0., 0., 0., 0., 0.]])
    # The supplied strain increment is zero under rigid rotation. Baseline
    # non-objective code leaves the Cauchy stress unchanged; a later objective
    # update must rotate it. This limitation is deliberate and documented.
    update_linear_elastic_stress(p, inc(p), LinearElastic(10e6, .3))
    np.testing.assert_allclose(p.stress[0], [100., 0., 0., 0., 0., 0.])


def test_3d_particle_set_is_rejected():
    p = ParticleSet([[0., 0., 0.]], [1.], [1.])
    with pytest.raises(ValueError, match='2D plane strain'):
        update_linear_elastic_stress(p, np.zeros((1, 6)), LinearElastic(10e6, .3))


def test_stage3d_increment_to_stress_and_next_grid_internal_force():
    p = ParticleSet([[.5, .5]], [450.], [.25])
    g = BackgroundGrid((0, 0), (1, 1), (1, 1))
    g.velocity_bc_mask[[0, 1], 1] = True
    ip = evaluate_quad4(p, g)
    particle_to_grid(p, g, ip)
    update_grid_momentum(p, g, ip, dt=.01, gravity=(0, -9.81))
    kin = update_particle_kinematics(p, g, ip, dt=.01)
    old_F = p.deformation_gradient.copy()
    old_v = p.volume.copy()
    update_linear_elastic_stress(p, kin.strain_increment, LinearElastic(20e6, .25))
    np.testing.assert_allclose(p.stress[0], [-7848, -23544, -7848, 0, 0, 0], rtol=1e-13)
    np.testing.assert_array_equal(p.deformation_gradient, old_F)
    np.testing.assert_array_equal(p.volume, old_v)
    grid_to_particle(p, g, ip, dt=.01, pic_fraction=1.0)
    # Start the next time step and use updated particle stress for force.
    new_interp = evaluate_quad4(p, g)
    particle_to_grid(p, g, new_interp)
    update_grid_momentum(p, g, new_interp, dt=.0001, gravity=(0, 0))
    assert np.any(np.abs(g.force_internal) > 0.0)
    np.testing.assert_allclose(g.force_internal.sum(axis=0), [0., 0.], atol=1e-10)


def test_compiled_numba():
    p = particles()
    update_linear_elastic_stress(p, inc(p, eyy=.001), LinearElastic(10e6, .3))
    assert linear_elastic_plane_strain_kernel.signatures
