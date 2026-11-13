"""Stage 13B: smooth-face MC, finite-difference constitutive checks, rollback and restart."""
import numpy as np
import pytest

from geopympm.core import ParticleSet, BackgroundGrid
from geopympm.materials import (
    MohrCoulomb, LinearElastic, MaterialState, MC_HISTORY,
    mohr_coulomb_yield, update_mohr_coulomb_stress,
)
from geopympm.io import write_checkpoint, inspect_checkpoint, load_checkpoint
from geopympm.solvers import ExplicitMPMSolver


def particle_set(n=1):
    return ParticleSet([[.37 + .06 * k, .45] for k in range(n)],
                       np.full(n, 1.0), np.full(n, .01))


def mc(*, psi=0., cohesion=10.):
    return MohrCoulomb(1000., .30, cohesion, 30., psi)


def apply(p, strain, m=None, state=None):
    m = mc() if m is None else m
    state = MaterialState(p.n_particles, MC_HISTORY) if state is None else state
    update_mohr_coulomb_stress(p, np.asarray(strain, dtype=np.float64), m, state)
    return state


@pytest.mark.parametrize('kwargs', [
    {'young_modulus': 0}, {'poisson_ratio': 0.5},
    {'cohesion': -1}, {'friction_angle_deg': 90},
    {'friction_angle_deg': -1}, {'dilation_angle_deg': -1},
    {'dilation_angle_deg': 35}, {'cohesion': np.nan},
    {'young_modulus': True},
])
def test_invalid_material_params(kwargs):
    arguments = dict(young_modulus=1000., poisson_ratio=.3, cohesion=10.,
                     friction_angle_deg=30., dilation_angle_deg=0.)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        MohrCoulomb(**arguments)


def test_simple_elastic_increment_and_zero_history():
    p = particle_set()
    inc = np.array([[1e-4, -2e-4, 0, 3e-4, 0, 0]])
    state = apply(p, inc)
    mat = mc()
    lam, G = mat.lame_lambda, mat.shear_modulus
    tr = inc[0, 0] + inc[0, 1]
    np.testing.assert_allclose(p.stress[0, :4], [lam * tr + 2*G*inc[0, 0],
                                                 lam * tr + 2*G*inc[0, 1],
                                                 lam * tr, G*inc[0, 3]], atol=1e-12)
    assert not np.any(state['plastic_strain'])
    assert not np.any(state['equivalent_plastic_strain'])


@pytest.mark.parametrize('psi', [0, 15, 30])
def test_shear_return_on_single_smooth_face(psi):
    p = particle_set()
    state = apply(p, [[0, 0, 0, .1, 0, 0]], m=mc(psi=psi))
    assert abs(mohr_coulomb_yield(p.stress[0], mc(psi=psi))) < 1e-9
    assert state['equivalent_plastic_strain'][0] > 0
    assert state['plastic_strain'][0, 3] > 0
    assert p.stress[0, 4] == p.stress[0, 5] == 0


def test_nonassociated_volumetric_flow_vs_associated():
    p0, p1 = particle_set(), particle_set()
    h0 = apply(p0, [[0, 0, 0, .1, 0, 0]], mc(psi=0))
    h1 = apply(p1, [[0, 0, 0, .1, 0, 0]], mc(psi=30))
    np.testing.assert_allclose(h0['plastic_strain'][0, :3].sum(), 0, atol=1e-13)
    assert h1['plastic_strain'][0, :3].sum() > 0


def test_yield_tension_positive_hydrostatic():
    mat = mc()
    assert mohr_coulomb_yield([0,0,0,0,0,0], mat) < 0
    assert mohr_coulomb_yield([-1e3]*3 + [0]*3, mat) < 0
    assert mohr_coulomb_yield([1e3]*3 + [0]*3, mat) > 0


def test_input_unchanged_when_edge_failure_any_particle():
    p = particle_set(2)
    p.stress[1, :3] = 100  # hydrostatic tension beyond apex
    h = MaterialState(2, MC_HISTORY)
    h['plastic_strain'][:] = .1
    old_s = p.stress.copy()
    old_h = h.snapshot()
    with pytest.raises(ValueError, match='edge/apex'):
        apply(p, [[0,0,0,.1,0,0], [0,0,0,0,0,0]], state=h)
    np.testing.assert_array_equal(p.stress, old_s)
    for name, old in old_h.items():
        np.testing.assert_array_equal(h[name], old)


def test_missing_history_no_silent_creation_in_direct_call():
    with pytest.raises(ValueError, match='Missing'):
        apply(particle_set(), [[0,0,0,.1,0,0]], state=MaterialState(1))


@pytest.mark.parametrize('scheme', ['usl', 'usf', 'musl'])
@pytest.mark.parametrize('interpolation', ['mpm', 'gimp', 'cpgimp', 'cpdi'])
def test_solver_mc_smoke_by_scheme_interpolation(scheme, interpolation):
    p = particle_set()
    if interpolation != 'mpm':
        p.domain_size[:] = p.domain_size0[:] = .025
    g = BackgroundGrid([0,0], [.1,.1], (10,10))
    solver = ExplicitMPMSolver(p,g,mc(),dt=1e-4,scheme=scheme,
                               interpolation=interpolation)
    assert solver.material_state.names == ('plastic_strain', 'equivalent_plastic_strain')
    assert solver._constitutive_adapter.family == 'mc'
    out = solver.step()
    assert out.step == 1
    np.testing.assert_allclose(p.stress, 0, atol=1e-12)


def test_solver_rejects_objective_stress_mode_for_mc():
    with pytest.raises(ValueError, match='does not support'):
        ExplicitMPMSolver(particle_set(),BackgroundGrid([0,0],[.1,.1],(10,10)),
                          mc(),dt=1e-4,stress_update='corotational')


def test_solver_step_rollback_on_apex_error():
    p = particle_set()
    p.stress[0,:3] = 100.
    g = BackgroundGrid([0,0],[.1,.1],(10,10))
    s = ExplicitMPMSolver(p,g,mc(),dt=1e-4)
    old_p = p.stress.copy()
    old_h = s.material_state.snapshot()
    with pytest.raises(ValueError, match='edge/apex'):
        s.step()
    assert s.step_count == 0
    assert s.time == 0
    np.testing.assert_array_equal(p.stress, old_p)
    for name in old_h:
        np.testing.assert_array_equal(s.material_state[name],old_h[name])


def test_mapped_material_parameters_and_ids():
    p=particle_set(2)
    p.material_id[1] = 1
    h=MaterialState(2,MC_HISTORY)
    apply(p, [[0,0,0,.1,0,0],[0,0,0,.1,0,0]],
          {0:mc(), 1:mc(cohesion=20.)},h)
    assert p.stress.shape == (2,6)
    np.testing.assert_allclose(p.stress[0,3], 10 * np.cos(np.deg2rad(30)), atol=1e-9)
    np.testing.assert_allclose(p.stress[1,3], 20 * np.cos(np.deg2rad(30)), atol=1e-9)


def test_checkpoint_roundtrip_mc_and_continuation(tmp_path):
    p=particle_set()
    g=BackgroundGrid([0,0],[.1,.1],(10,10))
    s=ExplicitMPMSolver(p,g,mc(),dt=1e-4,scheme='musl')
    s.material_state['plastic_strain'][0,3]=.2
    s.material_state['equivalent_plastic_strain'][0]=.12
    s.step()
    record=write_checkpoint(s,tmp_path)
    assert inspect_checkpoint(record.manifest_path)['schema_version'] == 6
    r=load_checkpoint(record.manifest_path)
    assert isinstance(r.materials, MohrCoulomb)
    np.testing.assert_array_equal(r.material_state['plastic_strain'],s.material_state['plastic_strain'])
    np.testing.assert_array_equal(r.material_state['equivalent_plastic_strain'],s.material_state['equivalent_plastic_strain'])
    s.step();r.step()
    np.testing.assert_array_equal(r.particles.stress,s.particles.stress)
    np.testing.assert_array_equal(r.material_state['plastic_strain'],s.material_state['plastic_strain'])


def test_legacy_linear_checkpoint_unmodified(tmp_path):
    p=particle_set();g=BackgroundGrid([0,0],[.1,.1],(10,10))
    solver=ExplicitMPMSolver(p,g,LinearElastic(1000,.3),dt=1e-4)
    record=write_checkpoint(solver,tmp_path)
    assert inspect_checkpoint(record.manifest_path)['schema_version'] == 1
    assert isinstance(load_checkpoint(record.manifest_path).materials,LinearElastic)


def test_end_to_end_solver_reaches_yield_under_prescribed_grid_shear():
    """Exercise the actual MPM kinematics -> MC dispatch -> history path."""
    p=particle_set()
    g=BackgroundGrid([0,0],[.1,.1],(10,10))
    # Affine field v_x=200*y, v_y=0, prescribed to all grid nodes.
    g.velocity_bc_mask[:] = True
    g.velocity_bc_value[:,0] = 200 * g.node_coordinates[:,1]
    mat=mc(cohesion=2.)
    s=ExplicitMPMSolver(p,g,mat,dt=1e-4)
    s.step()
    assert abs(mohr_coulomb_yield(p.stress[0],mat)) < 1e-9
    assert s.material_state['equivalent_plastic_strain'][0] > 0.
    assert p.strain[0,3] == pytest.approx(0.02)
    assert s.step_count == 1
