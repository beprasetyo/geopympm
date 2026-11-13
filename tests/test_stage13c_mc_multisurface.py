"""Stage 13C: six-plane Mohr–Coulomb multi-surface return tests."""
import json
import math
import numpy as np
import pytest

from geopympm.core import ParticleSet, BackgroundGrid
from geopympm.materials import (
    MohrCoulomb, MaterialState, MC_HISTORY, mohr_coulomb_yield,
    update_mohr_coulomb_stress,
)
from geopympm.materials.mohr_coulomb import (
    _mc_multisurface_kernel, _particle_mc_parameters,
)
from geopympm.solvers import ExplicitMPMSolver
from geopympm.io import write_checkpoint, inspect_checkpoint, load_checkpoint


def particles(n=1):
    return ParticleSet([[0.35 + .1*i, .45] for i in range(n)], np.ones(n),
                       np.full(n, .01))


def model(psi=0, c=10):
    return MohrCoulomb(1000., .3, c, 30., psi)


# Grounded deterministic paths from zero initial Cauchy stress. Each
# is a 2D-plane-strain increment, with zero zz/yz/xz components.
FACE = np.array([-.14993903, .12121359, 0., -.32762793, 0., 0.])
EDGE = np.array([.21555855, -.19711230, 0., -.25622659, 0., 0.])
APEX = np.array([.01072789, .13656687, 0., .06050925, 0., 0.])


def apply_one(inc, psi=0):
    p = particles()
    h = MaterialState(1, MC_HISTORY)
    mat = model(psi)
    update_mohr_coulomb_stress(p, np.asarray(inc, dtype=float).reshape(1, 6), mat, h)
    return p, h, mat


def branch_for(inc, psi):
    p = particles()
    h = MaterialState(1, MC_HISTORY)
    args = _particle_mc_parameters(p, model(psi))
    sig, ep, q, statuses, branches = _mc_multisurface_kernel(
        p.stress, np.asarray(inc, dtype=float).reshape(1, 6), *args,
        h['plastic_strain'], h['equivalent_plastic_strain'])
    assert statuses[0] == 0
    return int(branches[0]), sig[0], ep[0], q[0]


@pytest.mark.parametrize('increment,psi,expected_branch', [
    (FACE, 0, 1),
    (EDGE, 0, 2),
    (APEX, 15, 3),
])
def test_three_plastic_branches(increment, psi, expected_branch):
    branch, sigma, epsp, eqp = branch_for(increment, psi)
    assert branch == expected_branch
    p, h, m = apply_one(increment, psi)
    np.testing.assert_allclose(sigma, p.stress[0], atol=1e-10, rtol=0)
    np.testing.assert_allclose(epsp, h['plastic_strain'][0], atol=1e-12, rtol=0)
    assert eqp == pytest.approx(h['equivalent_plastic_strain'][0])
    assert abs(mohr_coulomb_yield(sigma, m)) < 5e-8
    assert eqp > 0
    assert p.stress[0, 4] == p.stress[0, 5] == 0


def test_edge_two_equal_principal_stresses():
    p, _, _ = apply_one(EDGE, 0)
    eig = np.linalg.eigvalsh([[p.stress[0,0], p.stress[0,3], 0],
                             [p.stress[0,3], p.stress[0,1], 0],
                             [0,0,p.stress[0,2]]])
    assert min(np.diff(eig)) == pytest.approx(0.0, abs=1e-8)
    assert max(np.diff(eig)) > 0.01


def test_apex_all_principal_stresses_equal():
    p, h, m = apply_one(APEX, 15)
    expected = m.cohesion / math.tan(math.radians(m.friction_angle_deg))
    np.testing.assert_allclose(p.stress[0, :3], expected, atol=2e-8)
    assert abs(p.stress[0,3]) < 2e-8
    assert h['plastic_strain'][0, :3].sum() > 0


def test_hydrostatic_apex_dilatant_and_isochoric_incompatibility():
    p = particles(2)
    p.stress[:, :3] = 100.
    p.material_id[1] = 1
    h = MaterialState(2, MC_HISTORY)
    before_s = p.stress.copy()
    before_h = h.snapshot()
    with pytest.raises(ValueError, match='edge/apex'):
        update_mohr_coulomb_stress(p, np.zeros((2, 6)),
                                   {0: model(15), 1: model(0)}, h)
    np.testing.assert_array_equal(before_s, p.stress)
    for key, value in before_h.items():
        np.testing.assert_array_equal(h[key], value)
    # With positive dilation, the true multisurface apex return exists.
    p = particles()
    p.stress[0,:3] = 100.
    h = MaterialState(1, MC_HISTORY)
    update_mohr_coulomb_stress(p, np.zeros((1,6)), model(15), h)
    np.testing.assert_allclose(p.stress[0,:3], 10 / math.tan(math.radians(30)), atol=1e-8)
    assert h['plastic_strain'][0,:3].sum() > 0
    assert h['equivalent_plastic_strain'][0] == pytest.approx(0, abs=1e-12)


def test_multiple_material_ids_face_edge_apex_in_one_batch():
    p = particles(3)
    p.material_id[:] = [0, 1, 2]
    h = MaterialState(3, MC_HISTORY)
    increments = np.stack([FACE, EDGE, APEX])
    mapping = {0:model(0), 1:model(0), 2:model(15)}
    update_mohr_coulomb_stress(p, increments, mapping, h)
    for i, m in mapping.items():
        assert abs(mohr_coulomb_yield(p.stress[i], m)) < 1e-8
        assert h['equivalent_plastic_strain'][i] > 0


@pytest.mark.parametrize('increment,psi', [(FACE,0), (EDGE,0), (APEX,15)])
def test_inplane_rotation_objectivity_of_return_mapping(increment, psi):
    angle = .34
    c, s = math.cos(angle), math.sin(angle)
    q = np.array([[c,-s],[s,c]])
    strain = np.array([[increment[0], increment[3]/2],
                       [increment[3]/2, increment[1]]])
    rotated = q @ strain @ q.T
    inc2 = [rotated[0,0], rotated[1,1], 0, 2*rotated[0,1], 0, 0]
    p0, h0, m = apply_one(increment, psi)
    p1, h1, _ = apply_one(inc2, psi)
    st0 = np.array([[p0.stress[0,0], p0.stress[0,3]],
                    [p0.stress[0,3], p0.stress[0,1]]])
    st1 = np.array([[p1.stress[0,0], p1.stress[0,3]],
                    [p1.stress[0,3], p1.stress[0,1]]])
    np.testing.assert_allclose(st1, q @ st0 @ q.T, atol=1e-8, rtol=0)
    assert p1.stress[0,2] == pytest.approx(p0.stress[0,2], abs=1e-8)
    assert h1['equivalent_plastic_strain'][0] == pytest.approx(h0['equivalent_plastic_strain'][0], abs=1e-9)


def test_elastic_stress_history_is_unchanged():
    p, h, _ = apply_one([1e-5, -2e-5, 0, 3e-5, 0, 0], 0)
    assert not np.any(h['plastic_strain'])
    assert not np.any(h['equivalent_plastic_strain'])
    assert branch_for([1e-5, -2e-5, 0, 3e-5, 0, 0], 0)[0] == 0


def test_zero_dilation_does_not_invent_apex_flow():
    p = particles()
    p.stress[0, :3] = 100.
    h = MaterialState(1, MC_HISTORY)
    before = p.stress.copy()
    with pytest.raises(ValueError, match='edge/apex'):
        update_mohr_coulomb_stress(p, np.zeros((1, 6)), model(0), h)
    np.testing.assert_array_equal(before, p.stress)
    assert np.all(h['plastic_strain'] == 0)


def test_stage13c_schema6_checkpoint_restart(tmp_path):
    p = particles()
    g = BackgroundGrid([0,0], [.1,.1], (10,10))
    solver = ExplicitMPMSolver(p, g, model(15), dt=1e-4, scheme='musl')
    solver.material_state['plastic_strain'][0] = [.1, .2, .0, .05, 0, 0]
    solver.material_state['equivalent_plastic_strain'][0] = .15
    solver.step()
    cp = write_checkpoint(solver, tmp_path)
    meta = inspect_checkpoint(cp.manifest_path)
    assert meta['schema_version'] == 6
    assert meta['constitutive_algorithm'] == 'mohr_coulomb_multisurface_active_set_v1'
    resumed = load_checkpoint(cp.manifest_path)
    np.testing.assert_array_equal(resumed.material_state['plastic_strain'], solver.material_state['plastic_strain'])
    np.testing.assert_array_equal(resumed.particles.stress, solver.particles.stress)
    resumed.step()
    solver.step()
    np.testing.assert_array_equal(resumed.particles.stress, solver.particles.stress)


def test_schema6_rejects_algorithm_metadata_tampering(tmp_path):
    s = ExplicitMPMSolver(particles(),BackgroundGrid([0,0],[.1,.1],(10,10)),
                          model(15), dt=1e-4)
    cp = write_checkpoint(s, tmp_path)
    meta = json.loads(cp.manifest_path.read_text())
    for wrong in (None, 'unknown', 'mohr_coulomb_smooth_face_v1'):
        modified = dict(meta)
        modified['constitutive_algorithm'] = wrong
        cp.manifest_path.write_text(json.dumps(modified))
        with pytest.raises(ValueError, match='return-mapping algorithm'):
            inspect_checkpoint(cp.manifest_path, verify=False)


def test_legacy_schema5_readable_and_rewritten_as_schema6(tmp_path):
    s = ExplicitMPMSolver(particles(),BackgroundGrid([0,0],[.1,.1],(10,10)),
                          model(15), dt=1e-4)
    cp = write_checkpoint(s, tmp_path)
    m = json.loads(cp.manifest_path.read_text())
    m['schema_version'] = 5
    del m['constitutive_algorithm']
    cp.manifest_path.write_text(json.dumps(m))
    assert inspect_checkpoint(cp.manifest_path)['schema_version'] == 5
    loaded = load_checkpoint(cp.manifest_path)
    cp_new = write_checkpoint(loaded, tmp_path/'next')
    assert inspect_checkpoint(cp_new.manifest_path)['schema_version'] == 6


def test_solver_real_edge_apex_rollback_and_recovery():
    # The solver remains able to commit the state for valid MC initial values.
    s = ExplicitMPMSolver(particles(),BackgroundGrid([0,0],[.1,.1],(10,10)),
                          model(0), dt=1e-4)
    s.particles.stress[0,:3] = 100.  # inadmissible hydrostatic initial state
    old = s.particles.stress.copy()
    with pytest.raises(ValueError, match='edge/apex'):
        s.step()
    assert s.step_count == 0
    np.testing.assert_array_equal(s.particles.stress, old)
    s.particles.stress[:] = 0.
    assert s.step().step == 1


@pytest.mark.parametrize('scheme', ['usl', 'usf', 'musl'])
@pytest.mark.parametrize('interpolation', ['mpm', 'gimp', 'cpgimp', 'cpdi'])
@pytest.mark.parametrize('increment,psi', [(EDGE, 0), (APEX, 15)])
def test_full_solver_plastic_edge_apex_all_schemes(interpolation, scheme, increment, psi):
    """Affine prescribed field exercises actual P2G, kinematics and MC dispatch."""
    p = particles()
    if interpolation != 'mpm':
        p.domain_size[:] = p.domain_size0[:] = .025
    g = BackgroundGrid([0, 0], [.1, .1], (10, 10))
    dt = 1e-4
    # Lxx=epsilon_xx/dt, Lyy=epsilon_yy/dt, Lxy=gamma_xy/dt.
    L = np.array([[increment[0], increment[3]], [0, increment[1]]]) / dt
    g.velocity_bc_mask[:] = True
    g.velocity_bc_value[:] = (g.node_coordinates - p.position[0]) @ L.T
    m = model(psi)
    solver = ExplicitMPMSolver(p, g, m, dt=dt, scheme=scheme,
                               interpolation=interpolation)
    result = solver.step()
    assert result.step == 1
    assert abs(mohr_coulomb_yield(p.stress[0], m)) < 1e-8
    assert solver.material_state['equivalent_plastic_strain'][0] > 0
    np.testing.assert_allclose(p.position[0], [.35, .45], atol=1e-12)
