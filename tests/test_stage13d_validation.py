"""Stage 13D: independent MC analytical references + MPM affine-shear patch tests."""
import json
import math

import numpy as np
import pytest

from geopympm.core import ParticleSet
from geopympm.materials import (MC_HISTORY, MaterialState, MohrCoulomb,
                                 mohr_coulomb_yield, update_mohr_coulomb_stress)
from validation.stage13d.references import (
    shear_modulus, tresca_pure_shear, triaxial_compression_strength,
    plane_strain_confined_yield_strain,
)
from validation.stage13d.run import (
    INTERPOLATIONS, SCHEMES, shear_material_point,
    confined_compression_material_point, triaxial_yield_envelope,
    affine_shear_boundary_value, affine_confined_compression_boundary_value, main,
)


@pytest.mark.parametrize('nu', [.0, .2, .3, .45])
def test_shear_reference_initial_elastic_slope(nu):
    e = 1000.
    g = e / (2 * (1 + nu))  # independently derived Hooke slope
    result = tresca_pure_shear(.001, e, nu, 10.)
    assert result['tau_pa'] == pytest.approx(.001 * g, abs=1e-13)
    assert result['gamma_plastic'] == pytest.approx(0., abs=1e-14)
    assert shear_modulus(e, nu) == pytest.approx(g)


@pytest.mark.parametrize('total_gamma', [.001, .02, .026, .05, .08, -.08])
def test_material_point_pure_shear_tresca_closed_form(total_gamma):
    e, nu, c = 1000., .3, 10.
    p = ParticleSet([[.5, .5]], [1.], [.01])
    h = MaterialState(1, MC_HISTORY)
    m = MohrCoulomb(e, nu, c, 0., 0.)
    nsteps = 40
    for _ in range(nsteps):
        d = np.zeros((1,6))
        d[0,3] = total_gamma/nsteps
        update_mohr_coulomb_stress(p, d, m, h)
    expected = tresca_pure_shear(total_gamma, e, nu, c)
    np.testing.assert_allclose(p.stress[0,:3], [0,0,0], atol=2e-10)
    assert p.stress[0,3] == pytest.approx(expected['tau_pa'], abs=1e-10)
    assert h['plastic_strain'][0,3] == pytest.approx(expected['gamma_plastic'], abs=1e-10)
    assert h['equivalent_plastic_strain'][0] == pytest.approx(expected['equivalent_plastic_strain'], abs=1e-10)
    assert mohr_coulomb_yield(p.stress[0], m) <= 1e-8


def test_elastic_unloading_and_reloading_after_yield():
    p = ParticleSet([[.5,.5]], [1.], [.01])
    h = MaterialState(1, MC_HISTORY)
    m = MohrCoulomb(1000., .3, 10., 0., 0.)
    d = np.zeros((1,6))
    d[0,3] = .06
    update_mohr_coulomb_stress(p, d, m, h)
    prior_history = h.snapshot()
    d[0,3] = -.015
    update_mohr_coulomb_stress(p, d, m, h)
    assert p.stress[0,3] == pytest.approx(10. - m.shear_modulus * .015, abs=1e-9)
    for key in h.names:
        np.testing.assert_allclose(h[key], prior_history[key], atol=1e-12)
    d[0,3] = .015
    update_mohr_coulomb_stress(p, d, m, h)
    assert p.stress[0,3] == pytest.approx(10., abs=1e-9)
    for key in h.names:
        np.testing.assert_allclose(h[key], prior_history[key], atol=1e-12)


@pytest.mark.parametrize('friction', [0., 15., 30., 45.])
@pytest.mark.parametrize('confinement', [0., 5., 20.])
def test_triaxial_yield_threshold_is_independent_closed_form(friction, confinement):
    c = 10.
    qcrit = triaxial_compression_strength(confinement, c, friction)
    m = MohrCoulomb(1000., .3, c, friction, 0.)
    # Compression-positive principal magnitudes, mapped into the solver's
    # tension-positive three-dimensional plane-strain stress vector.
    def residual(q):
        return mohr_coulomb_yield(np.array([-confinement-q, -confinement,
                                            -confinement, 0,0,0]), m)
    assert residual(qcrit) == pytest.approx(0, abs=2e-11)
    assert residual(qcrit - 1.0) < 0
    assert residual(qcrit + 1.0) > 0


@pytest.mark.parametrize('nu,phi', [(.2, 0.), (.2, 10.), (.3, 10.)])
def test_confined_compression_first_yield_formula(nu, phi):
    m = MohrCoulomb(1000., nu, 10., phi, 0.)
    onset = plane_strain_confined_yield_strain(1000., nu, 10., phi)
    p = ParticleSet([[.5, .5]], [1.], [.01])
    h = MaterialState(1, MC_HISTORY)
    for fraction in (.25, .5, .24):
        d = np.array([[-onset*fraction,0,0,0,0,0]])
        update_mohr_coulomb_stress(p, d, m, h)
        assert h['equivalent_plastic_strain'][0] == pytest.approx(0, abs=1e-12)
    d = np.array([[-onset*.05,0,0,0,0,0]])
    update_mohr_coulomb_stress(p, d, m, h)
    assert h['equivalent_plastic_strain'][0] > 0
    assert abs(mohr_coulomb_yield(p.stress[0], m)) < 1e-9


def test_pure_shear_work_and_plastic_dissipation_closed_form():
    rows = shear_material_point(80)
    g = shear_modulus(1000., .3)
    c = 10.
    gamma = rows[-1]['gamma']
    expected_work = c * gamma - c**2/(2*g)
    # For this discretization the exact elastic-plastic transition at
    # gamma=c/G=.026 falls on a sample, making trapezoidal work exact.
    work = 0.
    tau_old = 0.
    gamma_old = 0.
    for row in rows:
        work += .5*(tau_old + row['tau_numerical_pa'])*(row['gamma']-gamma_old)
        tau_old, gamma_old = row['tau_numerical_pa'], row['gamma']
    assert work == pytest.approx(expected_work, abs=1e-10)
    dissipation = work - c**2/(2*g)
    assert dissipation == pytest.approx(c*(gamma-c/g), abs=1e-10)
    assert dissipation > 0


@pytest.mark.parametrize('method', INTERPOLATIONS)
@pytest.mark.parametrize('scheme', SCHEMES)
def test_affine_shear_uniform_boundary_value_matches_tresca(method, scheme):
    output = affine_shear_boundary_value(method=method, scheme=scheme, nsteps=40)
    assert output['max_tau_error_pa'] < 1e-7
    assert output['max_eqp_error'] < 1e-8
    assert output['max_sigma_normal_abs_pa'] < 1e-7
    assert output['max_yield_function_pa'] < 1e-7
    assert output['min_jacobian'] == pytest.approx(1., abs=1e-9)
    assert output['total_mass_kg'] == pytest.approx(4.)



@pytest.mark.parametrize('method', INTERPOLATIONS)
@pytest.mark.parametrize('scheme', SCHEMES)
def test_confined_compression_uniform_mpm_matches_onset_and_material_point(method, scheme):
    result = affine_confined_compression_boundary_value(method=method, scheme=scheme)
    assert result['first_plastic_sample'] is not None
    assert 0 <= result['first_plastic_sample'] - result['analytical_yield_onset'] <= 20*1e-4 + 1e-12
    assert result['max_stress_error_pa'] < 1e-7
    assert result['max_eqp_error'] < 1e-8
    assert result['max_yield_residual_pa'] < 1e-7

def test_triaxial_reference_and_confined_onset_runner():
    envelope = triaxial_yield_envelope()
    assert len(envelope) == 5
    assert max(abs(r['yield_residual_pa']) for r in envelope) < 1e-10
    compression = confined_compression_material_point(nsteps=60)
    assert compression['first_plastic_sample'] is not None
    assert 0 <= (compression['first_plastic_sample'] - compression['onset_analytical']) <= compression['increment_size'] + 1e-12
    assert compression['samples'][-1]['eqp'] > 0
    assert abs(compression['samples'][-1]['yield_residual_pa']) < 1e-8


def test_reproducible_cli_outputs_and_thresholds(tmp_path):
    code = main(['--output-dir', str(tmp_path), '--steps','40',
                 '--interpolations','mpm', '--schemes','usl'])
    assert code == 0
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert summary['passed'] is True
    assert summary['mpm_cases'] == 1
    assert summary['compression_mpm_cases'] == 1
    assert {p.name for p in tmp_path.glob('*.csv')} == {
        'pure_shear.csv', 'confined_compression.csv',
        'triaxial_yield_envelope.csv', 'affine_shear_mpm.csv',
        'affine_confined_compression_mpm.csv'}


@pytest.mark.parametrize('bad', [(-1,.3), (1000,.5), (1000,-1.0)])
def test_analytic_reference_rejects_nonphysical_elastic_input(bad):
    with pytest.raises(ValueError):
        shear_modulus(*bad)


def test_no_yield_confined_path_rejected_when_no_closed_form_onset():
    with pytest.raises(ValueError, match='no yield onset'):
        plane_strain_confined_yield_strain(1000., .3, 10., 30.)


@pytest.mark.parametrize('method', ['mpm', 'cpdi'])
def test_plastic_shear_checkpoint_restart_equals_continuous(method, tmp_path):
    """Exercise material-history schema-6 restart halfway through plastic loading."""
    from geopympm.core import BackgroundGrid
    from geopympm.solvers import ExplicitMPMSolver
    from geopympm.io import write_checkpoint, load_checkpoint

    def init():
        p = ParticleSet([[.32,.32],[.46,.38],[.54,.52],[.68,.62]],
                        np.ones(4), np.full(4, .01))
        if method == 'cpdi':
            p.domain_size[:] = p.domain_size0[:] = .025
        g = BackgroundGrid([0,0],[.1,.1],(10,10))
        g.velocity_bc_mask[:] = True
        g.velocity_bc_value[:,0] = 20*(g.node_coordinates[:,1]-.5)
        return ExplicitMPMSolver(
            p, g, MohrCoulomb(1000.,.3,10.,0.,0.), dt=1e-4,
            scheme='musl', interpolation=method,
        )

    continuous, restarted = init(), init()
    continuous.run(40)
    restarted.run(20)
    assert np.any(restarted.material_state['equivalent_plastic_strain'] > 0.)
    checkpoint = write_checkpoint(restarted, tmp_path)
    restarted = load_checkpoint(checkpoint.manifest_path)
    restarted.run(20)
    assert continuous.step_count == restarted.step_count == 40
    for key in ('position', 'stress', 'deformation_gradient'):
        np.testing.assert_array_equal(getattr(continuous.particles, key),
                                      getattr(restarted.particles, key))
    for key in continuous.material_state.names:
        np.testing.assert_array_equal(continuous.material_state[key], restarted.material_state[key])
