"""Stage 10F independent randomized interpolation invariants, no solver changes."""
import numpy as np
import pytest

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.interpolation.cpdi import update_cpdi_domain, evaluate_cpdi
from geopympm.interpolation.gimp import evaluate_gimp
from geopympm.interpolation.mpm import evaluate_quad4
from validation.stage10f.final_audit import measure_invariants, METHODS, run


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('seed', (17, 101, 4003))
def test_randomized_affine_reproduction_and_conservation(method, seed):
    result = measure_invariants(method, count=64, seed=seed)
    assert all(result['checks'].values()), result
    assert result['support_slots'] == (4 if method == 'mpm' else 16)


@pytest.mark.parametrize('method', METHODS)
def test_mass_and_internal_force_symmetry_under_random_positions(method):
    result = measure_invariants(method, count=257, seed=97)
    assert result['mass_transfer_relative_error'] < 1e-12
    assert result['momentum_transfer_max_abs_error'] < 1e-10
    assert result['total_internal_force_max_abs_N'] < 1e-10


def test_cpdi_affine_reproduction_for_domain_extending_multiple_cells():
    grid = BackgroundGrid((0., 0.), (.4, .5), (36, 30))
    p = ParticleSet([[6., 6.]], [1.], [.2], domain_size=[[.6, .55]])
    p.deformation_gradient[0] = [[2.0, .9], [.4, 1.3]]
    update_cpdi_domain(p, grid)
    assert p.domain_size[0, 0] > grid.cell_size[0]
    ip = evaluate_cpdi(p, grid)
    X = grid.node_coordinates[ip.node_ids[0]]
    np.testing.assert_allclose(ip.weights.sum(), 1., atol=2e-14)
    np.testing.assert_allclose(X.T @ ip.weights[0], p.position[0], atol=1e-13)
    np.testing.assert_allclose(X.T @ ip.gradients[0], np.eye(2), atol=1e-13)


def test_gimp_and_point_mpm_coincide_if_rectangular_domain_inside_one_cell():
    grid = BackgroundGrid((0., 0.), (1., 1.), (12, 12))
    p = ParticleSet([[5.5, 6.5]], [1.], [.1], domain_size=[[.23, .37]])
    a, b = evaluate_quad4(p, grid), evaluate_gimp(p, grid)
    mass_a = np.bincount(a.node_ids[0], weights=a.weights[0], minlength=grid.n_nodes)
    mass_b = np.bincount(b.node_ids[0], weights=b.weights[0], minlength=grid.n_nodes)
    np.testing.assert_allclose(mass_a, mass_b, atol=1e-14)
    for d in range(2):
        da = np.bincount(a.node_ids[0], weights=a.gradients[0,:,d], minlength=grid.n_nodes)
        db = np.bincount(b.node_ids[0], weights=b.gradients[0,:,d], minlength=grid.n_nodes)
        np.testing.assert_allclose(da, db, atol=1e-14)


def test_audit_command_writes_machine_readable_results(tmp_path):
    summary = run(tmp_path, count=12, seed=35, performance=False)
    assert summary['all_invariant_checks_pass']
    assert (tmp_path/'10F_audit.json').exists()
    assert 'performance' not in summary
