"""Stage 10 uGIMP basis, support, solver integration and regression protections."""
import numpy as np
import pytest

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.interpolation import evaluate_quad4, evaluate_gimp
from geopympm.interpolation.gimp import validate_current_gimp
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver
from geopympm.solvers.p2g import particle_to_grid
from geopympm.solvers.grid_update import update_grid_momentum


def make_case(position=((2.37, 2.41),), side=((0.5, 0.5),), velocity=None):
    n = len(position)
    p = ParticleSet(position=position, mass=np.ones(n), volume=np.full(n, 0.25),
                    velocity=velocity, domain_size=side)
    g = BackgroundGrid((0, 0), (1, 1), (6, 6))
    return p, g


def test_partition_unity_gradient_and_affine_precision():
    points = [(1.1, 1.2), (1.99, 2.05), (2.0, 2.0), (3.49, 4.31)]
    p, grid = make_case(points, [(0.6, 0.7)] * len(points))
    ip = evaluate_gimp(p, grid)
    assert ip.node_ids.shape == (4, 16)
    assert ip.weights.shape == (4, 16)
    assert ip.gradients.shape == (4, 16, 2)
    np.testing.assert_allclose(ip.weights.sum(axis=1), 1.0, atol=1e-13)
    np.testing.assert_allclose(ip.gradients.sum(axis=1), 0.0, atol=1e-13)
    assert np.min(ip.weights) >= 0
    # A normalized integrated hat reproduces any linear nodal field.
    nodes = grid.node_coordinates[ip.node_ids]
    np.testing.assert_allclose(np.sum(ip.weights[:, :, None] * nodes, axis=1),
                               p.position, atol=1e-13)
    grad_coordinate = np.einsum('pai,paj->pij', nodes, ip.gradients)
    np.testing.assert_allclose(grad_coordinate, np.broadcast_to(np.eye(2), (4, 2, 2)),
                               atol=1e-13)


def test_gradient_matches_central_difference():
    p, grid = make_case(position=((2.13, 2.74),), side=((0.65, 0.48),))
    reference = evaluate_gimp(p, grid)
    for axis in (0, 1):
        eps = 1e-6
        p.position[0, axis] += eps
        plus = evaluate_gimp(p, grid)
        p.position[0, axis] -= 2 * eps
        minus = evaluate_gimp(p, grid)
        p.position[0, axis] += eps
        np.testing.assert_array_equal(plus.node_ids, reference.node_ids)
        np.testing.assert_array_equal(minus.node_ids, reference.node_ids)
        np.testing.assert_allclose((plus.weights-minus.weights)/(2*eps),
                                   reference.gradients[:, :, axis], atol=1e-9)


def test_small_domain_reduces_to_quad4_weights_and_gradients():
    p, grid = make_case(position=((2.37, 2.41),), side=((1e-5, 1e-5),))
    quad = evaluate_quad4(p, grid)
    gimp = evaluate_gimp(p, grid)
    for i, node in enumerate(quad.node_ids[0]):
        selector = gimp.node_ids[0] == node
        assert gimp.weights[0, selector].sum() == pytest.approx(quad.weights[0, i], abs=1e-10)
        np.testing.assert_allclose(gimp.gradients[0, selector].sum(axis=0),
                                   quad.gradients[0, i], atol=1e-10)


def test_gimp_nodal_force_continuous_across_cell_boundary():
    grid = BackgroundGrid((0, 0), (1, 1), (4, 4))

    def force(position, method):
        p = ParticleSet([[position, 1.5]], [1.], [0.2], domain_size=[[0.5, 0.5]],
                        stress=[[100., 0., 0., 0., 0., 0.]])
        ip = evaluate_gimp(p, grid) if method == 'gimp' else evaluate_quad4(p, grid)
        particle_to_grid(p, grid, ip)
        update_grid_momentum(p, grid, ip, 0.001)
        return grid.force_internal[grid.node_id(1, 1), 0]

    left_mpm, right_mpm = force(0.999, 'mpm'), force(1.001, 'mpm')
    left_gimp, right_gimp = force(0.999, 'gimp'), force(1.001, 'gimp')
    assert abs(left_mpm-right_mpm) > 19.0
    assert abs(left_gimp-right_gimp) < 0.25
    assert left_gimp == pytest.approx(-right_gimp, abs=1e-12)


@pytest.mark.parametrize('kin', ('euler', 'exponential'))
@pytest.mark.parametrize('adv', ('euler', 'affine_exponential'))
@pytest.mark.parametrize('pic', (0.0, 1.0))
def test_explicit_gimp_rigid_translation(kin, adv, pic):
    p, grid = make_case(position=((2.1, 2.2), (3.4, 3.3)),
                        side=((0.6, 0.6), (0.4, 0.4)),
                        velocity=((0.5, -0.3), (0.5, -0.3)))
    solver = ExplicitMPMSolver(p, grid, LinearElastic(100., 0.25),
                               dt=0.01, pic_fraction=pic,
                               interpolation='gimp', kinematics_update=kin,
                               stress_update='corotational', advection_update=adv)
    records = solver.run(15)
    np.testing.assert_allclose(p.position, p.position0 + np.array([0.075, -0.045]), atol=2e-13)
    np.testing.assert_allclose(p.velocity, [[0.5, -0.3], [0.5, -0.3]], atol=2e-13)
    np.testing.assert_allclose(p.stress, 0, atol=2e-12)
    np.testing.assert_allclose(p.deformation_gradient, np.broadcast_to(np.eye(2), (2, 2, 2)),
                               atol=2e-13)
    np.testing.assert_array_equal(p.domain_size, p.domain_size0)
    assert records[-1].total_mass == pytest.approx(2.0)


def test_reject_missing_large_and_outside_domains():
    grid = BackgroundGrid((0, 0), (1, 1), (4, 4))
    p = ParticleSet([[1.5, 1.5]], [1.], [0.25])
    with pytest.raises(ValueError, match='domain_size'):
        evaluate_gimp(p, grid)
    p.domain_size[:] = (1.01, 0.5)
    with pytest.raises(ValueError, match='domain_size <='):
        evaluate_gimp(p, grid)
    p.domain_size[:] = (0.5, 0.5)
    p.position[0] = (0.1, 1.5)
    with pytest.raises(ValueError, match='extends beyond'):
        evaluate_gimp(p, grid)


def test_stale_interpolation_detects_domain_change_even_when_weights_identical():
    p, grid = make_case(position=((2.5, 2.5),), side=((0.25, 0.25),))
    ip = evaluate_gimp(p, grid)
    p.domain_size[:] = (0.3, 0.3)
    with pytest.raises(ValueError, match='stale'):
        validate_current_gimp(p, grid, ip)
    with pytest.raises(ValueError, match='stale'):
        particle_to_grid(p, grid, ip)


def test_failed_gimp_domain_exit_rolls_back_entire_solver_step():
    p = ParticleSet([[0.27, 2.0]], [5.], [0.2], velocity=[[-1.0, 0.0]],
                    domain_size=[[0.5, 0.5]])
    grid = BackgroundGrid((0, 0), (1, 1), (4, 4))
    solver = ExplicitMPMSolver(p, grid, LinearElastic(10., 0.25), dt=0.05,
                               interpolation='gimp')
    before = p.position.copy()
    with pytest.raises(ValueError, match='domain extends beyond'):
        solver.step()
    np.testing.assert_array_equal(p.position, before)
    assert solver.step_count == 0
    assert grid.mass.sum() == 0.0


def test_default_solver_remains_quad4():
    p, grid = make_case()
    solver = ExplicitMPMSolver(p, grid, LinearElastic(100., 0.2), dt=.001)
    assert solver.interpolation == 'mpm'
    assert solver.step().active_nodes == 4


def test_nonsquare_grid_and_domains_reproduce_affine_field():
    grid = BackgroundGrid(origin=(-2., 1.), cell_size=(0.5, 2.0), n_cells=(12, 6))
    p = ParticleSet(position=[[0.13, 5.8]], mass=[2.], volume=[0.4],
                    domain_size=[[0.3, 1.2]])
    ip = evaluate_gimp(p, grid)
    coords = grid.node_coordinates[ip.node_ids]
    np.testing.assert_allclose(np.einsum('pa,pad->pd', ip.weights, coords),
                               p.position, atol=1e-12)
    np.testing.assert_allclose(np.einsum('pai,paj->pij', coords, ip.gradients),
                               np.eye(2)[None], atol=1e-12)


def test_domains_touching_grid_boundary_are_valid():
    grid = BackgroundGrid(origin=(0., 0.), cell_size=(1., 1.), n_cells=(4, 4))
    p = ParticleSet([[0.25, 3.75]], mass=[1.], volume=[0.2],
                    domain_size=[[0.5, 0.5]])
    ip = evaluate_gimp(p, grid)
    np.testing.assert_allclose(ip.weights.sum(axis=1), 1.0, atol=1e-13)
    np.testing.assert_allclose(ip.gradients.sum(axis=1), 0.0, atol=1e-13)


def test_invalid_interpolation_selection_and_unresolvably_tiny_domains():
    p, grid = make_case()
    with pytest.raises(ValueError, match='interpolation'):
        ExplicitMPMSolver(p, grid, LinearElastic(100., 0.2),
                          dt=0.001, interpolation='unsupported')
    p.domain_size[:] = 1e-16
    with pytest.raises(ValueError, match='too small'):
        evaluate_gimp(p, grid)
