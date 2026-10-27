"""Stage 5: test the integrated 2D elastic explicit MPM cycle."""

import numpy as np
import pytest

from geopympm.core import ParticleSet, BackgroundGrid
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver, StepDiagnostics


def make_free_case(*, position=(2.5, 2.5), velocity=(0.0, 0.0),
                   gravity=(0.0, 0.0), dt=0.01, E=200.0):
    particles = ParticleSet(
        position=[position], mass=[1800.0], volume=[1.0], velocity=[velocity]
    )
    grid = BackgroundGrid(origin=(0.0, 0.0), cell_size=(1.0, 1.0), n_cells=(5, 5))
    solver = ExplicitMPMSolver(
        particles, grid, LinearElastic(E, .25),
        dt=dt, cfl=.25, pic_fraction=1.0, gravity=gravity,
    )
    return particles, grid, solver


def test_stable_dt_formula_wave_speed_plus_velocity():
    p, g, solver = make_free_case(velocity=(3, 4), E=1000)
    rho = p.mass[0] / p.volume[0]
    mu = 1000 / (2 * (1 + .25))
    lam = 1000 * .25 / ((1 + .25) * (1 - 2 * .25))
    expected = .25 / (np.sqrt((lam + 2 * mu) / rho) + 5.0)
    np.testing.assert_allclose(solver.stable_dt_estimate(), expected)


def test_run_zero_steps_does_not_mutate_state():
    p, g, solver = make_free_case()
    before = (p.position.copy(), g.mass.copy(), p.stress.copy())
    assert solver.run(0) == []
    np.testing.assert_array_equal(p.position, before[0])
    np.testing.assert_array_equal(g.mass, before[1])
    np.testing.assert_array_equal(p.stress, before[2])
    assert solver.step_count == 0 and solver.time == 0


def test_uniform_velocity_no_strain_and_constant_momentum_many_steps():
    p, g, solver = make_free_case(position=(2.2, 2.4), velocity=(1.0, -0.5))
    results = solver.run(20)
    assert len(results) == 20 and all(isinstance(r, StepDiagnostics) for r in results)
    np.testing.assert_allclose(p.position[0], [2.4, 2.3], rtol=0, atol=1e-12)
    np.testing.assert_allclose(p.velocity[0], [1.0, -.5], atol=1e-12)
    np.testing.assert_allclose(p.stress[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(p.deformation_gradient[0], np.eye(2), atol=1e-12)
    np.testing.assert_allclose(p.volume, p.volume0, atol=1e-12)
    np.testing.assert_allclose(results[-1].momentum, (1800, -900), atol=1e-10)
    assert results[-1].active_nodes == 4
    assert results[-1].time == pytest.approx(0.20)


def test_gravity_only_free_fall_predictable_euler_advection():
    p, g, solver = make_free_case(gravity=(0.0, -1.0))
    results = solver.run(10)
    np.testing.assert_allclose(p.velocity[0], [0.0, -.1], atol=1e-12)
    # Forward Euler with updated velocity: sum_{k=1}^{10} -g*k*dt^2.
    np.testing.assert_allclose(p.position[0], [2.5, 2.4945], atol=1e-12)
    np.testing.assert_allclose(p.stress, 0.0, atol=1e-10)
    np.testing.assert_allclose(p.strain, 0.0, atol=1e-12)
    np.testing.assert_allclose(p.volume, p.volume0, atol=1e-12)
    assert results[-1].kinetic_energy == pytest.approx(9.0)
    assert solver.step_count == 10 and solver.time == pytest.approx(.1)


def test_cell_crossing_re_evaluates_interpolation():
    p = ParticleSet([[.95, .5]], mass=[1.0], volume=[.2], velocity=[[1., 0.]])
    g = BackgroundGrid((0., 0.), (1., 1.), (4, 2))
    solver = ExplicitMPMSolver(p, g, LinearElastic(2., .2), dt=.08)
    solver.step()
    assert p.position[0, 0] == pytest.approx(1.03)
    solver.step()  # fails if Stage-2 interpolation is improperly cached
    assert p.position[0, 0] == pytest.approx(1.11)
    np.testing.assert_allclose(p.stress, 0.0, atol=1e-12)


def test_grid_boundary_conditions_persist_and_stress_feeds_next_step():
    p = ParticleSet([[.5, .5]], mass=[450.], volume=[.25])
    g = BackgroundGrid((0., 0.), (1., 1.), (1, 1))
    g.velocity_bc_mask[[0, 1], 1] = True
    mat = LinearElastic(20000., .25)
    solver = ExplicitMPMSolver(p, g, mat, dt=.001, gravity=(0., -9.81))
    solver.step()
    assert p.stress[0, 1] < 0
    first_stress = p.stress.copy()
    assert np.all(g.velocity[[0, 1], 1] == 0.)
    solver.step()
    assert np.any(np.abs(g.force_internal) > 0.)
    assert np.all(g.velocity_bc_mask[[0, 1], 1])
    assert np.all(g.velocity[[0, 1], 1] == 0.)
    assert not np.array_equal(p.stress, first_stress)
    assert solver.step_count == 2


def test_multimaterial_mapping_runs():
    p = ParticleSet(position=[[1.25, 1.25], [3.25, 3.25]],
                    mass=[20, 30], volume=[.2, .3], material_id=[0, 3])
    g = BackgroundGrid((0, 0), (1, 1), (5, 5))
    materials = {0: LinearElastic(100, .2), 3: LinearElastic(200, .3)}
    solver = ExplicitMPMSolver(p, g, materials, dt=.001)
    assert solver.step().total_mass == pytest.approx(50.0)
    np.testing.assert_allclose(p.stress, 0.0)


def test_reject_missing_material_at_construction():
    p, g, _ = make_free_case()
    p.material_id[:] = 7
    with pytest.raises(ValueError, match='material_id=7'):
        ExplicitMPMSolver(p, g, {0: LinearElastic(100, .2)}, dt=.001)


def test_reject_unsafe_dt_without_changing_state():
    p, g, solver = make_free_case(E=1e7, dt=0.5)
    originals = (p.position.copy(), p.stress.copy(), g.mass.copy())
    with pytest.raises(ValueError, match='CFL estimate'):
        solver.step()
    np.testing.assert_array_equal(p.position, originals[0])
    np.testing.assert_array_equal(p.stress, originals[1])
    np.testing.assert_array_equal(g.mass, originals[2])
    assert solver.time == 0


def test_failed_g2p_rolls_back_particle_and_grid_states():
    p = ParticleSet([[.95, .5]], mass=[1.], volume=[.1], velocity=[[10., 0.]])
    g = BackgroundGrid((0., 0.), (1., 1.), (1, 1))
    g.mass[:] = 42.0  # preexisting state must also survive failed step
    before_p = {k: getattr(p, k).copy() for k in ExplicitMPMSolver._PARTICLE_MUTABLE}
    before_g = {k: getattr(g, k).copy() for k in ExplicitMPMSolver._GRID_MUTABLE}
    mat = LinearElastic(1., .25)
    solver = ExplicitMPMSolver(p, g, mat, dt=.02)
    with pytest.raises(ValueError, match='outside background grid'):
        solver.step()
    for k, array in before_p.items():
        np.testing.assert_array_equal(getattr(p, k), array)
    for k, array in before_g.items():
        np.testing.assert_array_equal(getattr(g, k), array)
    assert solver.step_count == 0 and solver.time == 0


@pytest.mark.parametrize('dt', [0, -.001, float('nan'), float('inf'), True])
def test_reject_bad_dt(dt):
    p, g, _ = make_free_case()
    with pytest.raises(ValueError, match='dt'):
        ExplicitMPMSolver(p, g, LinearElastic(100, .25), dt=dt)


@pytest.mark.parametrize('fraction', [-0.1, 1.1, float('nan'), True])
def test_reject_bad_fraction(fraction):
    p, g, _ = make_free_case()
    with pytest.raises(ValueError, match='pic_fraction'):
        ExplicitMPMSolver(p, g, LinearElastic(100, .25), dt=.001,
                          pic_fraction=fraction)


@pytest.mark.parametrize('cfl', [0., -.1, 1.1, True, float('nan')])
def test_reject_bad_cfl(cfl):
    p, g, _ = make_free_case()
    with pytest.raises(ValueError, match='cfl'):
        ExplicitMPMSolver(p, g, LinearElastic(100, .25), dt=.001, cfl=cfl)


def test_reject_bad_step_count():
    p, g, solver = make_free_case()
    for value in (1.0, True, '2'):
        with pytest.raises(TypeError, match='n_steps'):
            solver.run(value)
    with pytest.raises(ValueError, match='n_steps'):
        solver.run(-1)
    assert solver.step_count == 0


def test_external_force_array_is_copied_and_validated():
    p, g, _ = make_free_case()
    f = np.zeros((g.n_nodes, 2))
    solver = ExplicitMPMSolver(p, g, LinearElastic(100., .25),
                               dt=.001, nodal_external_force=f)
    f[0, 0] = 99.
    assert solver.nodal_external_force[0, 0] == 0.
    with pytest.raises(ValueError, match='nodal_external_force'):
        ExplicitMPMSolver(p, g, LinearElastic(100., .25), dt=.001,
                          nodal_external_force=np.ones((2, 2)))
