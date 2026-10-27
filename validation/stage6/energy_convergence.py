"""Stage 6F: separated energy, space, time and CFL checks of Stage-5 MPM.

This module is *verification only*: it does not change the solver.  It uses
Stage-6D's stress-driven plane-strain standing-wave benchmark (not kinematically
prescribed interior nodal motion). Boundary velocities are zero so boundary
work is zero. No external force, contact, damping or plasticity is present.

Caveats: energy is K + 0.5 * sum(V0 * sigma:engineering_epsilon) for this
small-strain linear-elastic benchmark, NOT an exact discrete MPM Hamiltonian;
there is no external work or gravitational potential in this particular case.
The temporal fine-step reference is numerical, not an exact analytical solution.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt

import numpy as np

from geopympm.core.grid import BackgroundGrid
from geopympm.core.particles import ParticleSet
from geopympm.materials.linear_elastic import LinearElastic
from geopympm.solvers.explicit import ExplicitMPMSolver
from .elastic_wave import WaveRun, vibrating_bar


@dataclass
class EnergySuite:
    passed: bool
    metrics: dict
    histories: dict[str, list[dict]]
    notes: str


@dataclass
class ConvergenceSuite:
    passed: bool
    rows: list[dict]
    orders: list[float]
    notes: str


def particle_energy(particles: ParticleSet) -> dict[str, float]:
    """K and small-strain U for an isotropic, *purely elastic* particle state.

    Voigt: sigma = [xx,yy,zz,xy,yz,xz], engineering gamma for strain.
    sigma:epsilon = sigma_ij epsilon_ij + sigma_xy gamma_xy + ... .
    Use V0 (reference volume), NOT deformed volume, for this small-strain
    energy measure. This is not valid for plastic or finite-rotation histories.
    """
    if not isinstance(particles, ParticleSet):
        raise TypeError('particles must be ParticleSet.')
    stress, strain = particles.stress, particles.strain
    if (not np.all(np.isfinite(stress)) or not np.all(np.isfinite(strain))
            or not np.all(np.isfinite(particles.mass))
            or not np.all(np.isfinite(particles.velocity))
            or not np.all(np.isfinite(particles.volume0))):
        raise ValueError('Particle state contains nonfinite energy inputs.')
    kinetic = float(0.5 * np.einsum('p,pd,pd->', particles.mass,
                                    particles.velocity, particles.velocity))
    elastic = float(0.5 * np.einsum('p,pi,pi->', particles.volume0, stress, strain))
    return {'kinetic_energy_j': kinetic, 'elastic_energy_j': elastic,
            'total_energy_j': kinetic + elastic}


def energy_comparison(*, nx: int = 32, dt: float = 0.0005,
                      t_end: float = 0.6, alphas=(0.0, 0.05, 1.0)) -> EnergySuite:
    """Compare undamped-wave energy history for FLIP, blended and PIC transfer.

    PASS is gated by FLIP's near-conservation and finite energy histories.
    PIC damping is *reported* rather than an algorithm failure condition.
    """
    if tuple(alphas) != (0.0, 0.05, 1.0):
        raise ValueError('Stage-6F comparison uses alphas=(0,0.05,1).')
    runs = {float(alpha): vibrating_bar(nx=nx, dt=dt, t_end=t_end,
                                         pic_fraction=alpha) for alpha in alphas}
    labels = {0.0: 'FLIP', 0.05: 'BLEND_0p05', 1.0: 'PIC'}
    histories: dict[str, list[dict]] = {}
    metrics: dict[str, dict] = {}
    for alpha, wave in runs.items():
        name = labels[alpha]
        records = wave.benchmark.history
        e0 = float(records[0]['total_energy_j'])
        if not np.isfinite(e0) or e0 <= 0:
            raise ValueError('Initial energy must be finite and positive.')
        histories[name] = [dict(step=int(r['step']), time_s=float(r['time']),
                                kinetic_energy_j=float(r['kinetic_energy_j']),
                                elastic_energy_j=float(r['elastic_energy_j']),
                                total_energy_j=float(r['total_energy_j']),
                                relative_energy_residual=float(r['energy_relative_drift']))
                           for r in records]
        relative_residual = np.array([r['relative_energy_residual'] for r in histories[name]])
        kinetic = np.array([r['kinetic_energy_j'] for r in histories[name]])
        elastic = np.array([r['elastic_energy_j'] for r in histories[name]])
        metrics[name] = dict(alpha=alpha, initial_energy_j=e0,
                             final_energy_ratio=float(records[-1]['total_energy_j']/e0),
                             max_abs_energy_residual=float(np.max(np.abs(relative_residual))),
                             max_kinetic_energy_j=float(np.max(kinetic)),
                             min_kinetic_energy_j=float(np.min(kinetic)),
                             min_elastic_energy_j=float(np.min(elastic)),
                             n_steps=int(len(records)-1))
    flip = metrics['FLIP']
    finite = all(np.isfinite(m[k]) for m in metrics.values() for k in (
        'initial_energy_j','final_energy_ratio','max_abs_energy_residual',
        'max_kinetic_energy_j','min_kinetic_energy_j','min_elastic_energy_j'))
    passed = bool(finite and flip['max_abs_energy_residual'] < 0.02
                  and abs(flip['final_energy_ratio']-1.0) < 0.02
                  and all(m['min_kinetic_energy_j'] >= -1e-12 and
                          m['min_elastic_energy_j'] >= -1e-12 for m in metrics.values()))
    return EnergySuite(passed, metrics, histories,
        'Zero-work constrained elastic standing wave; K + small-strain elastic U. '
        'FLIP is the conservation-focused baseline. PIC/BLEND dissipation is '
        'an observed algorithm property, not physical damping. No exact discrete energy invariant is claimed.')


def spatial_convergence(*, nx_values=(16,32,64), dt=0.00025, t_end=0.3) -> ConvergenceSuite:
    """Change h ONLY. Time step, physical size and 4 particles/cell fixed.

    Stress error is against Stage-6D analytical standing-wave solution.  The
    reference is continuum small-strain plane-strain elasticity, not an FEM run.
    """
    if tuple(nx_values) != (16, 32, 64):
        raise ValueError('Stage-6F spatial comparison uses nx=(16,32,64).')
    rows = []
    for nx in nx_values:
        wave = vibrating_bar(nx=nx, dt=dt, t_end=t_end, pic_fraction=0.0)
        m = wave.benchmark.metrics
        rows.append(dict(nx=nx, dx_m=float(wave.parameters['L_m']/nx),
                         dt_s=dt, n_steps=m['steps'],
                         max_stress_l2_error=float(m['stress_relative_l2_max']),
                         max_displacement_l2_error=float(m['displacement_relative_l2_max']),
                         max_velocity_l2_error=float(m['velocity_relative_l2_max']),
                         energy_drift_abs_max=float(m['energy_drift_abs_max'])))
    orders = [float(log(rows[i]['max_stress_l2_error']/rows[i+1]['max_stress_l2_error'])
                    / log(rows[i]['dx_m']/rows[i+1]['dx_m'])) for i in range(2)]
    passed = bool(all(np.isfinite(v) for v in orders)
                  and all(rows[i+1]['max_stress_l2_error'] < rows[i]['max_stress_l2_error']
                          for i in range(2)))
    return ConvergenceSuite(passed, rows, orders,
        'Space refinement nx=16/32/64 at identical dt=0.00025s, ny=2, four particles/cell; '
        'observed rates apply only to this small-amplitude standing-wave benchmark. '
        'A fixed dt does not remove all temporal error; spatial dominance is checked empirically.')


def _final_field(wave: WaveRun, key: str) -> np.ndarray:
    records = [p for p in wave.profiles if p['snapshot'] == 'final']
    expected_n = wave.benchmark.metrics['particles']
    if len(records) != expected_n:
        raise ValueError('Expected exactly one final profile sample per particle.')
    return np.array([float(row[key]) for row in records], dtype=np.float64)


def temporal_convergence(*, nx=32, dt_values=(0.002,0.001,0.0005,0.00025),
                         dt_reference=0.000125, t_end=0.3) -> ConvergenceSuite:
    """Change dt ONLY at fixed grid/particles and compare final-time fields.

    Both an independent fine numerical reference and consecutive-pair errors
    are reported. Consecutive-difference ratios provide Richardson-like
    self-convergence without conflating finite-reference errors with order.
    """
    if tuple(dt_values) != (0.002,0.001,0.0005,0.00025) or dt_reference != 0.000125:
        raise ValueError('Stage-6F temporal comparison uses fixed dt list and reference.')
    runs = [vibrating_bar(nx=nx,dt=dt,t_end=t_end,pic_fraction=0.) for dt in dt_values]
    fine = vibrating_bar(nx=nx,dt=dt_reference,t_end=t_end,pic_fraction=0.)
    ref_u = _final_field(fine, 'displacement_m')
    ref_s = _final_field(fine, 'sigma_xx_pa')
    fields_u, fields_s, rows = [], [], []
    for dt, wave in zip(dt_values, runs):
        u = _final_field(wave, 'displacement_m')
        s = _final_field(wave, 'sigma_xx_pa')
        fields_u.append(u)
        fields_s.append(s)
        rows.append(dict(nx=nx, dx_m=float(wave.parameters['L_m']/nx),dt_s=dt,
                         reference_dt_s=dt_reference,
                         n_steps=int(wave.benchmark.metrics['steps']),
                         final_displacement_relative_reference_error=float(
                             np.linalg.norm(u-ref_u)/np.linalg.norm(ref_u)),
                         final_stress_relative_reference_error=float(
                             np.linalg.norm(s-ref_s)/np.linalg.norm(ref_s)),
                         max_stress_l2_vs_continuum=float(
                             wave.benchmark.metrics['stress_relative_l2_max'])))
    # Norms of pairwise differences, not analytical errors. For first-order
    # time stepping halving dt should approximately halve these differences.
    differences = [float(np.linalg.norm(fields_s[i]-fields_s[i+1])) for i in range(3)]
    orders = [float(log(differences[i]/differences[i+1])/log(2.)) for i in range(2)]
    for i, row in enumerate(rows):
        row['self_difference_stress_pa_norm'] = differences[i] if i < len(differences) else None
    passed = bool(all(np.isfinite(x) and 0.65 < x < 1.35 for x in orders)
                  and all(rows[i+1]['final_stress_relative_reference_error'] <
                          rows[i]['final_stress_relative_reference_error'] for i in range(3))
                  and all(rows[i+1]['final_displacement_relative_reference_error'] <
                          rows[i]['final_displacement_relative_reference_error'] for i in range(3)))
    return ConvergenceSuite(passed, rows, orders,
        'Time refinement dt=.002/.001/.0005/.00025 s at fixed nx=32. Fine numerical '
        'reference dt=.000125s at same grid. Order inferred independently from '
        'successive stress-profile differences at identical final time.')


def cfl_guard_check() -> dict:
    """Test Stage-5 CFL *input guard*, not absolute numerical stability.

    The trial uses a free particle initially at rest with zero stress, so the
    screening limit can be calculated directly from elastic wave speed.
    """
    L,H,nx,ny = 4.0,0.5,32,2
    E,nu,rho = 120_000.0,0.25,1200.0
    lam = E*nu/((1+nu)*(1-2*nu))
    G = E/(2*(1+nu))
    cp = sqrt((lam+2*G)/rho)
    expected_limit = 0.25 * min(L/nx,H/ny) / cp
    def make_solver(dt: float):
        grid = BackgroundGrid((0.,0.),(L/nx,H/ny),(nx,ny))
        volume = (L/nx)*(H/ny)
        particles = ParticleSet(position=[[1.25,0.125]], mass=[rho*volume], volume=[volume])
        solver = ExplicitMPMSolver(particles,grid,LinearElastic(E,nu),dt=dt,
                                   pic_fraction=0.,cfl=.25)
        return solver, particles
    low, _ = make_solver(0.9*expected_limit)
    reported_limit = low.stable_dt_estimate()
    accepted = low.step()
    high, p = make_solver(1.05*expected_limit)
    x_before=p.position.copy(); stress_before=p.stress.copy()
    rejected = False
    try:
        high.step()
    except ValueError as exc:
        rejected = 'exceeds current Stage-5 CFL estimate' in str(exc)
    guard_preserves_state = bool(np.array_equal(x_before,p.position) and
                                 np.array_equal(stress_before,p.stress)
                                 and high.step_count == 0 and high.time == 0)
    passed = bool(rejected and guard_preserves_state and accepted.step==1
                  and np.isclose(reported_limit,expected_limit,rtol=1e-13))
    return dict(passed=passed, wave_speed_m_s=cp,
                dt_limit_s=float(reported_limit), accepted_dt_s=low.dt,
                rejected_dt_s=high.dt, rejected=rejected,
                rejected_step_state_intact=guard_preserves_state,
                notes='Tests only Stage-5 heuristic CFL rejection and rollback; does not prove stability below its threshold.')


def full_stage6f() -> dict:
    energy=energy_comparison()
    space=spatial_convergence()
    time=temporal_convergence()
    cfl=cfl_guard_check()
    passed=bool(energy.passed and space.passed and time.passed and cfl['passed'])
    return dict(passed=passed,energy=energy,spatial=space,temporal=time,cfl=cfl)
