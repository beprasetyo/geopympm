"""Stage 6D: longitudinal standing wave in a 2D plane-strain elastic bar.

An *initial* displacement u(X,0)=A sin(pi X/L) and a mechanically
consistent small-strain initial stress/strain field excite the bar; particle
velocities start at zero. Only x velocities at the two ends and all y
velocities are constrained. The interior x velocities are NOT prescribed.

The exact linear continuum solution for a laterally constrained homogeneous
bar is u(X,t)=A sin(k X) cos(omega t), omega=k sqrt((lambda+2G)/rho).
This verification tests force-driven motion, including stress -> grid forces
-> acceleration -> G2P, but does not validate objective large-strain MPM.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, sqrt

import numpy as np

from geopympm.core.grid import BackgroundGrid
from geopympm.core.particles import ParticleSet
from geopympm.materials.linear_elastic import LinearElastic
from geopympm.solvers.explicit import ExplicitMPMSolver
from .benchmarks import BenchmarkResult


@dataclass
class WaveRun:
    """Wave benchmark result with scalar histories and selected profiles."""

    benchmark: BenchmarkResult
    parameters: dict[str, float | int]
    profiles: list[dict[str, float | int | str]]


def _norm_rel(numerical: np.ndarray, expected: np.ndarray, scale: np.ndarray) -> float:
    """L2 error normalized by *fixed initial-mode L2 scale*, not instantaneous mode."""
    return float(np.linalg.norm(numerical - expected) / np.linalg.norm(scale))


def _zero_crossings(history: list[dict]) -> list[float]:
    """Interpolate zero crossings of modal displacement in the sampled history."""
    zeros: list[float] = []
    for previous, current in zip(history[:-1], history[1:]):
        a = previous['modal_displacement_m']
        b = current['modal_displacement_m']
        if a * b < 0.0:
            f = -a / (b - a)
            zeros.append(float(previous['time'] + f * (current['time'] - previous['time'])))
    return zeros


def vibrating_bar(
    *, nx: int = 32, ny: int = 2, dt: float = 0.0005,
    t_end: float = 0.6, pic_fraction: float = 0.0,
    amplitude: float = 0.002,
) -> WaveRun:
    """Excite and track the first longitudinal standing-wave mode.

    Units: L=4 m, H=0.5 m, rho=1200 kg/m^3, E=120 kPa, nu=0.25.
    Four particles/cell; x boundaries fixed in x, all nodes fixed in y.
    Positive stress is tension; plane strain epsilon_yy=epsilon_zz=0.

    ``pic_fraction=0`` (FLIP) is the main 6D verification configuration.
    ``pic_fraction=1`` (PIC) is supplied to *measure numerical damping*;
    its output is expected to fail the tight FLIP wave-accuracy criteria.
    """
    if isinstance(nx, bool) or isinstance(ny, bool) or not isinstance(nx, int) or not isinstance(ny, int):
        raise ValueError('nx and ny must be positive integers.')
    if nx < 4 or ny < 1:
        raise ValueError('Use nx >= 4 and ny >= 1.')
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(t_end) or t_end <= 0:
        raise ValueError('dt and t_end must be positive finite numbers.')
    n_float = t_end / dt
    n_steps = int(round(n_float))
    if n_steps < 1 or not np.isclose(n_steps * dt, t_end, rtol=0, atol=1e-11):
        raise ValueError('t_end must be a positive integer multiple of dt.')
    if not np.isfinite(amplitude) or not 0.0 < amplitude < 0.01:
        raise ValueError('amplitude must be finite and in (0, 0.01) m for this small-strain benchmark.')
    if not np.isfinite(pic_fraction) or not 0.0 <= pic_fraction <= 1.0:
        raise ValueError('pic_fraction must be between 0 and 1.')

    L, H, rho = 4.0, 0.5, 1200.0
    E, nu = 120_000.0, 0.25
    shear = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    longitudinal_modulus = lam + 2.0 * shear
    wave_speed = sqrt(longitudinal_modulus / rho)
    k = pi / L
    omega = k * wave_speed
    period = 2.0 * pi / omega
    grid = BackgroundGrid((0.0, 0.0), (L / nx, H / ny), (nx, ny))

    # Undeformed particle material coordinates X. Four integration particles
    # within each Quad4 cell, arranged at 1/4 and 3/4 of each side.
    X = np.asarray([
        ((ix + sx) * L / nx, (iy + sy) * H / ny)
        for iy in range(ny) for ix in range(nx)
        for sy in (0.25, 0.75) for sx in (0.25, 0.75)
    ], dtype=np.float64)
    n = len(X)
    basis = np.sin(k * X[:, 0])
    stress_basis = np.cos(k * X[:, 0])
    u0 = amplitude * basis
    initial_strain = amplitude * k * stress_basis

    # Material coordinates are X, but the MPM configuration at t=0 is
    # x=X+u0(X). Initialize the *full compatible kinematic state* explicitly.
    initial_positions = X.copy()
    initial_positions[:, 0] += u0
    V0 = L * H / n  # representative reference volume at unit thickness
    p = ParticleSet(
        position=initial_positions,
        mass=np.full(n, rho * V0),
        volume=np.full(n, V0),
    )
    p.volume[:] = V0 * (1.0 + initial_strain)
    p.deformation_gradient[:, 0, 0] = 1.0 + initial_strain
    p.strain[:, 0] = initial_strain
    p.stress[:, 0] = longitudinal_modulus * initial_strain
    p.stress[:, 1] = lam * initial_strain
    p.stress[:, 2] = lam * initial_strain
    # ParticleSet.position0 is the initial *deformed* MPM configuration;
    # displacement against material coordinates X is evaluated separately.

    # Uniaxial strain wave: zero transverse velocity throughout the grid,
    # and zero axial velocity ONLY at the two end surfaces.
    grid.velocity_bc_mask[:, 1] = True
    xnodes = grid.node_coordinates[:, 0]
    grid.velocity_bc_mask[np.isclose(xnodes, 0.0) | np.isclose(xnodes, L), 0] = True
    solver = ExplicitMPMSolver(
        p, grid, LinearElastic(E, nu), dt=dt,
        pic_fraction=pic_fraction, gravity=(0.0, 0.0),
    )

    # Normalize errors by the fixed t=0 mode, not by a signal that vanishes
    # at quarter-period. This avoids singular error metrics near zero crossings.
    displacement_scale = amplitude * basis
    velocity_scale = amplitude * omega * basis
    stress_scale = longitudinal_modulus * amplitude * k * stress_basis
    initial_elastic_energy = float(
        0.5 * longitudinal_modulus * V0 * np.dot(initial_strain, initial_strain)
    )

    # Save a few snapshots for external 2D/1D plotting, without VTK dependency.
    snapshots = {
        0: 'initial',
        min(n_steps, int(round(0.25 * period / dt))): 'quarter_period',
        min(n_steps, int(round(0.50 * period / dt))): 'half_period',
        n_steps: 'final',
    }
    profiles: list[dict[str, float | int | str]] = []
    history: list[dict[str, float | int]] = []
    max_internal_force = 0.0
    min_jacobian = float('inf')

    def record(step: int, time: float, kinetic: float) -> None:
        expected_u = amplitude * np.cos(omega * time) * basis
        expected_v = -amplitude * omega * np.sin(omega * time) * basis
        expected_sigma = longitudinal_modulus * amplitude * k * np.cos(omega * time) * stress_basis
        actual_u = p.position[:, 0] - X[:, 0]
        elastic_energy = float(0.5 * longitudinal_modulus * np.dot(p.volume0, p.strain[:, 0] ** 2))
        total_energy = kinetic + elastic_energy
        history.append(dict(
            step=step, time=float(time),
            modal_displacement_m=float(np.dot(actual_u, basis) / np.dot(basis, basis)),
            modal_displacement_reference_m=float(amplitude * np.cos(omega * time)),
            modal_velocity_m_s=float(np.dot(p.velocity[:, 0], basis) / np.dot(basis, basis)),
            modal_velocity_reference_m_s=float(-amplitude * omega * np.sin(omega * time)),
            displacement_relative_l2=_norm_rel(actual_u, expected_u, displacement_scale),
            velocity_relative_l2=_norm_rel(p.velocity[:, 0], expected_v, velocity_scale),
            stress_relative_l2=_norm_rel(p.stress[:, 0], expected_sigma, stress_scale),
            kinetic_energy_j=float(kinetic),
            elastic_energy_j=elastic_energy,
            total_energy_j=total_energy,
            energy_relative_drift=float((total_energy - initial_elastic_energy) / initial_elastic_energy),
        ))
        if step in snapshots:
            snapshot_name = snapshots[step]
            for idx in range(n):
                profiles.append(dict(
                    snapshot=snapshot_name, step=step, time=float(time), particle=idx,
                    X_m=float(X[idx, 0]), Y_m=float(X[idx, 1]),
                    displacement_m=float(actual_u[idx]),
                    displacement_reference_m=float(expected_u[idx]),
                    sigma_xx_pa=float(p.stress[idx, 0]),
                    sigma_xx_reference_pa=float(expected_sigma[idx]),
                ))

    record(0, 0.0, 0.0)
    for step in range(1, n_steps + 1):
        diag = solver.step()
        max_internal_force = max(max_internal_force, float(np.max(np.abs(grid.force_internal[:, 0]))))
        min_jacobian = min(min_jacobian, diag.min_jacobian)
        record(step, diag.time, diag.kinetic_energy)

    crossing_times = _zero_crossings(history)
    period_numerical = (
        2.0 * (crossing_times[1] - crossing_times[0])
        if len(crossing_times) >= 2 else float('nan')
    )
    relative_period_error = (
        abs(period_numerical / period - 1.0)
        if np.isfinite(period_numerical) else float('inf')
    )
    thresholds = dict(
        displacement_relative_l2_max=0.03,
        velocity_relative_l2_max=0.04,
        stress_relative_l2_max=0.08,
        relative_period_error=0.015,
        energy_drift_abs_max=0.12,
        mass_relative_error=1e-12,
    )
    metrics = dict(
        displacement_relative_l2_max=max(r['displacement_relative_l2'] for r in history),
        velocity_relative_l2_max=max(r['velocity_relative_l2'] for r in history),
        stress_relative_l2_max=max(r['stress_relative_l2'] for r in history),
        relative_period_error=relative_period_error,
        period_numerical_s=float(period_numerical),
        period_analytical_s=float(period),
        wave_speed_analytical_m_s=float(wave_speed),
        energy_drift_abs_max=max(abs(r['energy_relative_drift']) for r in history),
        energy_ratio_final=float(history[-1]['total_energy_j'] / initial_elastic_energy),
        mass_relative_error=float(abs(np.sum(p.mass) - rho * L * H) / (rho * L * H)),
        max_internal_force_n=max_internal_force,
        min_jacobian=float(min_jacobian),
        particles=n,
        steps=n_steps,
        pic_fraction=float(pic_fraction),
    )
    passed = (
        all(np.isfinite(metrics[key]) and metrics[key] <= lim for key, lim in thresholds.items())
        and max_internal_force > 0
        and min_jacobian > 0
        and len(crossing_times) >= 2
    )
    benchmark = BenchmarkResult(
        name=f'6D vibrating elastic bar PIC fraction={pic_fraction:g}',
        passed=bool(passed), metrics=metrics, thresholds=thresholds,
        history=history,
        notes=(
            'Plane-strain longitudinal first mode, pre-stressed consistent initial state; '
            'only end x-velocity and all y-velocity constrained. '
            'FLIP=0 is verification baseline; PIC=1 expected to exhibit numerical damping. '
            'Energy estimate is small-strain modal diagnostic, not an exact discrete conservation proof.'
        ),
    )
    parameters = dict(
        L_m=L, H_m=H, nx=nx, ny=ny, particles_per_cell=4,
        rho_kg_m3=rho, E_pa=E, nu=nu, amplitude_m=amplitude,
        c_p_m_s=wave_speed, angular_frequency_rad_s=omega,
        analytic_period_s=period, dt_s=dt, t_end_s=t_end,
        pic_fraction=pic_fraction,
    )
    return WaveRun(benchmark=benchmark, parameters=parameters, profiles=profiles)


def grid_refinement() -> dict:
    """Joint grid/time refinement at fixed CFL fraction; NOT pure space order."""
    runs = [
        vibrating_bar(nx=16, dt=0.001),
        vibrating_bar(nx=32, dt=0.0005),
        vibrating_bar(nx=64, dt=0.00025),
    ]
    rows = [dict(nx=int(run.parameters['nx']), dt_s=float(run.parameters['dt_s']),
                 stress_relative_l2_max=float(run.benchmark.metrics['stress_relative_l2_max']),
                 displacement_relative_l2_max=float(run.benchmark.metrics['displacement_relative_l2_max']),
                 velocity_relative_l2_max=float(run.benchmark.metrics['velocity_relative_l2_max']))
            for run in runs]
    # Cell-crossing / particle integration errors can affect exact rates.
    monotone = all(rows[j+1]['stress_relative_l2_max'] < rows[j]['stress_relative_l2_max']
                   for j in range(len(rows)-1))
    return dict(passed=monotone, rows=rows,
                notes='Joint h,dt refinement only. Do not claim a spatial convergence order.')
