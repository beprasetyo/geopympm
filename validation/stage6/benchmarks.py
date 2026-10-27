"""GeoPyMPM Stage 6A–6C quantitative *verification* benchmarks.

All benchmarks use the UNMODIFIED Stage-5 ExplicitMPMSolver. They deliberately
separate mathematically prescribed reference solutions from MPM predictions.

6A: free rigid translation of a multi-particle cloud (including cell crossing).
6B: free fall compared to continuous and exact discretized Euler references.
6C: prescribed affine nodal velocity field: kinematic / constitutive patch.

6C is NOT a traction-based force-equilibrium patch test: all nodal velocities
are imposed. It checks interpolation, kinematics, deformation gradients,
volumes, particle advection, and incremental linear elastic stresses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver


@dataclass
class BenchmarkResult:
    name: str
    passed: bool
    metrics: dict[str, float | int]
    thresholds: dict[str, float | int]
    history: list[dict[str, float | int]] = field(default_factory=list)
    notes: str = ""


def _maxabs(array: Any) -> float:
    return float(np.max(np.abs(array)))


def _cloud(xs: list[float], ys: list[float], rho: float, volume: float,
           velocity: tuple[float, float]) -> ParticleSet:
    position = np.array([[x, y] for y in ys for x in xs], dtype=float)
    n = len(position)
    return ParticleSet(
        position=position,
        mass=np.full(n, rho * volume),
        volume=np.full(n, volume),
        velocity=np.tile(np.asarray(velocity), (n, 1)),
    )


def rigid_translation(*, n_steps: int = 50, dt: float = 0.01,
                      pic_fraction: float = 1.0) -> BenchmarkResult:
    """6A: homogeneous constant velocity must cause no strain or stress."""
    grid = BackgroundGrid(origin=(-0.5, -0.5), cell_size=(0.5, 0.5), n_cells=(10, 8))
    velocity = np.array([1.1, -0.35])
    p = _cloud([0.15, 0.42, 0.71, 1.06, 1.38, 1.74],
               [0.40, 0.73, 1.07, 1.39], rho=1800.0, volume=0.1,
               velocity=(float(velocity[0]), float(velocity[1])))
    solver = ExplicitMPMSolver(p, grid, LinearElastic(3.0e4, 0.25), dt=dt,
                               gravity=(0.0, 0.0), pic_fraction=pic_fraction)
    x0 = p.position.copy()
    total_mass = float(np.sum(p.mass))
    momentum0 = np.sum(p.mass[:, None] * p.velocity, axis=0)
    start_cells = np.floor((x0 - grid.origin) / grid.cell_size).astype(int)
    history = []
    for k in range(1, n_steps + 1):
        solver.step()
        t = k * dt
        expected_x = x0 + t * velocity
        actual_momentum = np.sum(p.mass[:, None] * p.velocity, axis=0)
        history.append(dict(
            step=k, time=float(t),
            position_error=_maxabs(p.position - expected_x),
            velocity_error=_maxabs(p.velocity - velocity),
            stress_abs_max=_maxabs(p.stress),
            deformation_error=_maxabs(p.deformation_gradient - np.eye(2)),
            mass_error=abs(float(np.sum(p.mass)) - total_mass),
            momentum_error=_maxabs(actual_momentum - momentum0),
        ))
    end_cells = np.floor((p.position - grid.origin) / grid.cell_size).astype(int)
    crossing_count = int(np.count_nonzero(np.any(start_cells != end_cells, axis=1)))
    thresholds = dict(position_error=1e-10, velocity_error=1e-10,
                      stress_abs_max=1e-6, deformation_error=1e-10,
                      mass_error=1e-9, momentum_error=1e-8)
    metrics = {key: max(float(row[key]) for row in history) for key in thresholds}
    metrics['crossing_particles'] = crossing_count
    metrics['particles'] = p.n_particles
    success = crossing_count > 0 and all(metrics[k] <= value for k, value in thresholds.items())
    return BenchmarkResult(
        name="6A rigid translation", passed=success,
        metrics=metrics, thresholds=thresholds, history=history,
        notes=f"{n_steps} steps, PIC fraction={pic_fraction:g}; crossing_particles must be >0.",
    )


def free_fall(*, dt: float = 0.01, t_end: float = 0.2,
              pic_fraction: float = 1.0) -> BenchmarkResult:
    """6B: free fall; compare both continuous and solver's Euler position rule.

    The Stage-5 G2P position rule uses the UPDATED velocity, so under constant
    acceleration a, x_k=x_0 + v_0*k*dt + a*dt^2*k*(k+1)/2.
    Its continuous-position error is a*T*dt/2, first order in dt.
    """
    steps_f = t_end / dt
    n_steps = int(round(steps_f))
    if n_steps < 1 or not np.isclose(n_steps * dt, t_end, atol=1e-12, rtol=0.0):
        raise ValueError("t_end must be a positive integer multiple of dt.")
    grid = BackgroundGrid((-1.0, -3.0), (1.0, 1.0), (6, 9))
    v0 = np.array([0.20, 0.25], dtype=float)
    g = np.array([0.0, -9.81], dtype=float)
    p = _cloud([0.45, 1.35, 2.25], [2.1, 2.6, 3.1],
               rho=1800.0, volume=0.1, velocity=(v0[0], v0[1]))
    solver = ExplicitMPMSolver(p, grid, LinearElastic(3e4, 0.25), dt=dt,
                               pic_fraction=pic_fraction, gravity=tuple(g))
    x0 = p.position.copy()
    history = []
    for k in range(1, n_steps + 1):
        solver.step()
        t = k * dt
        ref_v = v0 + g * t
        exact_x = x0 + v0 * t + 0.5 * g * t**2
        discrete_x = x0 + v0 * t + 0.5 * g * dt**2 * k * (k + 1)
        history.append(dict(
            step=k, time=float(t),
            velocity_error=_maxabs(p.velocity - ref_v),
            euler_position_error=_maxabs(p.position - discrete_x),
            continuum_position_error=_maxabs(p.position - exact_x),
            expected_continuum_position_error=float(0.5 * abs(g[1]) * t * dt),
            strain_abs_max=_maxabs(p.strain),
            stress_abs_max=_maxabs(p.stress),
        ))
    thresholds = dict(velocity_error=1e-10, euler_position_error=1e-10,
                      strain_abs_max=1e-10, stress_abs_max=1e-6)
    metrics = {key: max(float(row[key]) for row in history) for key in thresholds}
    metrics['continuum_position_error_end'] = history[-1]['continuum_position_error']
    metrics['expected_continuum_error_end'] = history[-1]['expected_continuum_position_error']
    metrics['dt'] = float(dt)
    metrics['n_steps'] = n_steps
    success = all(metrics[k] <= limit for k, limit in thresholds.items())
    # Check that the Euler error behaves as predicted instead of asserting
    # the continuous x(t) must match an inherently first-order integrator.
    success &= abs(metrics['continuum_position_error_end'] -
                   metrics['expected_continuum_error_end']) < 1e-10
    return BenchmarkResult(
        name=f"6B free fall dt={dt:g}", passed=bool(success),
        metrics=metrics, thresholds=thresholds, history=history,
        notes="Updated-velocity Euler advection has O(dt) position error vs continuum.",
    )


def free_fall_refinement(dt_values=(0.02, 0.01, 0.005), t_end=0.2) -> dict:
    """Check first-order reduction in free-fall continuous-position error."""
    results = [free_fall(dt=float(dt), t_end=t_end) for dt in dt_values]
    errors = [float(r.metrics['continuum_position_error_end']) for r in results]
    # Adjacent dt must decrease by factor two for the advertised order.
    rates = [float(np.log(errors[i] / errors[i+1]) /
                   np.log(float(dt_values[i]) / float(dt_values[i+1])))
             for i in range(len(errors)-1)]
    success = all(r.passed for r in results) and all(abs(rate - 1.0) < 1e-6 for rate in rates)
    return dict(passed=success, dt_values=[float(x) for x in dt_values],
                errors=errors, observed_orders=rates)


def affine_elastic(*, n_steps: int = 5, dt: float = 0.001) -> BenchmarkResult:
    """6C: exact affine nodal BC test, NOT a traction/force equilibrium test.

    Every nodal velocity component is prescribed v(x)=A x + b. For a
    structured Quad4 grid, the affine v and L=A are reproduced exactly at
    each particle. We compare incremental stress, strain, F, V and motion
    to references independently constructed from A, E, nu and explicit Euler.
    """
    grid = BackgroundGrid(origin=(0.0, 0.0), cell_size=(1.0, 1.0), n_cells=(4, 3))
    initial_positions = np.array([
        [ix + sx, iy + sy]
        for iy in range(grid.n_cells[1])
        for ix in range(grid.n_cells[0])
        for sx in (0.25, 0.75)
        for sy in (0.25, 0.75)
    ], dtype=float)
    n = initial_positions.shape[0]
    rho = 1800.0
    volume0 = np.full(n, 0.25)
    p = ParticleSet(position=initial_positions, mass=rho * volume0,
                    volume=volume0)
    # Inverse second: L = dv/dx = A. Includes normal strain, engineering
    # shear, a small spin component and rigid translation.
    A = np.array([[0.20, 0.15], [0.05, -0.30]], dtype=float)
    b = np.array([0.04, -0.03], dtype=float)
    grid.velocity_bc_mask[:, :] = True
    grid.velocity_bc_value[:, :] = grid.node_coordinates @ A.T + b
    E = 20e6
    nu = 0.25
    mat = LinearElastic(E, nu)
    lam = E * nu / ((1+nu)*(1-2*nu))
    G = E / (2*(1+nu))
    # Independent small-increment plane strain constitutive reference.
    sigma_step = dt * np.array([
        (lam + 2*G)*A[0, 0] + lam*A[1, 1],
        lam*A[0, 0] + (lam + 2*G)*A[1, 1],
        lam*(A[0, 0] + A[1, 1]),
        G*(A[0, 1] + A[1, 0]), 0.0, 0.0,
    ])
    strain_step = dt * np.array([A[0,0], A[1,1], 0.0, A[0,1]+A[1,0], 0.0, 0.0])
    M = np.eye(2) + dt * A
    solver = ExplicitMPMSolver(p, grid, mat, dt=dt, gravity=(0, 0), pic_fraction=1.0)
    xref = initial_positions.copy()
    Fref = np.eye(2)
    history = []
    for k in range(1, n_steps + 1):
        solver.step()
        # Reference discrete deformation map under constant imposed affine v.
        xref = xref @ M.T + dt * b
        Fref = M @ Fref
        ref_volume = volume0 * np.linalg.det(Fref)
        history.append(dict(
            step=k, time=float(k*dt),
            stress_error=_maxabs(p.stress - k*sigma_step),
            strain_error=_maxabs(p.strain - k*strain_step),
            deformation_error=_maxabs(p.deformation_gradient - Fref),
            volume_error=_maxabs(p.volume - ref_volume),
            position_error=_maxabs(p.position - xref),
        ))
    thresholds = dict(stress_error=1e-6, strain_error=1e-11,
                      deformation_error=1e-11, volume_error=1e-11,
                      position_error=1e-11)
    metrics = {key: max(float(row[key]) for row in history) for key in thresholds}
    metrics['sigma_zz_expected_pa'] = float(n_steps*sigma_step[2])
    metrics['sigma_xy_expected_pa'] = float(n_steps*sigma_step[3])
    metrics['particles'] = n
    success = all(metrics[k] <= limit for k, limit in thresholds.items())
    return BenchmarkResult(
        name="6C affine elastic patch (kinematic)", passed=success,
        metrics=metrics, thresholds=thresholds, history=history,
        notes="All nodal velocities are prescribed; no traction equilibrium or force-response verified.",
    )
