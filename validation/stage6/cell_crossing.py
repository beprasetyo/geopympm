"""Stage 6E: standard Quad4 MPM cell-crossing force-transfer verification.

Two complementary *controlled* tests are provided:

1. A single prestressed particle placed epsilon either side of an interior
   grid line demonstrates the non-continuity of dN/dx for point MPM.
2. A uniformly translating, uniformly prestressed rectangular block is
   advected using prescribed uniform nodal velocity. Its nodal internal force
   can be compared with exact integration of -sigma_xx*dN/dx over the moving
   continuum rectangle. Particle quadrature/gradient switching produces force
   sampling errors; increasing particles per cell should reduce them.

The imposed nodal velocity intentionally removes force-driven acceleration:
Stage 6E isolates the assembly/interpolation error. This is NOT a free-run
landslide simulation, nor a stability proof. Neither benchmark implements GIMP.

Positive tensile stress convention, 2D plane strain with unit thickness.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from geopympm.core.grid import BackgroundGrid
from geopympm.core.particles import ParticleSet
from geopympm.interpolation.mpm import evaluate_quad4
from geopympm.materials.linear_elastic import LinearElastic
from geopympm.solvers.explicit import ExplicitMPMSolver
from geopympm.solvers.p2g import particle_to_grid
from geopympm.solvers.grid_update import update_grid_momentum
from .benchmarks import BenchmarkResult


@dataclass
class CrossingRun:
    """Scalar metrics + nodal force time series for one particle resolution."""
    benchmark: BenchmarkResult
    parameters: dict[str, int | float]


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a strictly positive integer.')
    return int(value)


def _positive_float(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a finite positive scalar.')
    try:
        v = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite positive scalar.') from exc
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f'{name} must be a finite positive scalar.')
    return v


def _static_internal_force(x: float, *, volume: float, sigma_xx: float) -> float:
    """Read assembled x force of grid node (2,0), with a particle at (x,0.5)."""
    grid = BackgroundGrid((0.0, 0.0), (1.0, 1.0), (4, 1))
    p = ParticleSet(position=[[x, 0.5]], mass=[100.0], volume=[volume])
    p.stress[0, 0] = sigma_xx
    interpolation = evaluate_quad4(p, grid)
    particle_to_grid(p, grid, interpolation)
    update_grid_momentum(p, grid, interpolation, 0.0001)
    return float(grid.force_internal[grid.node_id(2, 0), 0])


def one_particle_probe(*, epsilon: float = 1e-7, sigma_xx: float = 120.0,
                       volume: float = 0.2) -> BenchmarkResult:
    """Compare the left/right limits of point-MPM nodal force at x=2.

    For the lower node (2,0), y=0.5, dx=1:
        lim(x -> 2-) f_x = -0.5 * V * sigma_xx
        lim(x -> 2+) f_x = +0.5 * V * sigma_xx.
    These *individual particle nodal contributions* are discontinuous; this
    is not itself proof that a smooth many-particle solution fails.
    """
    epsilon = _positive_float('epsilon', epsilon)
    volume = _positive_float('volume', volume)
    if not 1e-12 <= epsilon <= 1e-3:
        raise ValueError('epsilon must be between 1e-12 and 1e-3 m.')
    try:
        sigma_xx = float(sigma_xx)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('sigma_xx must be finite and nonzero.') from exc
    if not math.isfinite(sigma_xx) or sigma_xx == 0.0:
        raise ValueError('sigma_xx must be finite and nonzero.')

    left = _static_internal_force(2.0 - epsilon, volume=volume, sigma_xx=sigma_xx)
    right = _static_internal_force(2.0 + epsilon, volume=volume, sigma_xx=sigma_xx)
    exact_left = -0.5 * volume * sigma_xx
    exact_right = +0.5 * volume * sigma_xx
    error = max(abs(left - exact_left), abs(right - exact_right))
    jump = right - left
    exact_jump = volume * sigma_xx
    metrics = dict(
        left_force_n=left,
        right_force_n=right,
        signed_jump_n=jump,
        absolute_jump_n=abs(jump),
        expected_signed_jump_n=exact_jump,
        max_limit_error_n=error,
        jump_error_n=abs(jump - exact_jump),
    )
    atol = 1e-10 * max(1.0, abs(exact_jump))
    thresholds = dict(max_limit_error_n=atol, jump_error_n=atol)
    return BenchmarkResult(
        name='6E-1 single-particle gradient jump',
        passed=all(metrics[k] <= v for k, v in thresholds.items()),
        metrics=metrics, thresholds=thresholds,
        history=[dict(side='left', x_m=2.0-epsilon, force_n=left, analytical_n=exact_left),
                 dict(side='right', x_m=2.0+epsilon, force_n=right, analytical_n=exact_right)],
        notes='The discontinuity concerns one particle contribution to a grid node, not the continuum solution.',
    )


def exact_nodal_force_x(*, node_x: float, x_left: float, x_right: float,
                        h: float, thickness: float, sigma_xx: float,
                        transverse_weight: float = 0.5) -> float:
    """Analytical -integral sigma_xx*dN/dx dV for a uniform rectangle.

    For a node at x_i, its 1D linear hat gradient is +1/h in [x_i-h,x_i]
    and -1/h in [x_i,x_i+h]. Integrate a block occupying [left,right] and
    full transverse height. At a bottom/top node of a one-cell-high mesh,
    integrating N_y over y gives a factor H/2 (default transverse_weight=.5).
    """
    if h <= 0 or thickness <= 0 or x_left >= x_right:
        raise ValueError('Invalid geometry for analytical force.')
    overlap_left = max(0.0, min(x_right, node_x) - max(x_left, node_x-h))
    overlap_right = max(0.0, min(x_right, node_x+h) - max(x_left, node_x))
    return float(sigma_xx * thickness * transverse_weight / h * (overlap_right-overlap_left))


def moving_stressed_block(*, particles_per_cell: int = 4,
                          dt: float = 0.01, t_end: float = 1.5,
                          speed: float = 0.9,
                          sigma_xx: float = 120.0) -> CrossingRun:
    """Advect a prestressed 3 m by 1 m block and compare nodal force to an integral.

    The sample node is (5,0) in a grid of 8x1 cells (h=1 m). The left block
    edge starts at 1.22 m; right edge at 4.22 m. At each MPM step, sample
    internal force *at the pre-advection position* (t=(step-1)*dt) and compare
    against exact continuum integration of that same stress field and shape.

    All nodal velocities are prescribed (speed,0) to enforce pure advection;
    there is no acceleration driven by the internal forces. The comparison
    diagnoses particle-quadrature and cell-crossing force fluctuations.
    """
    ppc = _positive_int('particles_per_cell', particles_per_cell)
    if ppc > 64:
        raise ValueError('particles_per_cell is limited to 64 for benchmark runtime.')
    dt = _positive_float('dt', dt)
    t_end = _positive_float('t_end', t_end)
    speed = _positive_float('speed', speed)
    sigma_xx = _positive_float('sigma_xx', sigma_xx)
    steps = round(t_end / dt)
    if steps < 2 or not np.isclose(steps*dt, t_end, rtol=0, atol=1e-11):
        raise ValueError('t_end must be an integer multiple of dt and include >=2 steps.')
    left0, length, H = 1.22, 3.0, 1.0
    if left0 + length + speed*t_end >= 8.0:
        raise ValueError('The block must remain inside the eight-cell background grid.')
    n_x = 3*ppc
    xs = left0 + (np.arange(n_x, dtype=np.float64)+0.5) / ppc
    ys = [0.25, 0.75]
    xy = np.array([(x,y) for y in ys for x in xs], dtype=np.float64)
    n = len(xy)
    volume0 = H * length / n
    density = 1200.0
    particles = ParticleSet(
        xy, np.full(n, density*volume0), np.full(n, volume0),
        velocity=np.tile(np.array([speed,0.0]), (n,1)),
    )
    particles.stress[:, 0] = sigma_xx
    grid = BackgroundGrid((0.0,0.0), (1.0,1.0), (8,1))
    grid.velocity_bc_mask[:,:] = True
    grid.velocity_bc_value[:,0] = speed
    material = LinearElastic(120000.0, 0.25)
    solver = ExplicitMPMSolver(particles, grid, material, dt=dt, pic_fraction=0.0)
    sample_node = grid.node_id(5,0)
    force_scale = sigma_xx * H * 0.5
    start_positions = particles.position.copy()
    records = []
    crossings = set()
    previous_x = start_positions[:,0].copy()
    for step in range(1, steps+1):
        diagnostics = solver.step()
        # Stage 3B nodal forces use particle positions at *start* of this step.
        sampled_time = (step-1)*dt
        exact_force = exact_nodal_force_x(
            node_x=5.0,
            x_left=left0+speed*sampled_time,
            x_right=left0+length+speed*sampled_time,
            h=1.0, thickness=H, sigma_xx=sigma_xx,
        )
        computed_force = float(grid.force_internal[sample_node,0])
        # Track distinct particles that crossed the x=5 cell boundary.
        crossed = np.nonzero((previous_x < 5.0) & (particles.position[:,0] >= 5.0))[0]
        crossings.update(int(i) for i in crossed)
        previous_x = particles.position[:,0].copy()
        records.append(dict(
            step=step, time=float(sampled_time),
            x_left_m=float(left0+speed*sampled_time),
            x_right_m=float(left0+length+speed*sampled_time),
            numerical_force_n=computed_force,
            continuum_force_n=exact_force,
            signed_error_n=computed_force-exact_force,
            abs_error_n=abs(computed_force-exact_force),
            stress_drift_pa=float(np.max(np.abs(particles.stress[:,0]-sigma_xx))),
            min_jacobian=float(diagnostics.min_jacobian),
        ))
    errors = np.array([r['signed_error_n'] for r in records])
    force = np.array([r['numerical_force_n'] for r in records])
    reference = np.array([r['continuum_force_n'] for r in records])
    drift = max(r['stress_drift_pa'] for r in records)
    final_x_ref = start_positions[:,0] + speed * t_end
    position_error = float(np.max(np.abs(particles.position[:,0]-final_x_ref)))
    force_error = float(np.max(np.abs(errors)))
    normalized_max_error = force_error/force_scale
    normalized_rms_error = float(np.sqrt(np.mean(errors**2))/force_scale)
    # Compare *changes* in nodal force after subtracting smooth analytical
    # changes. This captures the staircase jumps during particle crossing.
    jump_noise = float(np.max(np.abs(np.diff(force)-np.diff(reference))))
    metrics = dict(
        max_force_error_n=force_error,
        normalized_max_force_error=normalized_max_error,
        normalized_rms_force_error=normalized_rms_error,
        max_force_change_noise_n=jump_noise,
        max_stress_drift_pa=drift,
        final_position_error_m=position_error,
        crossing_particles=len(crossings),
        particles=n,
        steps=steps,
    )
    # Acceptance here is a *quadrature error bound* for uniform regular particles,
    # NOT an assertion that cell-crossing error vanishes.
    thresholds = dict(
        normalized_max_force_error=1.1/ppc,
        max_stress_drift_pa=1e-7,
        final_position_error_m=1e-10,
    )
    passed = (
        all(metrics[k] <= t for k,t in thresholds.items())
        and metrics['crossing_particles'] > 0
        and min(r['min_jacobian'] for r in records) > 0.999999999
    )
    return CrossingRun(
        benchmark=BenchmarkResult(
            name=f'6E-2 moving prestressed block ppc={ppc}',
            passed=passed, metrics=metrics, thresholds=thresholds,
            history=records,
            notes=('Fully prescribed uniform nodal velocity isolates force-assembly quadrature. '
                   'Nonzero nodal force does not drive motion in this case; continuum force '
                   'reference integrates identical spatially constant tensile stress.'),
        ),
        parameters=dict(particles_per_cell=ppc, dt_s=dt, t_end_s=t_end,
                        block_length_m=length, speed_m_s=speed,
                        sigma_xx_pa=sigma_xx, target_node_x_m=5.0,
                        force_scale_n=force_scale),
    )


def particle_refinement(*, ppc_values: tuple[int, ...] = (2,4,8),
                        dt: float = 0.01, t_end: float = 1.5) -> dict:
    """Check whether nodal force sampling errors decrease with more particles."""
    if len(ppc_values) < 2:
        raise ValueError('At least two particle resolutions are required.')
    runs = [moving_stressed_block(particles_per_cell=int(ppc), dt=dt,t_end=t_end)
            for ppc in ppc_values]
    rows = [dict(particles_per_cell=int(ppc),
                 normalized_max_force_error=r.benchmark.metrics['normalized_max_force_error'],
                 normalized_rms_force_error=r.benchmark.metrics['normalized_rms_force_error'],
                 max_force_change_noise_n=r.benchmark.metrics['max_force_change_noise_n'],
                 particles=r.benchmark.metrics['particles'])
            for ppc,r in zip(ppc_values,runs)]
    errors = [row['normalized_max_force_error'] for row in rows]
    success = bool(all(r.benchmark.passed for r in runs) and
                   all(b < a for a,b in zip(errors[:-1],errors[1:])))
    return dict(passed=success, rows=rows, runs=runs,
                description='Particle refinement with fixed grid spacing and time step; not grid refinement.')


__all__ = ['CrossingRun','one_particle_probe','exact_nodal_force_x',
           'moving_stressed_block','particle_refinement']
