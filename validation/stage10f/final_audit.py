"""Stage 10F: reproducible interpolation integrity and runtime audit.

Run: python -m validation.stage10f.final_audit --performance

This is a correctness/performance audit, not independent finite-strain
constitutive validation. No solver or kernel modification is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import numba

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.interpolation.mpm import evaluate_quad4
from geopympm.interpolation.gimp import evaluate_gimp
from geopympm.interpolation.cpgimp import update_cpgimp_domain
from geopympm.interpolation.cpdi import evaluate_cpdi, update_cpdi_domain
from geopympm.kernels.grid_update import assemble_internal_force_quad4_kernel
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver

METHODS = ('mpm', 'gimp', 'cpgimp', 'cpdi')
EVALUATORS = dict(mpm=evaluate_quad4, gimp=evaluate_gimp,
                  cpgimp=evaluate_gimp, cpdi=evaluate_cpdi)


def make_interpolation_case(method: str, *, count: int = 128, seed: int = 101):
    if method not in METHODS:
        raise ValueError(f'Unsupported interpolation {method!r}')
    if count < 1:
        raise ValueError('count must be >= 1')
    rng = np.random.default_rng(seed)
    grid = BackgroundGrid((-2., -3.), (.5, .75), (32, 24))
    x = np.column_stack((rng.uniform(1., 7., count), rng.uniform(1., 9., count)))
    domain = np.column_stack((rng.uniform(.08, .27, count),
                              rng.uniform(.12, .33, count)))
    p = ParticleSet(x, mass=rng.uniform(.1, 3., count),
                    volume=rng.uniform(.01, .07, count), domain_size=domain,
                    velocity=rng.uniform(-.2, .2, (count, 2)))
    if method == 'cpgimp':
        p.deformation_gradient[:, 0, 0] = rng.uniform(.8, 1.15, count)
        p.deformation_gradient[:, 1, 1] = rng.uniform(.75, 1.15, count)
        update_cpgimp_domain(p, grid)
    if method == 'cpdi':
        for i in range(count):
            theta = rng.uniform(-math.pi*.7, math.pi*.7)
            c, s = math.cos(theta), math.sin(theta)
            R = np.array([[c, -s], [s, c]])
            T = np.array([[rng.uniform(.7, 1.3), rng.uniform(-.45, .45)],
                          [rng.uniform(-.25, .25), rng.uniform(.7, 1.3)]])
            p.deformation_gradient[i] = R @ T
        update_cpdi_domain(p, grid)
    return p, grid


def measure_invariants(method: str, *, count: int = 128, seed: int = 101) -> dict:
    """Stress-independent identities and mass/force transfer at random F."""
    p, g = make_interpolation_case(method, count=count, seed=seed)
    interpolation = EVALUATORS[method](p, g)
    ids = interpolation.node_ids
    w = interpolation.weights
    grad = interpolation.gradients
    X = g.node_coordinates[ids]
    weighted_x = np.einsum('pa,pai->pi', w, X)
    grad_x = np.einsum('pai,paj->pij', X, grad)
    test_A = np.array([[.23, -.17], [.08, .19]])
    test_b = np.array([1.4, -.9])
    nodal_v = g.node_coordinates @ test_A.T + test_b
    reconstructed_v = np.einsum('pa,pai->pi', w, nodal_v[ids])
    reconstructed_L = np.einsum('pai,paj->pij', nodal_v[ids], grad)

    # Exercise the production Numba accumulation kernels (no surrogate transfer).
    from geopympm.kernels.p2g import p2g_mass_momentum_kernel
    node_mass = np.zeros(g.n_nodes)
    node_p = np.zeros((g.n_nodes, 2))
    p2g_mass_momentum_kernel(ids, w, p.mass, p.velocity, node_mass, node_p)
    fi = np.zeros((g.n_nodes, 2))
    stress = np.zeros_like(p.stress)
    stress[:, 0], stress[:, 1], stress[:, 3] = 11., 6., 2.5
    assemble_internal_force_quad4_kernel(ids, grad, p.volume, stress, fi)

    metrics = {
        'method': method,
        'particle_count': count,
        'support_slots': int(ids.shape[1]),
        'max_nonzero_slots': int(np.max(np.count_nonzero(w > 1e-14, axis=1))),
        'partition_unity_max_abs_error': float(np.max(np.abs(w.sum(axis=1)-1.))),
        'gradient_sum_max_abs_error_1_per_m': float(np.max(np.abs(grad.sum(axis=1)))),
        'position_reproduction_max_error_m': float(np.max(np.abs(weighted_x - p.position))),
        'linear_gradient_reproduction_max_error': float(
            np.max(np.abs(grad_x - np.eye(2)))),
        'affine_velocity_reproduction_max_error_m_s': float(
            np.max(np.abs(reconstructed_v - (p.position @ test_A.T + test_b)))),
        'affine_velocity_gradient_max_error_1_per_s': float(
            np.max(np.abs(reconstructed_L - test_A))),
        'mass_transfer_relative_error': float(abs(node_mass.sum()-p.mass.sum())/p.mass.sum()),
        'momentum_transfer_max_abs_error': float(
            np.max(np.abs(node_p.sum(axis=0)-(p.mass[:, None]*p.velocity).sum(axis=0)))),
        'total_internal_force_max_abs_N': float(np.max(np.abs(fi.sum(axis=0)))),
        'min_weight': float(w.min()),
        'max_weight': float(w.max()),
    }
    metrics['checks'] = {
        'partition_of_unity': metrics['partition_unity_max_abs_error'] < 1e-11,
        'gradient_sum': metrics['gradient_sum_max_abs_error_1_per_m'] < 1e-10,
        'linear_field_reproduction': metrics['position_reproduction_max_error_m'] < 1e-10,
        'linear_gradient_reproduction': metrics['linear_gradient_reproduction_max_error'] < 1e-10,
        'affine_velocity_and_gradient': (
            metrics['affine_velocity_reproduction_max_error_m_s'] < 1e-10 and
            metrics['affine_velocity_gradient_max_error_1_per_s'] < 1e-10),
        'conserved_p2g_mass_momentum': (
            metrics['mass_transfer_relative_error'] < 1e-11 and
            metrics['momentum_transfer_max_abs_error'] < 1e-10),
        'zero_net_internal_force_for_uniform_stress': (
            metrics['total_internal_force_max_abs_N'] < 1e-10),
        'nonnegative_weights': metrics['min_weight'] >= -1e-13,
    }
    return metrics


def make_runtime_case(method: str, count: int, seed: int = 222):
    rng = np.random.default_rng(seed)
    h = .25
    grid = BackgroundGrid((0., 0.), (h, h), (64, 64))
    x = rng.uniform(4.5, 11.5, size=(count, 2))
    lp = np.full((count, 2), .125)
    v = np.tile([.03, -.025], (count, 1))
    vol = np.full(count, lp[0, 0]*lp[0, 1])
    p = ParticleSet(x, vol*1200., vol, velocity=v, domain_size=lp)
    return p, grid


def warm_median(call, *, repetitions: int, min_seconds: float = 0.02):
    """Discard warmup/JIT; use median of batches to reduce clock noise."""
    call()
    values = []
    for _ in range(repetitions):
        t0 = perf_counter()
        n = 0
        while perf_counter()-t0 < min_seconds:
            call()
            n += 1
        values.append((perf_counter()-t0)/n)
    return float(median(values))


def measure_performance(*, sizes=(256, 1024), repetitions=3):
    if repetitions < 2:
        raise ValueError('repetitions must be >= 2')
    rows = []
    for count in sizes:
        for method in METHODS:
            p, g = make_runtime_case(method, count)
            # Full interpolation evaluator + wrapper validation including array alloc.
            t_eval = warm_median(lambda: EVALUATORS[method](p, g), repetitions=repetitions)
            # A separate fresh case for the complete step; includes all old and
            # new interpolation freshness checks, rollback copies and kernels.
            p2, g2 = make_runtime_case(method, count)
            solver = ExplicitMPMSolver(
                p2, g2, LinearElastic(1200., .25), dt=.001,
                pic_fraction=0., stress_update='corotational',
                kinematics_update='exponential',
                advection_update='affine_exponential', interpolation=method,
            )
            t_step = warm_median(solver.step, repetitions=repetitions)
            rows.append({
                'method': method,
                'particle_count': int(count),
                'grid_nodes': int(g.n_nodes),
                'interpolation_s': t_eval,
                'solver_step_s': t_step,
                'interpolation_us_per_particle': t_eval/count*1e6,
                'solver_step_us_per_particle': t_step/count*1e6,
            })
    return rows


def run(output='validation/results/stage10f', *, count=128, performance=False,
        sizes=(256, 1024), repetitions=3, seed=101):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    checks = [measure_invariants(method, count=count, seed=seed) for method in METHODS]
    summary = {
        'stage': '10F',
        'scope': 'invariants and benchmark timing only; not full finite-strain constitutive validation',
        'environment': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'numba': numba.__version__,
            'platform': platform.platform(),
        },
        'invariants': checks,
        'all_invariant_checks_pass': all(all(q['checks'].values()) for q in checks),
    }
    perf = []
    if performance:
        perf = measure_performance(sizes=sizes, repetitions=repetitions)
        summary['performance'] = {
            'conditions': ('same particle count / background grid, unstressed uniform translation, '
                           'domain 0.125 m, h=0.25 m; serial CPU; JIT excluded; '
                           'median of repeated timing batches; solver cost includes Python '
                           'checks, state copies, and all substeps'),
            'note': 'These timings are machine-dependent, not an HPC scaling claim.',
            'rows': perf,
        }
        with (out/'10F_performance.csv').open('w', newline='', encoding='utf8') as f:
            writer = csv.DictWriter(f, fieldnames=list(perf[0]))
            writer.writeheader(); writer.writerows(perf)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            for method in METHODS:
                sub = [v for v in perf if v['method'] == method]
                ax.plot([v['particle_count'] for v in sub],
                        [v['solver_step_s']*1000 for v in sub], marker='o', label=method)
            ax.set(xlabel='Number of particles', ylabel='Warm solver timestep (ms)',
                   title='Stage 10F — End-to-end explicit timestep cost')
            ax.legend(); ax.grid(alpha=.2)
            fig.tight_layout(); fig.savefig(out/'10F_step_performance.png', dpi=160)
            plt.close(fig)
        except ImportError:
            pass
    (out/'10F_audit.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print('Stage 10F interpolation audit')
    for m in checks:
        print(f"  {m['method']:7}  slots={m['support_slots']:2}  "
              f"unity={m['partition_unity_max_abs_error']:.2e}  "
              f"linear_grad={m['linear_gradient_reproduction_max_error']:.2e}  "
              f"all_checks={all(m['checks'].values())}")
    if perf:
        print('Warm timing medians (machine-dependent; JIT excluded):')
        for row in perf:
            print(f"  {row['method']:7} n={row['particle_count']:5} "
                  f"eval={row['interpolation_s']*1000:9.4f} ms "
                  f"full_step={row['solver_step_s']*1000:9.4f} ms")
    print('Results:', out)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='validation/results/stage10f')
    ap.add_argument('--count', type=int, default=128)
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--performance', action='store_true')
    ap.add_argument('--sizes', type=int, nargs='+', default=[256, 1024])
    ap.add_argument('--repetitions', type=int, default=3)
    opts = ap.parse_args(argv)
    result = run(opts.output, count=opts.count, performance=opts.performance,
                 sizes=tuple(opts.sizes), repetitions=opts.repetitions,
                 seed=opts.seed)
    return 0 if result['all_invariant_checks_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
