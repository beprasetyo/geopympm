"""GeoPyMPM Stage 6E: cell-crossing diagnostics for standard Quad4 point MPM.

Run as: python -m validation.stage6.run_crossing --plots --refinement
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

from .cell_crossing import one_particle_probe, moving_stressed_block, particle_refinement


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plots(out: Path, probe, baseline, refinement) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('Matplotlib is optional; install it or omit --plots.') from exc

    fig, ax = plt.subplots(figsize=(7,4))
    x = [(r['x_m']-2.0)*1e6 for r in probe.history]
    y = [r['force_n'] for r in probe.history]
    ax.scatter(x, y, marker='o', label='Point MPM one-particle contribution')
    ax.axvline(0.0, linestyle='--', linewidth=1, color='black', label='Cell interface x=2 m')
    ax.set(xlabel='Particle offset from x=2 (micrometers)',ylabel='Bottom node (2,0) internal force fx (N)',
           title='6E-1 — gradient/stencil discontinuity at cell crossing')
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(out/'6E_single_particle_jump.png',dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,4.3))
    h=baseline.benchmark.history
    ax.plot([r['time'] for r in h],[r['numerical_force_n'] for r in h],
            label=f'Point MPM {baseline.parameters["particles_per_cell"]} particles/cell',lw=1.5)
    ax.plot([r['time'] for r in h],[r['continuum_force_n'] for r in h],
            label='Analytical continuum nodal force',linestyle='--',color='black',lw=1.5)
    ax.set(xlabel='Time (s)',ylabel='Internal force fx (N) at node (5,0)',
           title='6E-2 — moving prestressed block: force staircase')
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(out/'6E_nodal_force_history.png',dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,4.3))
    ax.plot([r['time'] for r in h],[r['signed_error_n'] for r in h])
    ax.axhline(0.0,color='black',lw=1,linestyle='--')
    ax.set(xlabel='Time (s)',ylabel='Numerical − continuum force (N)',
           title='6E-2 — cell-crossing force quadrature error')
    ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(out/'6E_force_error_history.png',dpi=160); plt.close(fig)
    if refinement is not None:
        fig,ax=plt.subplots(figsize=(7,4.3))
        rows=refinement['rows']
        ax.plot([r['particles_per_cell'] for r in rows],
                [r['normalized_max_force_error'] for r in rows],marker='o',label='Maximum normalized error')
        ax.plot([r['particles_per_cell'] for r in rows],
                [r['normalized_rms_force_error'] for r in rows],marker='s',label='RMS normalized error')
        ax.set(xlabel='Particles per x-cell (2 rows in y)',ylabel='Error / (sigma * H / 2)',
               title='6E — particle refinement with fixed grid')
        ax.grid(alpha=.2);ax.legend();fig.tight_layout()
        fig.savefig(out/'6E_particle_refinement.png',dpi=160);plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='validation/results/stage6e',help='Output directory')
    parser.add_argument('--plots',action='store_true',help='Write PNG plots (requires Matplotlib)')
    parser.add_argument('--refinement',action='store_true',help='Run ppc=2/4/8 comparison')
    args=parser.parse_args(argv)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    probe=one_particle_probe()
    baseline=moving_stressed_block(particles_per_cell=4)
    refinement=particle_refinement() if args.refinement else None
    passed=bool(probe.passed and baseline.benchmark.passed and
                (refinement is None or refinement['passed']))
    summary=dict(
        suite='GeoPyMPM Stage 6E — point-MPM cell-crossing transfer error',
        passed=passed,
        probe={k:v for k,v in asdict(probe).items() if k!='history'},
        primary_ppc4=dict(parameters=baseline.parameters,
                          benchmark={k:v for k,v in asdict(baseline.benchmark).items() if k!='history'}),
        particle_refinement=dict(passed=refinement['passed'],rows=refinement['rows'])
        if refinement is not None else None,
        scope=('Controlled prescribed-velocity motion with nonzero stress: verifies force-assembly '
               'sampling against continuum integral, NOT force-driven dynamics or GIMP.'),
    )
    (out/'6E_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    _csv(out/'6E_probe.csv',probe.history)
    _csv(out/'6E_ppc4_history.csv',baseline.benchmark.history)
    if refinement is not None:
        _csv(out/'6E_particle_refinement.csv',refinement['rows'])
        for r in refinement['runs']:
            ppc=r.parameters['particles_per_cell']
            if ppc!=4:
                _csv(out/f'6E_ppc{ppc}_history.csv',r.benchmark.history)
    if args.plots:
        _plots(out,probe,baseline,refinement)
    for res in (probe,baseline.benchmark):
        print(f'{"PASS" if res.passed else "FAIL"} | {res.name}')
        for k,v in res.metrics.items():
            print(f'    {k}: {v:.9g}' if isinstance(v,float) else f'    {k}: {v}')
    if refinement is not None:
        print(f'{"PASS" if refinement["passed"] else "FAIL"} | 6E particle refinement (fixed grid)')
        for row in refinement['rows']:
            print('    ppc={particles_per_cell} max_error={normalized_max_force_error:.6g} '
                  'rms_error={normalized_rms_force_error:.6g} jump_noise_n={max_force_change_noise_n:.6g}'.format(**row))
    print(f'Outputs: {out.resolve()}')
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
