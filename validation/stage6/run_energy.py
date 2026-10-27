"""Execute GeoPyMPM Stage-6F energy / separated convergence / CFL verification.

Run from repository root:
    python -m validation.stage6.run_energy --plots
All benchmark results are reproducible CSV/JSON and optional PNG plots.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

from .energy_convergence import full_stage6f


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w',encoding='utf-8',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _plots(out:Path,result:dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit('To use --plots, install matplotlib: python -m pip install matplotlib') from exc
    histories=result['energy'].histories
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    for label,records in histories.items():
        ax.plot([r['time_s'] for r in records],
                [r['total_energy_j']/records[0]['total_energy_j'] for r in records],label=label)
    ax.axhline(1,linestyle='--',color='black',linewidth=1,label='Initial energy')
    ax.set(xlabel='Time (s)',ylabel='(K + U)/E0',title='6F — mechanical energy ratio: PIC / FLIP comparison')
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(out/'6F_energy_ratio.png',dpi=160); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.5,4.5))
    records=histories['FLIP']
    for key,label in [('kinetic_energy_j','Kinetic'),('elastic_energy_j','Small-strain elastic'),
                      ('total_energy_j','Total')]:
        ax.plot([r['time_s'] for r in records],[r[key] for r in records],label=label)
    ax.set(xlabel='Time (s)',ylabel='Energy (J)',title='6F — FLIP exchange of kinetic and elastic energy')
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(out/'6F_flip_energy_components.png',dpi=160); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.5,4.5))
    rows=result['spatial'].rows
    ax.loglog([r['dx_m'] for r in rows],
              [r['max_stress_l2_error'] for r in rows],marker='o',label='Stress error (continuum)')
    ax.set(xlabel='Cell width h (m), dt fixed',ylabel='Max normalized L2 error',
           title='6F — spatial refinement, dt = 0.00025 s')
    ax.grid(alpha=.25,which='both'); ax.legend(); fig.tight_layout()
    fig.savefig(out/'6F_spatial_convergence.png',dpi=160); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.5,4.5))
    rows=result['temporal'].rows
    ax.loglog([r['dt_s'] for r in rows],
              [r['final_stress_relative_reference_error'] for r in rows],marker='o',label='Stress vs fine-step reference')
    ax.loglog([r['dt_s'] for r in rows],
              [r['final_displacement_relative_reference_error'] for r in rows],marker='s',label='Displacement vs fine-step reference')
    ax.set(xlabel='Time step dt (s), grid fixed',ylabel='Relative final-time field error',
           title='6F — temporal refinement, nx = 32')
    ax.grid(alpha=.25,which='both'); ax.legend(); fig.tight_layout()
    fig.savefig(out/'6F_temporal_convergence.png',dpi=160); plt.close(fig)


def main(argv: list[str]|None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='validation/results/stage6f',
                        help='Output directory for JSON/CSV/optional plots')
    parser.add_argument('--plots',action='store_true',help='Write PNG charts; requires matplotlib')
    args=parser.parse_args(argv)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    result=full_stage6f()
    energy,space,temporal,cfl=(result[x] for x in ('energy','spatial','temporal','cfl'))
    summary=dict(suite='GeoPyMPM Stage 6F — Energy and Separated Convergence',
                 passed=result['passed'],
                 energy={k:v for k,v in asdict(energy).items() if k!='histories'},
                 spatial=asdict(space),temporal=asdict(temporal),cfl=cfl,
                 important_limitations=(
                     'No nonzero prescribed external work: energy is kinetic + '
                     'small-strain elastic only for this constrained wave. '
                     'No finite-rotation stress objectivity, contact or plasticity. '
                     'CFL check is a heuristic, not a stability proof. '
                     'Fixed dt spatial comparison retains temporal error; '
                     'fine-step temporal reference is numerical.'
                 ))
    (out/'6F_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    for label,records in energy.histories.items():
        _write_csv(out/f'6F_{label}_energy_history.csv',records)
    _write_csv(out/'6F_spatial_refinement.csv',space.rows)
    _write_csv(out/'6F_temporal_refinement.csv',temporal.rows)
    if args.plots:
        _plots(out,result)
    print(f"{'PASS' if energy.passed else 'FAIL'} | 6F-1 small-strain energy balance")
    for name,m in energy.metrics.items():
        print(f"    {name}: final_ratio={m['final_energy_ratio']:.8g}  "
              f"max_abs_residual={m['max_abs_energy_residual']:.8g}")
    print(f"{'PASS' if space.passed else 'FAIL'} | 6F-2 spatial refinement (dt fixed)")
    for row in space.rows:
        print(f"    nx={row['nx']} dt={row['dt_s']:.7g} stress_error={row['max_stress_l2_error']:.8g}")
    print(f"    observed stress orders: {[round(v, 5) for v in space.orders]}")
    print(f"{'PASS' if temporal.passed else 'FAIL'} | 6F-3 temporal refinement (grid fixed)")
    for row in temporal.rows:
        print(f"    nx={row['nx']} dt={row['dt_s']:.7g} stress_ref_error={row['final_stress_relative_reference_error']:.8g}")
    print(f"    stress self-convergence orders: {[round(v, 5) for v in temporal.orders]}")
    print(f"{'PASS' if cfl['passed'] else 'FAIL'} | 6F-4 Stage-5 CFL input guard")
    print(f"    dt_limit={cfl['dt_limit_s']:.9g}  accepted_dt={cfl['accepted_dt_s']:.9g} "
          f"rejected_dt={cfl['rejected_dt_s']:.9g}")
    print(f"Outputs: {out.resolve()}")
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
