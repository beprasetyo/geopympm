"""Run Stage-8 comparison: python -m validation.stage8.run_kinematics --plots."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
from .finite_rotation import rotation_comparison


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', default='validation/results/stage8')
    ap.add_argument('--plots', action='store_true')
    args = ap.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    case = rotation_comparison()
    (output/'8_summary.json').write_text(json.dumps({k:v for k,v in asdict(case).items() if k!='history'}, indent=2)+'\n', encoding='utf-8')
    with (output/'8_rigid_rotation_history.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(case.history[0]))
        writer.writeheader(); writer.writerows(case.history)
    if args.plots:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError('Install matplotlib to use --plots.') from exc
        fig,ax=plt.subplots(figsize=(7,4))
        ax.plot([r['time'] for r in case.history], [r['j_error_euler'] for r in case.history], label='Euler (Stage 3D)')
        ax.plot([r['time'] for r in case.history], [r['j_error_exponential'] for r in case.history], label='Exponential map (Stage 8)')
        ax.set(xlabel='Time (s)', ylabel='|det(F) - 1|', title='Stage 8 — Rigid-rotation volume drift')
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
        fig.savefig(output/'8_jacobian_drift.png',dpi=160)
        plt.close(fig)
    print(('PASS' if case.passed else 'FAIL')+' | Stage 8 exponential-map finite-rotation kinematics')
    for k,v in case.metrics.items():
        print(f'    {k}: {v:.10g}')
    print(f'Outputs: {output.resolve()}')
    return 0 if case.passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
