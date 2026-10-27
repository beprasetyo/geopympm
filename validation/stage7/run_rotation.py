"""CLI: python -m validation.stage7.run_rotation [--plots] [--output PATH]."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

from .objective_rotation import rigid_spin


def _csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', default='validation/results/stage7')
    ap.add_argument('--plots', action='store_true')
    args = ap.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    obj = rigid_spin(stress_update="corotational")
    old = rigid_spin(stress_update="incremental")
    info = dict(title="GeoPyMPM Stage 7 objective stress rotation verification",
                corotational={k:v for k,v in asdict(obj).items() if k!='history'},
                legacy_comparison={k:v for k,v in asdict(old).items() if k!='history'},
                passed=obj.passed and old.metrics['stress_objectivity_error_pa'] > 1.0)
    (output/'7_summary.json').write_text(json.dumps(info, indent=2)+'\n', encoding='utf-8')
    _csv(output/'7_corotational_history.csv', obj.history)
    _csv(output/'7_incremental_history.csv', old.history)
    if args.plots:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError('Install matplotlib to use --plots.') from exc
        fig, ax = plt.subplots(figsize=(7,4))
        for run, label in [(obj,'Corotational'), (old,'Non-objective (legacy)')]:
            ax.plot([r['time'] for r in run.history],
                    [r['stress_objectivity_error_pa'] for r in run.history],label=label)
        ax.set(xlabel='Time (s)', ylabel='Maximum stress rotation error (Pa)',
               title='Stage 7 — Rigid-body spin: stress objectivity')
        ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
        fig.savefig(output/'7_stress_rotation_error.png', dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot([r['time'] for r in obj.history],
                [r['jacobian_max'] for r in obj.history])
        ax.axhline(1.0, color='black', linewidth=1, linestyle='--', label='Exact rigid rotation: J=1')
        ax.set(xlabel='Time (s)', ylabel='det(F)', title='Known Stage-3D Euler kinematics drift')
        ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
        fig.savefig(output/'7_kinematic_jacobian_drift.png', dpi=160)
        plt.close(fig)
    print(('PASS' if info['passed'] else 'FAIL')+' | Stage 7 corotational rigid-spin verification')
    for k,v in obj.metrics.items():
        print(f'    {k}: {v:.9g}' if isinstance(v,(float,int)) else f'    {k}: {v}')
    print('INFO | legacy non-objective stress comparison (expected error)')
    print(f'    maximum_rotation_error_pa: {old.metrics["stress_objectivity_error_pa"]:.9g}')
    print(f'Outputs: {output.resolve()}')
    return 0 if info['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
