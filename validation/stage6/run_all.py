"""Run Stage 6A–6C benchmarks and save quantitative reports.

From the repository root (with editable package installed):
    python -m validation.stage6.run_all --output validation/results/stage6
Optional plots when matplotlib is available:
    python -m validation.stage6.run_all --plots
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .benchmarks import affine_elastic, free_fall, free_fall_refinement, rigid_translation


def _save_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _plots(out: Path, data: list, refinement: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit('Matplotlib is optional. Install it to use --plots: pip install matplotlib') from exc
    for result in data:
        if result.name.startswith('6A'):
            xkey, ykey, ylabel = 'time', 'position_error', 'Max position error (m)'
        elif result.name.startswith('6B'):
            xkey, ykey, ylabel = 'time', 'continuum_position_error', 'Error vs continuous solution (m)'
        else:
            xkey, ykey, ylabel = 'time', 'stress_error', 'Max stress error (Pa)'
        fig, ax = plt.subplots(figsize=(6.8, 4.1))
        ax.plot([r[xkey] for r in result.history], [r[ykey] for r in result.history], marker='o', markersize=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel)
        ax.set_title(result.name)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        file_name = result.name[:2] + '_history.png'
        fig.savefig(out / file_name, dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    ax.loglog(refinement['dt_values'], refinement['errors'], marker='o', label='MPM vs exact continuum')
    ax.loglog(refinement['dt_values'],
              refinement['errors'][0] * np.asarray(refinement['dt_values']) / refinement['dt_values'][0],
              '--', label='First-order reference')
    ax.set_xlabel('Time step dt (s)')
    ax.set_ylabel('Max free-fall position error (m)')
    ax.set_title('Stage 6B: temporal convergence')
    ax.grid(alpha=0.25, which='both')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / '6B_convergence.png', dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='validation/results/stage6',
                        help='Folder for summary JSON and benchmark CSVs')
    parser.add_argument('--plots', action='store_true',
                        help='Also create PNG plots; requires optional matplotlib')
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    cases = [rigid_translation(), free_fall(), affine_elastic()]
    refinement = free_fall_refinement()
    summary = {
        'suite': 'GeoPyMPM Stage 6A–6C verification',
        'solver': 'Stage 5 integrated explicit, PIC=1.0 baseline, USL-like',
        'cases': [{k: v for k, v in asdict(r).items() if k != 'history'} for r in cases],
        'free_fall_refinement': refinement,
        'passed': all(r.passed for r in cases) and refinement['passed'],
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    for result in cases:
        _save_csv(out / (result.name[:2] + '_history.csv'), result.history)
    if args.plots:
        _plots(out, cases, refinement)
    for case in cases:
        print(f"{'PASS' if case.passed else 'FAIL'} | {case.name}")
        for name, value in case.metrics.items():
            print(f'    {name}: {value:.9g}')
    print(f"{'PASS' if refinement['passed'] else 'FAIL'} | 6B temporal refinement: "
          f"orders={refinement['observed_orders']}")
    print(f"Outputs: {out.resolve()}")
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
