"""Run GeoPyMPM Stage 6D: fixed-end elastic standing-wave verification.

From repository root, after installing the editable package:
    python -m validation.stage6.run_wave --plots --compare-pic --refinement

Primary verification = FLIP (no added artificial damping). The PIC case is
reported as a damping COMPARISON and does not count toward the suite PASS.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

from .elastic_wave import WaveRun, grid_refinement, vibrating_bar


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plots(out: Path, primary: WaveRun, comparison: WaveRun | None, refinement: dict | None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit('To use --plots, install matplotlib: python -m pip install matplotlib') from exc

    runs = [(primary, 'FLIP (alpha=0)')]
    if comparison is not None:
        runs.append((comparison, 'PIC (alpha=1)'))

    for key, ref_key, ylabel, name in [
        ('modal_displacement_m', 'modal_displacement_reference_m', 'Modal displacement (m)', '6D_modal_displacement.png'),
        ('modal_velocity_m_s', 'modal_velocity_reference_m_s', 'Modal velocity (m/s)', '6D_modal_velocity.png'),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for wave, label in runs:
            history = wave.benchmark.history
            ax.plot([r['time'] for r in history], [r[key] for r in history], label=label)
        h = primary.benchmark.history
        ax.plot([r['time'] for r in h], [r[ref_key] for r in h], color='black',
                linestyle='--', linewidth=1.4, label='Analytical 1D plane-strain mode')
        ax.set(xlabel='Time (s)', ylabel=ylabel, title='GeoPyMPM Stage 6D — longitudinal standing wave')
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / name, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for wave, label in runs:
        h = wave.benchmark.history
        ax.plot([r['time'] for r in h], [r['total_energy_j'] for r in h], label=label)
    e0 = primary.benchmark.history[0]['total_energy_j']
    ax.axhline(e0, linestyle='--', linewidth=1.4, color='black', label='Initial mechanical energy')
    ax.set(xlabel='Time (s)', ylabel='Approximate total mechanical energy (J)',
           title='6D — kinetic + small-strain elastic energy')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / '6D_energy.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for wave, label in runs:
        h = wave.benchmark.history
        ax.plot([r['time'] for r in h], [r['stress_relative_l2'] for r in h], label=label)
    ax.set(xlabel='Time (s)', ylabel='Normalized L2 stress error (σxx)',
           title='6D — stress wave error')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / '6D_stress_error.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name in ('initial', 'quarter_period', 'half_period', 'final'):
        records = [p for p in primary.profiles if p['snapshot'] == name]
        if not records:
            continue
        records.sort(key=lambda p: (p['X_m'], p['Y_m']))
        ax.plot([r['X_m'] for r in records], [r['displacement_m'] for r in records],
                linewidth=1.5, label=name)
    ax.set(xlabel='Reference x coordinate X (m)', ylabel='Particle displacement ux (m)',
           title='6D — FLIP bar displacement profiles')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / '6D_spatial_displacement.png', dpi=160)
    plt.close(fig)

    if refinement is not None:
        rows = refinement['rows']
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([r['nx'] for r in rows], [r['stress_relative_l2_max'] for r in rows],
                marker='o', label='Max normalized stress L2 error')
        ax.set(xlabel='Number of cells along bar (nx)', ylabel='Max normalized error',
               title='6D — joint grid and time-step refinement')
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / '6D_joint_refinement.png', dpi=160)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='validation/results/stage6d',
                        help='Output folder for JSON, CSV, optional PNG plots')
    parser.add_argument('--plots', action='store_true', help='Write PNG plots (requires matplotlib)')
    parser.add_argument('--compare-pic', action='store_true',
                        help='Run PIC alpha=1 as a numerical-damping comparison (not a PASS gate)')
    parser.add_argument('--refinement', action='store_true',
                        help='Run joint nx=16/32/64, dt=0.001/0.0005/0.00025 comparison')
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    primary = vibrating_bar(pic_fraction=0.0)
    comparison = vibrating_bar(pic_fraction=1.0) if args.compare_pic else None
    refinement = grid_refinement() if args.refinement else None

    def compact(wave: WaveRun) -> dict:
        return dict(
            parameters=wave.parameters,
            benchmark={key: value for key, value in asdict(wave.benchmark).items() if key != 'history'},
        )

    suite_passed = primary.benchmark.passed and (refinement is None or refinement['passed'])
    summary = dict(
        suite='GeoPyMPM Stage 6D — standing longitudinal elastic wave',
        passed=bool(suite_passed),
        primary_FLIP=compact(primary),
        comparison_PIC=compact(comparison) if comparison is not None else None,
        joint_refinement=refinement,
        interpretation='PIC is intentionally not part of the PASS gate: dissipative PIC is compared against an undamped continuum reference.',
    )
    (out / '6D_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    _write_csv(out / '6D_FLIP_history.csv', primary.benchmark.history)
    _write_csv(out / '6D_FLIP_profiles.csv', primary.profiles)
    if comparison is not None:
        _write_csv(out / '6D_PIC_history.csv', comparison.benchmark.history)
        _write_csv(out / '6D_PIC_profiles.csv', comparison.profiles)
    if refinement is not None:
        _write_csv(out / '6D_joint_refinement.csv', refinement['rows'])
    if args.plots:
        _plots(out, primary, comparison, refinement)

    print(f"{'PASS' if primary.benchmark.passed else 'FAIL'} | 6D FLIP standing-wave benchmark")
    for key, value in primary.benchmark.metrics.items():
        if isinstance(value, float):
            print(f'    {key}: {value:.9g}')
        else:
            print(f'    {key}: {value}')
    if comparison is not None:
        m = comparison.benchmark.metrics
        print('INFO | 6D PIC comparison (not a PASS criterion)')
        print(f"    energy_ratio_final: {m['energy_ratio_final']:.6g}")
        print(f"    displacement_relative_l2_max: {m['displacement_relative_l2_max']:.6g}")
        print(f"    period_numerical_s: {m['period_numerical_s']:.6g}")
    if refinement is not None:
        print(f"{'PASS' if refinement['passed'] else 'FAIL'} | 6D joint grid/time refinement")
        for row in refinement['rows']:
            print(f"    nx={row['nx']} dt={row['dt_s']:.6g} stress_error={row['stress_relative_l2_max']:.6g}")
    print(f'Outputs: {out.resolve()}')
    return 0 if suite_passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
