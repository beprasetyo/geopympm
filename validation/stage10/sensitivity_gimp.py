"""Stage 10A sensitivity: particle-domain width, PIC/FLIP blend, particles/cell.

Uses the Stage 10A independently referenced force-driven travelling elastic
wave benchmark. Each scenario is solved twice with rest/drift reference frames.

The nominal full particle domain width is dx/PPC. Scaling by domain_factor
other than one creates domains that *do not exactly tile* the reference bar;
this changes the interpolation averaging support and is only a numerical
sensitivity diagnostic, NOT an equally physical discretization.

The MPM case has no GIMP domain dependence; compute it once per (PPC, PIC).
No solver kernels or constitutive routines are changed by this module.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .force_driven_crossing import DX, STEP_DT, STEPS, analytical, one_run, rms

DOMAIN_FACTORS = (0.5, 1.0, 2.0)
PIC_FRACTIONS = (0.0, 0.1, 1.0)
PARTICLES_PER_CELL = (2, 4, 8)


def _save_csv(path: Path, items: list[dict]) -> None:
    if not items:
        return
    with path.open('w', newline='', encoding='utf8') as file:
        writer = csv.DictWriter(file, fieldnames=list(items[0]))
        writer.writeheader()
        writer.writerows(items)


def _pair(method: str, ppc: int, pic: float, factor: float, *,
          steps: int, dt: float, cell_dx: float, sample_every: int) -> dict:
    results = {}
    finals = {}
    Xlast = None
    for drift in (0.0, 1.0):
        _, _, summary, stress, X = one_run(
            method, drift=drift, ppc_x=ppc, steps=steps, dt=dt,
            sample_every=sample_every, cell_dx=cell_dx,
            domain_factor=factor, pic_fraction=pic,
        )
        results[drift] = summary
        finals[drift] = stress
        Xlast = X
    ref_final = analytical(Xlast, steps * dt, 0.0)[2]
    galilean = rms(finals[1.0] - finals[0.0]) / rms(ref_final)
    drift = results[1.0]
    rest = results[0.0]
    return dict(
        method=method, ppc_x=ppc, pic_fraction=pic,
        domain_factor=factor if method == 'gimp' else 1.0,
        domain_ratio_to_cell=(factor / ppc if method == 'gimp' else 0.0),
        domain_tiling=(method == 'mpm' or factor == 1.0),
        n_particles=drift['n_particles'], dt_s=dt,
        steps=steps, cell_dx_m=cell_dx, total_time_s=dt * steps,
        drift_cell_crossings=drift['n_particle_cell_crossings'],
        rest_cell_crossings=rest['n_particle_cell_crossings'],
        drift_time_rms_stress_rel_l2=drift['time_rms_stress_relative_l2'],
        rest_time_rms_stress_rel_l2=rest['time_rms_stress_relative_l2'],
        drift_final_stress_rel_l2=drift['final_stress_relative_l2'],
        rest_final_stress_rel_l2=rest['final_stress_relative_l2'],
        galilean_final_stress_rel_l2=galilean,
        crossing_error_increment_rms_pa=drift['stress_increment_crossing_rms_pa'],
        noncrossing_error_increment_rms_pa=drift['stress_increment_non_crossing_rms_pa'],
        crossing_active_samples=drift['crossing_stress_active_samples'],
        noncrossing_active_samples=drift['non_crossing_stress_active_samples'],
        max_mass_relative_error=max(drift['max_mass_relative_error'],rest['max_mass_relative_error']),
        max_momentum_relative_error=max(drift['max_momentum_x_relative_error'],
                                        rest['max_momentum_x_relative_error']),
    )


def _plots(directory: Path, records: list[dict]) -> list[str]:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    paths = []
    # Isolate PIC/FLIP and PPC by selecting a reference PPC=4.
    fig, ax = plt.subplots(figsize=(7.3, 4.6))
    for pic in PIC_FRACTIONS:
        subset = sorted((r for r in records if r['method'] == 'gimp' and
                         r['ppc_x'] == 4 and r['pic_fraction'] == pic),
                        key=lambda r: r['domain_factor'])
        if subset:
            ax.plot([r['domain_factor'] for r in subset],
                    [100 * r['drift_time_rms_stress_rel_l2'] for r in subset],
                    marker='o', label=f'uGIMP PIC={pic:g}')
            ref = next(r for r in records if r['method'] == 'mpm' and
                       r['ppc_x'] == 4 and r['pic_fraction'] == pic)
            ax.axhline(100 * ref['drift_time_rms_stress_rel_l2'], linestyle=':',
                       label=f'MPM PIC={pic:g}')
    ax.set(xlabel='GIMP domain scale factor (x direction)',
           ylabel='Time-RMS relative stress error (%)',
           title='Force-driven wave: interpolation width vs PIC blend (PPC=4)')
    ax.set_xticks(DOMAIN_FACTORS)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = directory/'10A_SENS_stress_vs_domain.png'
    fig.savefig(path, dpi=180);plt.close(fig);paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7.3, 4.6))
    for method in ('mpm','gimp'):
        for ppc in PARTICLES_PER_CELL:
            subset = sorted((r for r in records if r['method'] == method and
                             r['ppc_x'] == ppc and r['domain_factor'] == 1.0),
                            key=lambda r:r['pic_fraction'])
            if subset:
                ax.plot([r['pic_fraction'] for r in subset],
                        [100*r['galilean_final_stress_rel_l2'] for r in subset],
                        marker='o',label=f'{method.upper()}, PPC={ppc}')
    ax.set(xlabel='PIC fraction (0 = FLIP, 1 = PIC)',
           ylabel='Boost vs rest: final stress difference (%)',
           title='Galilean-frame sensitivity (nominal domain)')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = directory/'10A_SENS_galilean_vs_pic.png'
    fig.savefig(path,dpi=180);plt.close(fig);paths.append(str(path))

    fig,ax=plt.subplots(figsize=(7.3,4.6))
    for method in ('mpm','gimp'):
        for pic in (0.0, 0.1, 1.0):
            subset = sorted((r for r in records if r['method'] == method and
                             r['pic_fraction'] == pic and r['domain_factor'] == 1.0),
                            key=lambda r:r['ppc_x'])
            if subset:
                ax.plot([r['ppc_x'] for r in subset],
                        [100*r['drift_time_rms_stress_rel_l2'] for r in subset],
                        '-o',label=f'{method.upper()}, PIC={pic:g}')
    ax.set(xlabel='Particles per cell in x (PPC)',
           ylabel='Time-RMS relative stress error (%)',
           title='Particle resolution and PIC/FLIP sensitivity')
    ax.set_xticks(PARTICLES_PER_CELL)
    ax.legend(fontsize=8,ncol=2);fig.tight_layout()
    path=directory/'10A_SENS_stress_vs_ppc.png'
    fig.savefig(path,dpi=180);plt.close(fig);paths.append(str(path))
    return paths


def run(output='validation/results/stage10a_sensitivity', *, steps=STEPS,
        dt=STEP_DT, cell_dx=DX, sample_every=5, plot=True,
        ppcs=PARTICLES_PER_CELL, pics=PIC_FRACTIONS,
        domain_factors=DOMAIN_FACTORS, verbose=True) -> dict:
    if steps < 1 or sample_every < 1 or not math.isfinite(dt) or dt <= 0:
        raise ValueError('steps and sample_every must be positive, dt positive and finite')
    for val in ppcs:
        if val not in (2,4,8):
            raise ValueError('PPC must be 2, 4, or 8')
    for val in pics:
        if not (math.isfinite(val) and 0 <= val <= 1):
            raise ValueError('PIC must be within [0, 1]')
    for val in domain_factors:
        if not (math.isfinite(val) and 0 < val <= min(ppcs)):
            raise ValueError('GIMP scale factor must be in (0, minimum PPC]')
    out=Path(output)
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    references={}
    for ppc in ppcs:
        for pic in pics:
            ref = _pair('mpm',ppc,pic,1.0,steps=steps,dt=dt,cell_dx=cell_dx,
                        sample_every=sample_every)
            rows.append(ref)
            references[ppc,pic]=ref
            if verbose:
                print(f'MPM ppc={ppc} PIC={pic:g}: error={ref["drift_time_rms_stress_rel_l2"]:.4f} '
                      f'gal={ref["galilean_final_stress_rel_l2"]:.4f}',flush=True)
            for factor in domain_factors:
                val=_pair('gimp',ppc,pic,factor,steps=steps,dt=dt,cell_dx=cell_dx,
                          sample_every=sample_every)
                rows.append(val)
                if verbose:
                    print(f'GIMP ppc={ppc} PIC={pic:g} factor={factor:g}: '
                          f'error={val["drift_time_rms_stress_rel_l2"]:.4f} '
                          f'gal={val["galilean_final_stress_rel_l2"]:.4f}',flush=True)
    for row in rows:
        base=references[row['ppc_x'],row['pic_fraction']]
        row['stress_error_minus_matching_mpm_pct_points']=(
            100*(row['drift_time_rms_stress_rel_l2']-base['drift_time_rms_stress_rel_l2']))
        row['galilean_minus_matching_mpm_pct_points']=(
            100*(row['galilean_final_stress_rel_l2']-base['galilean_final_stress_rel_l2']))
    _save_csv(out/'10A_SENS_all_cases.csv',rows)
    paths=_plots(out,rows) if plot else []
    checks={
        'both_methods_run':set(r['method'] for r in rows) == {'mpm','gimp'},
        'force_driven_crossings':all(r['drift_cell_crossings']>0 for r in rows),
        'mass_conserved':all(r['max_mass_relative_error']<1e-10 for r in rows),
        'axial_momentum_conserved':all(r['max_momentum_relative_error']<1e-8 for r in rows),
        'metrics_finite':all(np.isfinite(r['galilean_final_stress_rel_l2']) and
                             np.isfinite(r['drift_time_rms_stress_rel_l2']) for r in rows),
    }
    default=[r for r in rows if r['ppc_x']==4 and r['pic_fraction']==0.0 and
             r['domain_factor']==1.0]
    result=dict(benchmark='Stage 10A GIMP sensitivity: domain width, PIC/FLIP, PPC',
        reference='independent infinitesimal 1D dAlembert elastic wave',
        baseline='GeoPyMPM Stage 10A force-driven crossing',
        note='domain_factor != 1 does not tile the material body, so only numerical support sensitivity is assessed',
        setup=dict(steps=steps,dt_s=dt,cell_dx_m=cell_dx,
                   ppcs=list(ppcs),pics=list(pics),domain_factors=list(domain_factors)),
        n_cases=len(rows),n_dynamics_runs=2*len(rows),checks=checks,
        default_cases=default,plots=paths)
    with (out/'10A_SENS_summary.json').open('w',encoding='utf8') as file:
        json.dump(result,file,indent=2)
    if verbose:
        print(f'FINISHED: {len(rows)} scenarios, {2*len(rows)} simulations; checks={checks}',flush=True)
        print(f'Output: {out}',flush=True)
    return dict(summary=result, records=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='validation/results/stage10a_sensitivity')
    parser.add_argument('--steps',type=int,default=STEPS)
    parser.add_argument('--dt',type=float,default=STEP_DT)
    parser.add_argument('--cell-dx',type=float,default=DX,choices=(0.125,0.25,0.5))
    parser.add_argument('--sample-every',type=int,default=5)
    parser.add_argument('--quick',action='store_true',
                        help='90 steps, PPC 2 and 4, PIC 0 and 0.1 (smoke test only).')
    parser.add_argument('--no-plot',action='store_true')
    args=parser.parse_args()
    if args.quick:
        run(args.output,steps=min(args.steps,90),dt=args.dt,cell_dx=args.cell_dx,
            sample_every=args.sample_every,plot=not args.no_plot,
            ppcs=(2,4),pics=(0.0,0.1))
    else:
        run(args.output,steps=args.steps,dt=args.dt,cell_dx=args.cell_dx,
            sample_every=args.sample_every,plot=not args.no_plot)


if __name__=='__main__':
    main()
