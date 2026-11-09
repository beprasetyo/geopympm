"""Stage 10E-B side-boundary peak-acceleration diagnostic (solver unchanged).

First run:
    python -m validation.stage10eb.nonlinear_shear --refinement
Then:
    python -m validation.stage10eb.edge_node_diagnostics

This reruns each h=0.25 method only up to the step at which its raw nodal
acceleration peaked. At that exact step, interpolate at x_n BEFORE the solver
step so observed grid arrays align with the interpolation. It reuses the
nonintrusive Stage 10E-A.1 diagnostic (no filtering during time integration).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from validation.stage10e.near_massless import analyze_step, evaluate_at_start
from validation.stage10eb.nonlinear_shear import DT, make_case

THRESHOLDS=(1e-6,1e-8,1e-10)


def analyze_peak(method, peak_step, *, h=.25, dt=DT):
    if not isinstance(peak_step,int) or peak_step<1:
        raise ValueError('peak_step must be a positive int')
    _,_,p,g,solver=make_case(method,h=h,dt=dt)
    for step in range(1,peak_step+1):
        if step==peak_step:
            interp=evaluate_at_start(method,p,g)
        solver.step()
    result=analyze_step(g.mass,g.force_internal,g.acceleration,g.velocity,
        interp.node_ids,interp.weights,interp.gradients,dt=dt,
        particle_mass=p.mass,thresholds=THRESHOLDS,bc_mask=g.velocity_bc_mask)
    return {'method':method,'grid_spacing_m':h,'peak_step':peak_step,**result}


def run(*,outdir='validation/results/stage10eb'):
    base=Path(outdir)
    filename=base/'10EB_summary.json'
    if not filename.exists():
        raise FileNotFoundError(f'Missing {filename}; run nonlinear_shear first.')
    summary=json.loads(filename.read_text(encoding='utf8'))
    rows=[]
    for entry in summary['cases']:
        if entry['grid_spacing_m']!=.25:continue
        method=entry['method']
        step=int(entry['peak_nodal_acceleration']['step'])
        print(f'Inspecting {method} at peak step {step}...',flush=True)
        r=analyze_peak(method,step)
        rows.append(r)
        print(f"  raw={r['raw_max_ax_m_s2']:.4g} m/s2; 1e-8 filtered={r['thresholds']['1e-08']['filtered_max_ax_m_s2']:.4g} m/s2; low-FLIP upper={r['thresholds']['1e-08']['max_abs_low_flip_dvx_upper_m_s']:.3g} m/s",flush=True)
    base.mkdir(parents=True,exist_ok=True)
    (base/'10EB_edge_peak_diagnostics.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description='Near-massless diagnosis at peak boundary accelerations')
    p.add_argument('--output',default='validation/results/stage10eb')
    opts=p.parse_args(argv)
    run(outdir=opts.output)
    return 0

if __name__=='__main__':raise SystemExit(main())
