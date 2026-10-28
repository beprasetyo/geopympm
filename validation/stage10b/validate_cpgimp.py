"""Stage 10B cpGIMP: affine-stretch verification + dynamic force-driven wave.

The wave case uses a separately derived small-strain analytical stress profile
from Stage 10A. It is a diagnostic, NOT an exact finite-strain reference.
Keep outputs separate from earlier Stage 10A results and never claim that
cpGIMP must outperform uGIMP in an arbitrary wave benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from geopympm.core import ParticleSet, BackgroundGrid
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver
from validation.stage10.force_driven_crossing import make_case, analytical, rms


def affine_verification():
    """Grid-imposed 2D affine velocity, independent reference exp(t*A)."""
    a, b, dt, nsteps = 0.2, -0.15, 0.01, 20
    p = ParticleSet([[2.25, 2.75]], [1.0], [0.2], domain_size=[[0.35,0.45]])
    g = BackgroundGrid((0.,0.),(1.,1.),(6,6))
    g.velocity_bc_mask[:] = True
    g.velocity_bc_value[:] = g.node_coordinates * np.array([a, b])
    s = ExplicitMPMSolver(p,g,LinearElastic(10.,.25),dt=dt,
        interpolation='cpgimp',kinematics_update='exponential',
        advection_update='affine_exponential',pic_fraction=1.0)
    rows=[]
    for k in range(1,nsteps+1):
        s.step()
        t=k*dt
        factor=np.exp([a*t, b*t])
        expected=p.domain_size0[0]*factor
        rows.append(dict(step=k,time_s=t,
            lx_m=float(p.domain_size[0,0]),ly_m=float(p.domain_size[0,1]),
            lx_reference_m=float(expected[0]),ly_reference_m=float(expected[1]),
            domain_max_abs_error_m=float(np.max(np.abs(p.domain_size[0]-expected))),
            jacobian=float(np.linalg.det(p.deformation_gradient[0])),
            reference_jacobian=float(np.prod(factor)),
            volume_relative_error=float(abs(p.volume[0]-p.volume0[0]*np.prod(factor))/
                                      (p.volume0[0]*np.prod(factor)))))
    max_err=max(r['domain_max_abs_error_m'] for r in rows)
    if max_err > 5.e-12:
        raise AssertionError(f'cpGIMP affine stretch mismatch: {max_err}')
    return rows, max_err


def wave_case(method, drift, ppc_x, steps, dt):
    """Benchmark with actual forces from stress, not imposed axial velocities."""
    X, p, g, solver = make_case('gimp' if method=='cpgimp' else method,
                               drift=drift,ppc_x=ppc_x,dt=dt,
                               cell_dx=.25,domain_factor=1.0,pic_fraction=0.0)
    if method=='cpgimp':
        # Stage 10A's small-strain initial condition has F_xx=1+exx. The
        # current interpolation lengths must match the reference lengths
        # transformed by that INITIAL F before a cpGIMP solver is built.
        p.domain_size[:] = p.domain_size0 * np.column_stack(
            (p.deformation_gradient[:,0,0],p.deformation_gradient[:,1,1]))
        solver=ExplicitMPMSolver(p,g,LinearElastic(1200.,.25),dt=dt,
                                 interpolation='cpgimp',pic_fraction=0.0)
    initial_mass=float(p.mass.sum())
    initial_momentum=float(p.mass @ p.velocity[:,0])
    sigma_scale=rms(analytical(X[:,0],0,drift)[2])
    crossings=0
    old_cell=np.floor(p.position[:,0]/.25).astype(int)
    rows=[]
    initial_domain=p.domain_size.copy()
    for k in range(1,steps+1):
        solver.step()
        this_cell=np.floor(p.position[:,0]/.25).astype(int)
        crossings += int(np.count_nonzero(this_cell != old_cell))
        old_cell = this_cell
        if k % max(1,steps//20)==0 or k==steps:
            sr=analytical(X[:,0],k*dt,drift)[2]
            rows.append(dict(method=method, drift_m_s=drift, ppc_x=ppc_x,
               step=k,time_s=k*dt,
               stress_relative_l2=float(rms(p.stress[:,0]-sr)/sigma_scale),
               mass_relative_error=float(abs(p.mass.sum()-initial_mass)/initial_mass),
               momentum_x_absolute_error=float(abs(p.mass@p.velocity[:,0]-initial_momentum)),
               mean_domain_x_ratio=float(np.mean(p.domain_size[:,0]/p.domain_size0[:,0])),
               max_domain_x_change_m=float(np.max(np.abs(p.domain_size[:,0]-initial_domain[:,0]))),
               crossings=crossings))
    if not rows or not np.all(np.isfinite([r['stress_relative_l2'] for r in rows])):
        raise AssertionError('Nonfinite stress diagnostics.')
    if crossings == 0 and drift > 0 and steps >= 120:
        raise AssertionError('Expected x cell crossing from bulk translation.')
    if rows[-1]['mass_relative_error'] > 1e-12:
        raise AssertionError('Particle mass not conserved.')
    if rows[-1]['momentum_x_absolute_error'] > max(1e-10, 1e-11*abs(initial_momentum)):
        raise AssertionError('Unexpected axial momentum drift.')
    if method=='gimp' and not np.array_equal(p.domain_size,p.domain_size0):
        raise AssertionError('uGIMP unexpectedly changed its particle domain.')
    if method=='cpgimp' and not np.allclose(
            p.domain_size, p.domain_size0 * np.column_stack((
                p.deformation_gradient[:,0,0],p.deformation_gradient[:,1,1])),
            rtol=1.e-12,atol=2e-14):
        raise AssertionError('cpGIMP domain lost synchronization with cumulative F.')
    return rows


def save_csv(path, rows):
    with open(path,'w',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)


def run(output='validation/results/stage10b', *, steps=160, ppc_x=2, dt=.002,
        plot=True):
    out=Path(output);out.mkdir(exist_ok=True,parents=True)
    affine,max_err=affine_verification()
    waves=[]; summary={'affine_domain_max_abs_error_m':max_err,'wave':{}}
    for method in ('mpm','gimp','cpgimp'):
        for drift in (0.0,1.0):
            rows=wave_case(method,drift,ppc_x,steps,dt)
            waves.extend(rows)
            summary['wave'][f'{method}_drift_{drift:g}']=rows[-1]
    save_csv(out/'10B_affine_history.csv',affine)
    save_csv(out/'10B_force_driven_history.csv',waves)
    summary['checks']={
        'affine_stretch_exact':max_err<5e-12,
        'all_methods_run':len(summary['wave'])==6,
        'all_stress_finite':all(np.isfinite(r['stress_relative_l2']) for r in waves),
        'force_driven_crossings':any(r['crossings']>0 for r in waves if r['drift_m_s']>0),
        'mass_conserved':all(r['mass_relative_error']<1e-12 for r in waves),
    }
    with open(out/'10B_summary.json','w',encoding='utf8') as f:
        json.dump(summary,f,indent=2)
    if plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(figsize=(7,4))
            for method in ('mpm','gimp','cpgimp'):
                data=[r for r in waves if r['method']==method and r['drift_m_s']==1.0]
                ax.plot([r['time_s'] for r in data],[r['stress_relative_l2'] for r in data],
                        label=method)
            ax.set(xlabel='Time (s)',ylabel='Relative L2 stress error',
                   title='Force-driven elastic wave, drift=1 m/s')
            ax.legend();fig.tight_layout()
            fig.savefig(out/'10B_force_driven_stress_error.png',dpi=160);plt.close(fig)
            fig,ax=plt.subplots(figsize=(7,4))
            ax.plot([r['time_s'] for r in affine],
                    [r['lx_m'] for r in affine],label='cpGIMP x')
            ax.plot([r['time_s'] for r in affine],
                    [r['lx_reference_m'] for r in affine],ls='--',label='Analytical x')
            ax.plot([r['time_s'] for r in affine],
                    [r['ly_m'] for r in affine],label='cpGIMP y')
            ax.plot([r['time_s'] for r in affine],
                    [r['ly_reference_m'] for r in affine],ls='--',label='Analytical y')
            ax.set(xlabel='Time (s)',ylabel='Full domain size (m)',
                   title='cpGIMP affine stretching (grid-imposed velocities)')
            ax.legend();fig.tight_layout()
            fig.savefig(out/'10B_affine_domain.png',dpi=160);plt.close(fig)
        except ImportError:
            pass
    print(json.dumps(summary,indent=2))
    print('Results:',out)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description='Verify Stage 10B cpGIMP')
    parser.add_argument('--output',default='validation/results/stage10b')
    parser.add_argument('--steps',type=int,default=160)
    parser.add_argument('--ppc-x',type=int,choices=(2,4,8),default=2)
    parser.add_argument('--no-plot',action='store_true')
    a=parser.parse_args(argv)
    if a.steps<1:
        parser.error('--steps must be positive')
    checks=run(a.output,steps=a.steps,ppc_x=a.ppc_x,plot=not a.no_plot)['checks']
    return 0 if all(checks.values()) else 1

if __name__=='__main__':
    raise SystemExit(main())
