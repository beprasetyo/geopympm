"""Stage 10C: analytical affine rotations/shear and force-driven elastic wave.

The affine studies prescribe grid velocities and verify exact finite
kinematics; the dynamic wave study uses internal forces and independent
small-strain d'Alembert stress. Do NOT interpret it as a finite-strain
reference or assert that CPDI must outperform standard MPM in every test.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from geopympm.core import ParticleSet, BackgroundGrid
from geopympm.interpolation import candidate_cpdi_geometry, update_cpdi_domain
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver
from validation.stage10.force_driven_crossing import make_case, analytical, rms


def affine_case(mode='shear', *, nsteps=25, dt=.01):
    p=ParticleSet([[4.,4.]],[1.],[.25],domain_size=[[.35,.55]])
    g=BackgroundGrid((0.,0.),(1.,1.),(9,9))
    g.velocity_bc_mask[:]=True
    if mode=='shear':
        rate=.45
        g.velocity_bc_value[:,0]=rate*(g.node_coordinates[:,1]-4.)
        A=np.array([[0.,rate],[0.,0.]])
    elif mode=='spin':
        omega=2.5
        r=g.node_coordinates - [4.,4.]
        g.velocity_bc_value[:]=np.column_stack((-omega*r[:,1], omega*r[:,0]))
        A=np.array([[0.,-omega],[omega,0.]])
    elif mode=='extension':
        a,b=.25,-.10
        g.velocity_bc_value[:]=(g.node_coordinates-[4.,4.])*np.array([a,b])
        A=np.diag([a,b])
    else:raise ValueError(mode)
    s=ExplicitMPMSolver(p,g,LinearElastic(1.,.25),dt=dt,
        interpolation='cpdi',pic_fraction=0.,
        kinematics_update='exponential',stress_update='corotational',
        advection_update='affine_exponential')
    rows=[]
    for k in range(1,nsteps+1):
        s.step();t=k*dt
        if mode=='shear': expected=np.array([[1.,A[0,1]*t],[0.,1.]])
        elif mode=='spin':
            theta=A[1,0]*t
            expected=np.array([[math.cos(theta),-math.sin(theta)],
                               [math.sin(theta), math.cos(theta)]])
        else:expected=np.diag(np.exp(np.diag(A)*t))
        expected_box=np.abs(expected)@p.domain_size0[0]
        F_error=float(np.max(np.abs(p.deformation_gradient[0]-expected)))
        box_error=float(np.max(np.abs(p.domain_size[0]-expected_box)))
        rows.append(dict(mode=mode,time_s=t,step=k,
                         F_max_error=F_error,domain_bbox_max_error_m=box_error,
                         determinant=float(np.linalg.det(p.deformation_gradient[0])),
                         stress_norm=float(np.linalg.norm(p.stress[0]))))
    return rows


def dynamic_wave(method='cpdi', *, drift=1., steps=100, ppc_x=2, dt=.002):
    X,p,g,solver=make_case('gimp' if method=='cpdi' else method,
                            drift=drift,ppc_x=ppc_x,dt=dt,
                            cell_dx=.25,domain_factor=1.,pic_fraction=0.)
    if method=='cpdi':
        # The transferred initial right-going pulse has F_xx=1+eps_xx.
        # Its corners/axis aligned envelope must match that INITIAL state.
        update_cpdi_domain(p,g)
        solver=ExplicitMPMSolver(p,g,LinearElastic(1200.,.25),dt=dt,
                   pic_fraction=0.,interpolation='cpdi')
    initial_mass=float(p.mass.sum())
    initial_momentum=float(p.mass @ p.velocity[:,0])
    scale=rms(analytical(X[:,0],0.,drift)[2])
    first_cell=np.floor(p.position[:,0]/.25).astype(int)
    crossings=0
    rows=[]
    for k in range(1,steps+1):
        solver.step()
        new_cell=np.floor(p.position[:,0]/.25).astype(int)
        crossings+=int(np.count_nonzero(first_cell!=new_cell))
        first_cell=new_cell
        if k%max(1,steps//20)==0 or k==steps:
            sigma_analytical=analytical(X[:,0],k*dt,drift)[2]
            err=float(rms(p.stress[:,0]-sigma_analytical)/scale)
            rows.append(dict(method=method,drift_m_s=drift,step=k,
                 time_s=k*dt,relative_stress_l2=err,
                 mass_relative_error=abs(float(p.mass.sum())-initial_mass)/initial_mass,
                 momentum_x_absolute_error=abs(float(p.mass@p.velocity[:,0])-initial_momentum),
                 crossings=crossings))
    return rows


def csv_file(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(output='validation/results/stage10c', *, steps=100, ppc_x=2,
        plot=True):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    affine=[]
    for mode in ('shear','spin','extension'):
        affine.extend(affine_case(mode))
    waves=[]
    for method in ('mpm','gimp','cpgimp','cpdi'):
        if method=='cpgimp':
            # Preserve Stage 10B's matched force-driven benchmark.
            from validation.stage10b.validate_cpgimp import wave_case
            old=wave_case('cpgimp',1.,ppc_x,steps,.002)
            waves.extend([dict(method='cpgimp',drift_m_s=1.,step=r['step'],
                    time_s=r['time_s'],relative_stress_l2=r['stress_relative_l2'],
                    mass_relative_error=r['mass_relative_error'],
                    momentum_x_absolute_error=r['momentum_x_absolute_error'],
                    crossings=r['crossings']) for r in old])
        else:
            waves.extend(dynamic_wave(method,steps=steps,ppc_x=ppc_x))
    checks={
        'affine_kinematics_exact':max(r['F_max_error'] for r in affine)<1e-11,
        'domain_geometry_exact':max(r['domain_bbox_max_error_m'] for r in affine)<1e-11,
        'all_four_force_driven_methods_run':len(set(r['method'] for r in waves))==4,
        'mass_conserved':max(r['mass_relative_error'] for r in waves)<1e-12,
        'momentum_conserved':max(r['momentum_x_absolute_error'] for r in waves)<1e-8,
        'cell_crossings_detected':all(r['crossings']>0 for r in waves if r['step']==steps),
        'finite_stress_error':all(np.isfinite(r['relative_stress_l2']) for r in waves),
    }
    summary={'checks':checks,'affine_max_F_error':max(r['F_max_error'] for r in affine),
             'affine_max_bbox_error':max(r['domain_bbox_max_error_m'] for r in affine),
             'wave_final':{m:next(r for r in reversed(waves) if r['method']==m)
                           for m in ('mpm','gimp','cpgimp','cpdi')},
             'note':'Small-strain wave analytical solution; not a finite-strain shear/runout validation.'}
    csv_file(out/'10C_affine_history.csv',affine)
    csv_file(out/'10C_force_driven_history.csv',waves)
    (out/'10C_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    if plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(figsize=(7,4))
            for m in ('shear','spin','extension'):
                z=[r for r in affine if r['mode']==m]
                ax.plot([r['time_s'] for r in z],
                        [r['domain_bbox_max_error_m'] for r in z],label=m)
            ax.set(xlabel='Time (s)',ylabel='Max bounding-box error (m)',
                   title='CPDI1 affine-domain verification')
            ax.legend();fig.tight_layout();fig.savefig(out/'10C_affine_geometry.png',dpi=150)
            plt.close(fig)
            fig,ax=plt.subplots(figsize=(7,4))
            for m in ('mpm','gimp','cpgimp','cpdi'):
                z=[r for r in waves if r['method']==m]
                ax.plot([r['time_s'] for r in z],
                        [r['relative_stress_l2'] for r in z],label=m)
            ax.set(xlabel='Time (s)',ylabel='Relative stress L2 error',
                   title='Stage 10C force-driven wave (drift=1 m/s)')
            ax.legend();fig.tight_layout();fig.savefig(out/'10C_force_driven_stress.png',dpi=150)
            plt.close(fig)
        except ImportError:pass
    print(json.dumps(summary,indent=2));print('Results:',out)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description='Stage 10C CPDI1 verification')
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--ppc-x',type=int,choices=(2,4,8),default=2)
    parser.add_argument('--output',default='validation/results/stage10c')
    parser.add_argument('--no-plot',action='store_true')
    opts=parser.parse_args(argv)
    if opts.steps<1:parser.error('--steps must be >0')
    summary=run(opts.output,steps=opts.steps,ppc_x=opts.ppc_x,plot=not opts.no_plot)
    return 0 if all(summary['checks'].values()) else 1

if __name__=='__main__':raise SystemExit(main())
