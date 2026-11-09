"""Stage 10E: force-driven finite-stretch longitudinal-wave validation.

Plane-strain longitudinal motion is a one-dimensional exact reduction of the
current corotational / Jaumann hypoelastic material: F=diag(J,1), spin=0,
σxx=M ln J, M=λ+2μ. In MATERIAL coordinates X the governing PDEs are
    J_t = v_X,     v_t = (M/ρ0) (ln J)_X.
A right-going simple wave with Riemann invariant v+2 c0 sqrt(J)=2 c0 has
    J_t + c0/sqrt(J) J_X = 0,
    v = 2 c0 (1-sqrt(J)).
Before the first characteristic intersection, the reference is implicit:
    X = ξ + c0*t / sqrt(J0(ξ)),   J(X,t)=J0(ξ).
The root is solved by independent bisection, never by the MPM kernels.

Only grid y-velocities are constrained (plane-strain transverse motion);
NO x-velocity boundary conditions, external loads or prescribed velocity
fields are imposed during the time stepping. The initial finite stretch,
nonuniform stress pulse and matched velocity launch the nonlinear wave.
Its spatial stress gradient drives *changes* to particle momentum.

Caveat: no transverse momentum is assessed, and the wave reference is valid
only before characteristics cross and before waves reach physical ends.
A numerical MPM profile is not a finite-strain hyperelastic reference, nor
is this a force-driven shear/rotation or realistic landslide benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.interpolation.cpdi import update_cpdi_domain
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver

RHO0 = 1200.0
E = 1200.0
NU = 0.25
AMP = 0.55   # 55% initial axial stretch; J0 max = 1.55
WIDTH = 0.7
X_CENTER = 4.0
X_LEFT, X_RIGHT = 1.0, 9.0
Y_LOWER, Y_UPPER = 3.5, 4.5
T_END = 0.45
MATERIAL = LinearElastic(E, NU)
LAM = MATERIAL.lame_lambda
MODULUS = LAM + 2.0 * MATERIAL.shear_modulus
C0 = math.sqrt(MODULUS / RHO0)


def stretch_initial(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return 1.0 + AMP*np.exp(-0.5*((X-X_CENTER)/WIDTH)**2)


def initial_position(X: np.ndarray) -> np.ndarray:
    """Integrate dx/dX=J0 exactly with error-function antiderivative."""
    X = np.asarray(X, dtype=float)
    factor = AMP * WIDTH * math.sqrt(math.pi/2.)
    return X + factor*np.array([math.erf((float(z)-X_CENTER)/(math.sqrt(2.)*WIDTH))
                                   for z in np.ravel(X)]).reshape(X.shape)


def characteristics(X: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Independent bisection inversion of X=xi+c(J0(xi))*t.

    For the positive Gaussian, 1 <= J0 <= 1+AMP, thus
    c0/sqrt(1+AMP) <= c(J0) <= c0. The bisection brackets span that range.
    """
    if not np.isfinite(t) or t < 0:
        raise ValueError('t must be finite and nonnegative')
    X = np.asarray(X, dtype=np.float64)
    if t == 0.0:
        xi = X.copy()
        return xi, stretch_initial(xi)
    lo = X - C0*t
    hi = X - (C0/math.sqrt(1.0+AMP))*t
    for _ in range(52):
        mid = .5*(lo+hi)
        f = mid + C0*t/np.sqrt(stretch_initial(mid)) - X
        hi = np.where(f>0.0, mid, hi)
        lo = np.where(f<=0.0, mid, lo)
    xi = .5*(lo+hi)
    return xi, stretch_initial(xi)


def reference(X: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, J = characteristics(X,t)
    return J, 2*C0*(1-np.sqrt(J)), MODULUS*np.log(J)


def minimum_characteristic_jacobian(t: float) -> float:
    xi = np.linspace(X_LEFT-3., X_RIGHT+3., 10001)
    J = stretch_initial(xi)
    Jprime = -AMP*(xi-X_CENTER)/(WIDTH**2)*np.exp(-.5*((xi-X_CENTER)/WIDTH)**2)
    dc_dxi = -0.5*C0*Jprime/J**1.5
    return float(np.min(1.+t*dc_dxi))


def make_case(method: str='cpdi', *, cell_dx: float=.25, ppc_x: int=2,
              dt: float=.003, amp: float=AMP):
    if method not in ('mpm','gimp','cpgimp','cpdi'):
        raise ValueError('unknown interpolation method')
    if not np.isclose(amp,AMP):
        raise ValueError('this reference fixes AMP=0.55; change the constant for a new benchmark')
    if ppc_x not in (2,4):
        raise ValueError('ppc_x must be 2 or 4')
    dx_ref = cell_dx/ppc_x
    nxp = int(round((X_RIGHT-X_LEFT)/dx_ref))
    if not np.isclose(nxp*dx_ref,X_RIGHT-X_LEFT,atol=1e-12):
        raise ValueError('particle spacing must divide reference interval')
    nyp = int(round((Y_UPPER-Y_LOWER)/dx_ref))
    dy_ref = (Y_UPPER-Y_LOWER)/nyp
    if not np.isclose(dy_ref,dx_ref,atol=1e-12):
        raise ValueError('transverse particle spacing must match axial spacing')
    X = X_LEFT+(np.arange(nxp)+0.5)*dx_ref
    Y = Y_LOWER+(np.arange(nyp)+0.5)*dy_ref
    XY = np.array([[x,y] for y in Y for x in X],dtype=float)
    J0 = stretch_initial(XY[:,0])
    x = initial_position(XY[:,0])
    positions = np.column_stack((x,XY[:,1]))
    initial_velocity = np.column_stack((2*C0*(1-np.sqrt(J0)),np.zeros(len(J0))))
    vol_ref = np.full(len(J0), dx_ref*dy_ref)
    stresses = np.zeros((len(J0),6))
    stresses[:,0]=MODULUS*np.log(J0)
    stresses[:,1]=LAM*np.log(J0)
    stresses[:,2]=LAM*np.log(J0)
    sides = np.tile([dx_ref,dy_ref],(len(J0),1))
    # Stage-8 F is relative to MPM initialization (already stretched).
    # The initial physical GIMP domains are J0 times the reference x side.
    domain = sides.copy()
    domain[:,0] *= J0
    p=ParticleSet(positions,vol_ref*RHO0,vol_ref*J0,
                  velocity=initial_velocity,stress=stresses,domain_size=domain)
    # Keep F=I at the MPM starting configuration, and volume0=current volume.
    # The total stretch referred to material X is J0(X)*F_MPM,xx(t).
    h=cell_dx
    extent=12.0
    nc=int(round(extent/h))
    g=BackgroundGrid((0.,0.),(h,h),(nc,nc))
    g.velocity_bc_mask[:,1]=True
    g.velocity_bc_value[:,1]=0.
    if method=='cpdi':update_cpdi_domain(p,g)
    s=ExplicitMPMSolver(p,g,MATERIAL,dt=dt,pic_fraction=0.,cfl=.45,
              interpolation=method,stress_update='corotational',
              kinematics_update='exponential',advection_update='affine_exponential')
    return XY, p, g, s


def scalar_rms(x):return float(np.sqrt(np.mean(np.asarray(x,dtype=float)**2)))


def run_one(method='cpdi', *, cell_dx=.25, ppc_x=2, dt=.003, nsteps=150):
    XY,p,g,s=make_case(method,cell_dx=cell_dx,ppc_x=ppc_x,dt=dt)
    X=XY[:,0]
    mass0=float(p.mass.sum()); momentum0=float(np.dot(p.mass,p.velocity[:,0]))
    max_force=0.; max_acc=0.; max_speed=0.; max_momentum_err=0.
    J_initial=stretch_initial(X)
    v_initial=p.velocity[:,0].copy()
    refvol=p.mass/RHO0
    def continuum_energy(J,v):
        return float(np.sum(refvol*(0.5*RHO0*v*v +
              MODULUS*(J*np.log(J)-J+1.0))))
    energy_initial=continuum_energy(J_initial,v_initial)
    energy_rel_max=0.
    ncell_crossings=0
    cell_old=np.floor(p.position[:,0]/cell_dx).astype(int)
    history=[]
    for k in range(nsteps):
        diag=s.step()
        force=np.linalg.norm(g.force_internal,axis=1)
        max_force=max(max_force,float(force.max()))
        max_acc=max(max_acc,float(np.max(np.abs(g.acceleration[:,0]))))
        max_speed=max(max_speed,float(np.max(np.abs(p.velocity[:,0]))))
        max_momentum_err=max(max_momentum_err,abs(float(np.dot(p.mass,p.velocity[:,0]))-momentum0))
        J_total=J_initial*np.linalg.det(p.deformation_gradient)
        energy_rel_max=max(energy_rel_max,abs(continuum_energy(J_total,p.velocity[:,0])-energy_initial)/energy_initial)
        cell_new=np.floor(p.position[:,0]/cell_dx).astype(int)
        ncell_crossings+=int(np.count_nonzero(cell_new!=cell_old))
        cell_old=cell_new
        if k==0 or (k+1)%max(1,nsteps//15)==0 or k==nsteps-1:
            J,velocity,sigma=reference(X,diag.time)
            history.append(dict(method=method,cell_dx_m=cell_dx,ppc_x=ppc_x,
               t_s=diag.time,relative_stress_l2=scalar_rms(p.stress[:,0]-sigma)/scalar_rms(sigma),
               relative_stretch_l2=scalar_rms(p.deformation_gradient[:,0,0]*stretch_initial(X)-J)/scalar_rms(J),
               velocity_rms_m_s=scalar_rms(p.velocity[:,0]-velocity),
               mass_rel_error=abs(p.mass.sum()-mass0)/mass0,
               momentum_abs_error=abs(np.dot(p.mass,p.velocity[:,0])-momentum0),
               min_J=float(np.min(np.linalg.det(p.deformation_gradient))),
               max_J=float(np.max(np.linalg.det(p.deformation_gradient)))))
    J,velocity,sigma=reference(X,s.time)
    final=dict(method=method,cell_dx_m=cell_dx,ppc_x=ppc_x,
        particles=p.n_particles,steps=nsteps,dt_s=dt,time_s=s.time,
        reference_peak_stretch=float(np.max(J)),
        numerical_peak_stretch=float(np.max(p.deformation_gradient[:,0,0]*stretch_initial(X))),
        final_stress_rel_l2=scalar_rms(p.stress[:,0]-sigma)/scalar_rms(sigma),
        final_stretch_rel_l2=scalar_rms(p.deformation_gradient[:,0,0]*stretch_initial(X)-J)/scalar_rms(J),
        final_velocity_rms=scalar_rms(p.velocity[:,0]-velocity),
        max_internal_force_N=max_force,max_axial_acceleration_m_s2=max_acc,
        particle_cell_crossings=ncell_crossings,
        energy_initial_J=energy_initial,maximum_energy_relative_drift=energy_rel_max,
        stress_log_stretch_rms_pa=scalar_rms(p.stress[:,0]-MODULUS*np.log(J_total)),
        max_mass_rel_error=abs(p.mass.sum()-mass0)/mass0,
        max_momentum_abs_error=max_momentum_err,
        min_J=float(np.min(np.linalg.det(p.deformation_gradient))),
        min_characteristic_jacobian=minimum_characteristic_jacobian(s.time))
    return final,history,dict(X=X,J=J,stress=sigma,num_stress=p.stress[:,0].copy(),
                              num_J=(p.deformation_gradient[:,0,0]*stretch_initial(X)).copy())


def write_csv(path,rows):
    if not rows:return
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)


def validate(*, output='validation/results/stage10e', refinement=False,
             plot=True,dt=.003,ppc_x=2):
    if not np.isfinite(dt) or dt<=0 or not np.isclose(round(T_END/dt)*dt,T_END,atol=1e-11,rtol=0):
        raise ValueError('dt must be finite positive and exactly divide T_END=0.45 s')
    if ppc_x not in (2,4):
        raise ValueError('ppc_x must be 2 or 4')
    nsteps=round(T_END/dt)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    pairs=[('mpm',.25),('gimp',.25),('cpgimp',.25),('cpdi',.25)]
    if refinement:pairs += [('mpm',.5),('cpdi',.5),('mpm',.125),('cpdi',.125)]
    results=[];histories=[];profiles={}
    for method,h in pairs:
        print(f'Running {method} h={h:g} ...',flush=True)
        result,hist,profile=run_one(method,cell_dx=h,ppc_x=ppc_x,dt=dt,nsteps=nsteps)
        results.append(result);histories.extend(hist);profiles[(method,h)]=profile
        print('  stress error',round(result['final_stress_rel_l2'],5),'F error',round(result['final_stretch_rel_l2'],5),flush=True)
    checks={
        'independent_reference_pre_shock':minimum_characteristic_jacobian(T_END)>.1,
        'all_four_methods':len({r['method'] for r in results})>=4,
        'force_feedback_nonzero':all(r['max_axial_acceleration_m_s2']>1e-5 and r['max_internal_force_N']>1e-5 for r in results),
        'finite_positive_deformation':all(r['min_J']>0 and np.isfinite(r['final_stress_rel_l2']) for r in results),
        'mass_conserved':all(r['max_mass_rel_error']<1e-12 for r in results),
        'energy_drift_bounded':all(r['maximum_energy_relative_drift']<.10 for r in results),
        'stress_stretch_consistency':all(r['stress_log_stretch_rms_pa']<3. for r in results),
        'momentum_conserved':all(r['max_momentum_abs_error']<1e-6 for r in results),
        'finite_strain':all(r['reference_peak_stretch']>1.5 for r in results),
        'cpdi_stress_bounded':next(r['final_stress_rel_l2'] for r in results if r['method']=='cpdi')<.6,
    }
    if refinement:
        cpdis=sorted((r for r in results if r['method']=='cpdi'),key=lambda r:r['cell_dx_m'],reverse=True)
        checks['cpdi_refinement_decreases_stress_error']=(cpdis[-1]['final_stress_rel_l2'] < cpdis[0]['final_stress_rel_l2'])
    summary={'checks':checks,'reference':'exact finite-stretch Lagrangian nonlinear simple-wave characteristics, before shock',
             'parameters':{'young_pa':E,'nu':NU,'rho_kg_m3':RHO0,'initial_peak_J':1+AMP,'time_s':T_END,'dt_s':dt,'ppc_x':ppc_x,'full_reference_span_m':[X_LEFT,X_RIGHT]},
             'cases':results,
             'caveat':'Exact for plane-strain axial Jaumann hypoelastic reduction only, not general nonlinear elasticity or shear/rotation.'}
    write_csv(out/'10E_cases.csv',results);write_csv(out/'10E_history.csv',histories)
    (out/'10E_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    if plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(figsize=(7.2,4.3))
            for r in results:
                if r['cell_dx_m']!=.25:continue
                data=profiles[(r['method'],.25)]
                x=data['X']
                # Plot y=first layer only; remaining layers are identical in exact plane-strain motion.
                idx=np.arange(len(x))<int((X_RIGHT-X_LEFT)/(.25/ppc_x))
                ax.plot(x[idx],data['num_stress'][idx],label=r['method'],alpha=.82)
            d=profiles[('cpdi',.25)]
            idx=np.arange(len(d['X']))<int((X_RIGHT-X_LEFT)/(.25/ppc_x))
            ax.plot(d['X'][idx],d['stress'][idx],'k--',label='analytical',linewidth=2.2)
            ax.set(xlabel='Material coordinate X (m)',ylabel='Axial Cauchy stress (Pa)',
                   title='Force-driven finite-stretch nonlinear wave: t=0.45 s')
            ax.legend();fig.tight_layout();fig.savefig(out/'10E_stress_profile.png',dpi=170);plt.close(fig)
            fig,ax=plt.subplots(figsize=(6.8,4.2))
            for method in ('mpm','gimp','cpgimp','cpdi'):
                dat=[r for r in histories if r['method']==method and r['cell_dx_m']==.25]
                ax.plot([r['t_s'] for r in dat],[r['relative_stress_l2'] for r in dat],label=method)
            ax.set(xlabel='Time (s)',ylabel='Relative stress L2 error',title='Stress error during nonlinear wave propagation')
            ax.legend();fig.tight_layout();fig.savefig(out/'10E_error_history.png',dpi=170);plt.close(fig)
            if refinement:
                fig,ax=plt.subplots(figsize=(6.4,4.2))
                for method in ('mpm','cpdi'):
                    dat=sorted((r for r in results if r['method']==method),key=lambda r:r['cell_dx_m'])
                    ax.plot([r['cell_dx_m'] for r in dat],[r['final_stress_rel_l2'] for r in dat],'-o',label=method)
                ax.set(xlabel='Grid spacing h (m)',ylabel='Final relative stress error',title='Grid-refinement sensitivity')
                ax.legend();fig.tight_layout();fig.savefig(out/'10E_refinement.png',dpi=170);plt.close(fig)
        except ImportError:
            pass
    print(json.dumps(summary,indent=2))
    print('Results:',out)
    return summary


def main(argv=None):
    ap=argparse.ArgumentParser(description='Stage 10E finite-stretch force-driven benchmark')
    ap.add_argument('--refinement',action='store_true')
    ap.add_argument('--dt',type=float,default=.003)
    ap.add_argument('--ppc-x',type=int,choices=(2,4),default=2)
    ap.add_argument('--no-plot',action='store_true')
    ap.add_argument('--output',default='validation/results/stage10e')
    opts=ap.parse_args(argv)
    summary=validate(output=opts.output,refinement=opts.refinement,plot=not opts.no_plot,
                     dt=opts.dt,ppc_x=opts.ppc_x)
    return 0 if all(summary['checks'].values()) else 1

if __name__=='__main__':raise SystemExit(main())
