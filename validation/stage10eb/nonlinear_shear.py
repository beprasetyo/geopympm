"""Stage 10E-B: force-driven plane-strain nonlinear shear/rotation wave.

Reference is an exact pre-shock right-running simple wave of a *constrained*
Jaumann hypoelastic simple-shear reduction, rather than a generic 2-D solution.

The continuum reduction assumes v=(v(y,t),0), L_xy = dv/dy, gamma_t=dv/dy,
with sigma_xy=mu*sin(gamma), sigma_xx=mu*(1-cos(gamma)),
sigma_yy=-mu*(1-cos(gamma)), and  rho*v_t = (sigma_xy)_y.

Then gamma_t + c(gamma)*gamma_y=0, c=c0*sqrt(cos(gamma)),
while v = -integral_0^gamma c(s) ds.  Before characteristic intersections:
    y = xi + t*c(gamma0(xi)), gamma(y,t)=gamma0(xi).

The initial physical MPM coordinates are an *incremental reference* with
preloaded stresses corresponding to the reference gamma0. Accordingly
F_MPM(t=0)=I and reference F_MPM,xy = gamma(y,t)-gamma0(y), not gamma itself.
The initial shear prestress is admissible as initial data for the rate-form
constitutive law. The constrained y-motion supplies the necessary reaction
force from nonuniform normal stresses; therefore this is NOT a traction-free
unconstrained 2-D wave nor a general large-rotation proof.

A wide finite strip is used and errors are measured only in a middle-x window
before lateral free-surface disturbances reach the window. No x-direction
velocity or displacement boundary condition is applied during the run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.interpolation.cpdi import update_cpdi_domain
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver

RHO0 = 1200.0
E = 1200.0
NU = 0.25
MATERIAL = LinearElastic(E, NU)
MU = MATERIAL.shear_modulus
C0 = math.sqrt(MU / RHO0)
AMP = 1.1
WIDTH = .65
YC = 5.0
X0, X1 = 4.0, 10.0
Y0, Y1 = 2.0, 8.0
XC0, XC1 = 6.0, 8.0
T_END = .9
DT = .003
# Independent, high-order quadrature for the characteristic velocity.
_GAUSS, _WGHT = leggauss(48)


def gamma0(y):
    y = np.asarray(y, dtype=float)
    return AMP * np.exp(-.5 * ((y-YC)/WIDTH)**2)


def c(gamma):
    gamma = np.asarray(gamma, dtype=float)
    if np.any(np.abs(gamma) >= math.pi/2):
        raise ValueError('Reference loses hyperbolicity for |gamma| >= pi/2')
    return C0 * np.sqrt(np.cos(gamma))


def velocity_of_gamma(gamma):
    """Independent 48-point Gauss integration of -int_0^gamma c(g) dg."""
    gamma = np.asarray(gamma, dtype=float)
    g = .5 * gamma[..., None] * (_GAUSS + 1.)
    return -.5 * gamma * np.sum(_WGHT * c(g), axis=-1)


def characteristic_jacobian_min(t):
    yy = np.linspace(Y0-1., Y1+1., 16001)
    gg = gamma0(yy)
    gp = -gg*(yy-YC)/(WIDTH**2)
    dcdy = -C0 * np.sin(gg) / (2*np.sqrt(np.cos(gg))) * gp
    return float(np.min(1. + t*dcdy))


def characteristics(y,t):
    y=np.asarray(y,dtype=float)
    if t<0 or not np.isfinite(t):raise ValueError('t must be finite and >=0')
    if characteristic_jacobian_min(t)<=0:raise ValueError('pre-shock reference invalid')
    if t==0:return y.copy(),gamma0(y)
    lo=y-C0*t
    hi=y-c(AMP)*t
    for _ in range(50):
        mid=.5*(lo+hi)
        f=mid+c(gamma0(mid))*t-y
        hi=np.where(f>0,mid,hi)
        lo=np.where(f<=0,mid,lo)
    xi=.5*(lo+hi)
    return xi,gamma0(xi)


def reference(y,t):
    _,gg=characteristics(y,t)
    return {'gamma':gg,'v_x':velocity_of_gamma(gg),
            'sig_xy':MU*np.sin(gg),'sig_xx':MU*(1-np.cos(gg)),
            'sig_yy':-MU*(1-np.cos(gg)),
            'F_xy':gg-gamma0(y),
            'polar_angle_rad':-np.arctan((gg-gamma0(y))/2.)}


def make_case(method='cpdi', *, h=.25, ppc=2, dt=DT):
    if method not in ('mpm','gimp','cpgimp','cpdi'):
        raise ValueError('method must be mpm/gimp/cpgimp/cpdi')
    if ppc not in (2,4):raise ValueError('ppc must be 2 or 4')
    if h<=0 or dt<=0:raise ValueError('h/dt must be positive')
    dp=h/ppc
    nx=round((X1-X0)/dp); ny=round((Y1-Y0)/dp)
    if abs(nx*dp-(X1-X0))>1e-10 or abs(ny*dp-(Y1-Y0))>1e-10:
        raise ValueError('particle spacing must divide physical strip sides')
    xx=X0+(np.arange(nx)+.5)*dp
    yy=Y0+(np.arange(ny)+.5)*dp
    X,Y=np.meshgrid(xx,yy)
    centers=np.column_stack([X.ravel(),Y.ravel()])
    ga=gamma0(centers[:,1]); n=centers.shape[0]
    vel=np.column_stack((velocity_of_gamma(ga),np.zeros(n)))
    sigma=np.zeros((n,6))
    sigma[:,0]=MU*(1-np.cos(ga))
    sigma[:,1]=-MU*(1-np.cos(ga))
    sigma[:,3]=MU*np.sin(ga)
    vol=np.full(n,dp*dp)
    sides=np.full((n,2),dp)
    p=ParticleSet(centers,vol*RHO0,vol,velocity=vel,stress=sigma,domain_size=sides)
    # A generous fixed grid, with all y velocities constrained; x remains free.
    extent=14.
    nc=int(round(extent/h))
    if not np.isclose(nc*h,extent):raise ValueError('grid spacing must divide 14 m')
    g=BackgroundGrid((0.,0.),(h,h),(nc,nc))
    g.velocity_bc_mask[:,1]=True
    g.velocity_bc_value[:,1]=0.
    if method=='cpdi':update_cpdi_domain(p,g)
    s=ExplicitMPMSolver(p,g,MATERIAL,dt=dt,pic_fraction=0.,cfl=.35,
        interpolation=method,stress_update='corotational',
        kinematics_update='exponential',advection_update='affine_exponential')
    core=(centers[:,0]>=XC0)&(centers[:,0]<=XC1)
    return centers,core,p,g,s


def relative_l2(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return float(np.linalg.norm(a-b)/max(float(np.linalg.norm(b)),1e-14))


def evaluate_errors(initial,core,p,time):
    ref=reference(initial[core,1],time)
    pp=p.stress[core]
    fxy=p.deformation_gradient[core,0,1]
    # For the simple shear F=[[1,fxy],[0,1]], the polar angle is -atan(fxy/2).
    theta=np.arctan2(p.deformation_gradient[core,1,0]-fxy,
                     p.deformation_gradient[core,0,0]+p.deformation_gradient[core,1,1])
    return dict(shear_stress_relative_l2=relative_l2(pp[:,3],ref['sig_xy']),
        normal_stress_xx_relative_l2=relative_l2(pp[:,0],ref['sig_xx']),
        Fxy_absolute_rms=float(np.sqrt(np.mean((fxy-ref['F_xy'])**2))),
        rotation_absolute_rms_rad=float(np.sqrt(np.mean((theta-ref['polar_angle_rad'])**2))),
        velocity_rms_m_s=float(np.sqrt(np.mean((p.velocity[core,0]-ref['v_x'])**2))),
        reference_max_incremental_shear=float(np.max(np.abs(ref['F_xy']))),
        numerical_max_incremental_shear=float(np.max(np.abs(fxy))),
        reference_max_polar_rotation_deg=float(np.max(np.abs(ref['polar_angle_rad']))*180/math.pi),
        numerical_max_polar_rotation_deg=float(np.max(np.abs(theta))*180/math.pi),
        core_particles=int(np.count_nonzero(core)))


def run_one(method='cpdi', *, h=.25, ppc=2, dt=DT, nsteps=300,
            record_profile=False):
    initial,core,p,g,s=make_case(method,h=h,ppc=ppc,dt=dt)
    initial_mass=float(np.sum(p.mass))
    p_init=float(np.dot(p.mass,p.velocity[:,0]))
    old_cells=np.floor(p.position/h).astype(int)
    max_force=0.;max_a=0.;max_mom=0.;n_crossings=0
    max_a_mass_supported=0.; max_low_node_count=0; max_low_mass_fraction=0.
    peak_nodal_acceleration_detail={}
    core_ref_gamma0=gamma0(initial[core,1])
    core_mass=p.mass[core].copy()
    def reduced_core_energy():
        # Only a diagnostic of the exact constrained 1-D shear reduction;
        # not a general conserved energy for Jaumann hypoelasticity in 2-D.
        gt=core_ref_gamma0+p.deformation_gradient[core,0,1]
        e=0.5*p.velocity[core,0]**2 + MU/RHO0*(1.-np.cos(gt))
        return float(np.dot(core_mass,e))
    initial_core_energy=reduced_core_energy()
    core_energy_drift_max=0.
    samples=[]
    for k in range(nsteps):
        diag=s.step()
        max_force=max(max_force,float(np.max(np.linalg.norm(g.force_internal,axis=1))))
        nodal_abs_a=np.abs(g.acceleration[:,0])
        imax=int(np.argmax(nodal_abs_a))
        this_max_a=float(nodal_abs_a[imax])
        if this_max_a>max_a:
            positive=g.mass>0
            scale=float(np.sum(p.mass)/np.count_nonzero(positive))
            peak_nodal_acceleration_detail={
                'step':k+1,'node_id':imax,
                'node_x_m':float(g.node_coordinates[imax,0]),
                'node_y_m':float(g.node_coordinates[imax,1]),
                'nodal_mass_kg':float(g.mass[imax]),
                'nodal_fx_N':float(g.force_internal[imax,0]),
                'mass_fraction_of_characteristic_node':float(g.mass[imax]/scale),
            }
            max_a=this_max_a
        active_mass=g.mass[g.mass>0.]
        if active_mass.size:
            threshold=1e-12*float(np.median(active_mass))
            heavy=g.mass>threshold
            light=(g.mass>0.)&~heavy
            if np.any(heavy):
                max_a_mass_supported=max(max_a_mass_supported,float(np.max(np.abs(g.acceleration[heavy,0]))))
            max_low_node_count=max(max_low_node_count,int(np.count_nonzero(light)))
            max_low_mass_fraction=max(max_low_mass_fraction,
                float(np.sum(g.mass[light])/np.sum(active_mass)))
        core_energy_drift_max=max(core_energy_drift_max,
            abs(reduced_core_energy()-initial_core_energy)/initial_core_energy)
        max_mom=max(max_mom,abs(float(np.dot(p.mass,p.velocity[:,0]))-p_init))
        new_cells=np.floor(p.position/h).astype(int)
        n_crossings+=int(np.sum(np.any(new_cells!=old_cells,axis=1)))
        old_cells=new_cells
        if (k+1)%max(1,nsteps//9)==0 or k==0 or k==nsteps-1:
            samples.append(dict(method=method,grid_spacing_m=h,t_s=diag.time,
                                **evaluate_errors(initial,core,p,diag.time)))
    err=evaluate_errors(initial,core,p,s.time)
    res=dict(method=method,grid_spacing_m=h,ppc=ppc,n_particles=p.n_particles,
        steps=nsteps,dt_s=dt,time_s=s.time,**err,
        final_mass_rel_error=abs(float(np.sum(p.mass))-initial_mass)/initial_mass,
        max_x_momentum_abs_error=max_mom,max_internal_force_N=max_force,
        max_nodal_ax_m_s2=max_a, peak_nodal_acceleration=peak_nodal_acceleration_detail,
        max_mass_supported_nodal_ax_m_s2=max_a_mass_supported,
        max_near_massless_node_count=max_low_node_count,
        max_near_massless_mass_fraction=max_low_mass_fraction,
        max_reduced_core_energy_drift=core_energy_drift_max,
        particle_cell_crossings=n_crossings,
        min_detF=float(np.min(np.linalg.det(p.deformation_gradient))),
        max_detF_deviation=float(np.max(np.abs(np.linalg.det(p.deformation_gradient)-1.))),
        max_transverse_displacement_m=float(np.max(np.abs(p.position[:,1]-initial[:,1]))),
        core_max_detF_deviation=float(np.max(np.abs(np.linalg.det(p.deformation_gradient[core])-1.))))
    profile=None
    if record_profile:
        ref=reference(initial[core,1],s.time)
        # Average over x in core to obtain a single y profile.
        core_yy=initial[core,1]
        ys, inv=np.unique(core_yy,return_inverse=True)
        def avg(arr):return np.bincount(inv,weights=arr)/np.bincount(inv)
        profile={'y':ys,'exact_sxy':avg(ref['sig_xy']),
            'num_sxy':avg(p.stress[core,3]),'exact_Fxy':avg(ref['F_xy']),
            'num_Fxy':avg(p.deformation_gradient[core,0,1])}
    return res,samples,profile


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf8') as f:
        out=csv.DictWriter(f,list(rows[0]));out.writeheader();out.writerows(rows)


def validate(*,output='validation/results/stage10eb',refinement=False,
             dt=DT,t_end=T_END,ppc=2,plot=True):
    if not np.isfinite(dt) or dt<=0 or not np.isfinite(t_end) or t_end<=0:
        raise ValueError('positive dt and t_end required')
    steps=round(t_end/dt)
    if abs(steps*dt-t_end)>1e-10:raise ValueError('dt must divide t_end')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    todo=[('mpm',.25),('gimp',.25),('cpgimp',.25),('cpdi',.25)]
    if refinement:todo += [('cpdi',.5),('cpdi',.125)]
    results=[];histories=[];profiles={}
    for method,h in todo:
        print(f'Running {method}, h={h:g}, time={t_end:g} s...',flush=True)
        row,hist,profile=run_one(method,h=h,ppc=ppc,dt=dt,nsteps=steps,record_profile=True)
        results.append(row);histories.extend(hist);profiles[(method,h)]=profile
        print(f"  stress L2={row['shear_stress_relative_l2']:.5f}, Fxy RMS={row['Fxy_absolute_rms']:.5f}, rotation RMS={row['rotation_absolute_rms_rad']:.5f}",flush=True)
    main=[r for r in results if r['grid_spacing_m']==.25]
    checks={'pre_shock_reference':characteristic_jacobian_min(t_end)>.05,
        'four_interpolation_methods':len({r['method'] for r in main})==4,
        'force_driven_x_momentum':all(r['max_internal_force_N']>1e-5 and r['max_nodal_ax_m_s2']>1e-5 for r in main),
        'mass_conserved':all(r['final_mass_rel_error']<1e-12 for r in results),
        'axial_momentum_conserved':all(r['max_x_momentum_abs_error']<1e-6 for r in results),
        'constraint_y_satisfied':all(r['max_transverse_displacement_m']<1e-8 for r in results),
        'positive_volume':all(r['min_detF']>0 for r in results),
        'shear_and_rotation_observed':all(r['numerical_max_incremental_shear']>0.02 and r['numerical_max_polar_rotation_deg']>0.5 for r in main),
        'all_metrics_finite':all(np.isfinite(r['shear_stress_relative_l2']) and np.isfinite(r['Fxy_absolute_rms']) for r in results)}
    if refinement:
        rows=sorted((r for r in results if r['method']=='cpdi'),key=lambda r:r['grid_spacing_m'],reverse=True)
        checks['cpdi_refinement_improves_shear_stress']=(rows[-1]['shear_stress_relative_l2'] < rows[0]['shear_stress_relative_l2'])
    summary={'checks':checks,
        'reference':'exact pre-shock Jaumann constrained simple-shear characteristic wave',
        'parameters':{'E_pa':E,'nu':NU,'rho_kg_m3':RHO0,'peak_initial_gamma':AMP,'t_end_s':t_end,'dt_s':dt,'ppc':ppc,'center_x_window':[XC0,XC1],'min_characteristic_jacobian':characteristic_jacobian_min(t_end)},
        'cases':results,
        'caveat':'Constrained y-velocity and a one-dimensional wave embedded in 2D; the x-end regions are excluded. This is not validation of arbitrary 2D traction/shear loading, bending, or large rigid-body rotation.'}
    write_csv(out/'10EB_cases.csv',results)
    write_csv(out/'10EB_history.csv',histories)
    (out/'10EB_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    if plot:
        try:
            import matplotlib;matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(figsize=(7.5,4.5))
            for meth in ('mpm','gimp','cpgimp','cpdi'):
                d=profiles[(meth,.25)];ax.plot(d['y'],d['num_sxy'],label=meth,alpha=.8)
            d=profiles[('cpdi',.25)];ax.plot(d['y'],d['exact_sxy'],'k--',lw=2,label='analytical')
            ax.set(xlabel='Initial y coordinate (m)',ylabel='Shear stress σxy (Pa)',title=f'Force-driven nonlinear shear: t={t_end:g} s')
            ax.legend();fig.tight_layout();fig.savefig(out/'10EB_shear_stress_profile.png',dpi=160);plt.close(fig)
            fig,ax=plt.subplots(figsize=(7.5,4.5))
            for meth in ('mpm','gimp','cpgimp','cpdi'):
                d=profiles[(meth,.25)];ax.plot(d['y'],d['num_Fxy'],label=meth,alpha=.8)
            ax.plot(d['y'],d['exact_Fxy'],'k--',lw=2,label='analytical')
            ax.set(xlabel='Initial y coordinate (m)',ylabel='Incremental Fxy',title='Shear deformation and local polar rotation')
            ax.legend();fig.tight_layout();fig.savefig(out/'10EB_shear_deformation.png',dpi=160);plt.close(fig)
            if refinement:
                dat=sorted((r for r in results if r['method']=='cpdi'),key=lambda r:r['grid_spacing_m'])
                fig,ax=plt.subplots(figsize=(6.5,4))
                ax.plot([r['grid_spacing_m'] for r in dat],[r['shear_stress_relative_l2'] for r in dat],'-o')
                ax.set(xlabel='h (m)',ylabel='Relative shear stress L2 error',title='CPDI1 grid sensitivity')
                fig.tight_layout();fig.savefig(out/'10EB_refinement.png',dpi=160);plt.close(fig)
        except ImportError:pass
    print(json.dumps(summary,indent=2));print('Results:',out)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description='Stage 10E-B: force-driven Jaumann shear-wave benchmark')
    parser.add_argument('--refinement',action='store_true')
    parser.add_argument('--dt',type=float,default=DT)
    parser.add_argument('--t-end',type=float,default=T_END)
    parser.add_argument('--ppc',type=int,choices=(2,4),default=2)
    parser.add_argument('--no-plot',action='store_true')
    parser.add_argument('--output',default='validation/results/stage10eb')
    a=parser.parse_args(argv)
    out=validate(output=a.output,refinement=a.refinement,dt=a.dt,t_end=a.t_end,ppc=a.ppc,plot=not a.no_plot)
    return 0 if all(out['checks'].values()) else 1

if __name__=='__main__':raise SystemExit(main())
