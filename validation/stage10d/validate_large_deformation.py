"""Stage 10D: large-deformation verification of 2D CPDI1.

These are *kinematically driven* (all active grid-node velocities prescribed)
affine benchmarks. Internal-force feedback is intentionally suppressed at the
constrained DOFs to isolate domain geometry, deformation kinematics, and the
Stage-7 Jaumann stress integrator. They are NOT a nonlinear force-driven or
large-deformation landslide validation.

Independent references:
 - F(t) from an independent Taylor matrix-exponential power series, not
   the solver's closed-form Numba exponential integrator;
 - Cauchy stress from a high-order Gauss-Legendre quadrature of the exact
   constant-L Jaumann-rate solution, not the midpoint stress update;
 - particle-domain corners from the reference map x(t) +/- F(t) a0 +/- F(t) b0.

No optional dependencies are required for CSV/JSON or tests. Matplotlib is
optional and used only for PNG plots.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.interpolation.cpdi import candidate_cpdi_geometry, evaluate_cpdi
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver

CENTER = np.array([5.0, 5.0], dtype=float)
SIDE = 0.4
GRID_STEP = 0.5
INITIAL_STRESS = {
    'shear': (0.0, 0.0, 0.0, 0.0),
    'spin': (45.0, -25.0, 12.0, 8.0),
    'combined': (25.0, -12.0, 7.0, 5.0),
}


def prescribed_rate(mode: str) -> tuple[np.ndarray, float, float]:
    """Return stationary L, physical end time, and optional drift vx."""
    if mode == 'shear':
        return np.array([[0., 1.], [0., 0.]]), 2.0, 0.7  # gamma=2, drift
    if mode == 'spin':
        return np.array([[0., -math.pi/2], [math.pi/2, 0.]]), 2.0, 0.0  # 180 deg
    if mode == 'combined':
        # Non-commuting stretch/shear/spin, not merely isotropic expansion.
        return np.array([[0.35, -1.20], [0.80, -0.12]]), 1.0, 0.0
    raise ValueError(f'Unknown affine mode: {mode}')


def expm_reference(A: np.ndarray, t: float) -> np.ndarray:
    """Independent constant-coefficient 2x2 reference via Taylor expansion.

    A direct power-series evaluation works even for defective Jordan matrices
    (the simple-shear matrix cannot be diagonalized). It is independent of
    Stage-8's trace/traceless closed-form exponential.
    """
    B = np.asarray(A, dtype=np.float64)*t
    total = np.eye(2, dtype=np.float64)
    term = np.eye(2, dtype=np.float64)
    for n in range(1, 65):
        term = (term@B)/n
        total = total+term
        if np.max(np.abs(term))<1e-16:
            break
    return total


def stress_reference(A: np.ndarray, t: float, material: LinearElastic,
                     initial_stress: np.ndarray) -> np.ndarray:
    """Exact constant-L Jaumann hypoelastic Cauchy stress via Gauss quadrature.

    sigma(t)=R(t)sigma(0)R^T(t) + int_0^t R(s)(C:D)R^T(s) ds.
    For the out-of-plane plane-strain term: sigma_zz=initial+lambda tr(D)*t.
    The stress is NOT an independent finite-strain hyperelastic solution.
    """
    D = .5*(A+A.T)
    omega = .5*(A[1, 0]-A[0, 1])
    lam = material.lame_lambda
    mu = material.shear_modulus
    source = 2.*mu*D + lam*np.trace(D)*np.eye(2)
    def rotation(s):
        c, sn = math.cos(omega*s), math.sin(omega*s)
        return np.array([[c, -sn], [sn, c]])
    Q = rotation(t)
    sigma0 = np.array([[initial_stress[0], initial_stress[2]],
                       [initial_stress[2], initial_stress[1]]])
    sigma = Q @ sigma0 @ Q.T
    # 32 nodes achieve ~machine precision for the tested rotations.
    pts, weights = np.polynomial.legendre.leggauss(32)
    for z, w in zip(pts, weights):
        s = .5*t*(z+1.)
        R = rotation(s)
        sigma += .5*t*w*(R @ source @ R.T)
    zz = initial_stress[3] + lam*np.trace(D)*t
    return np.array([sigma[0, 0], sigma[1, 1], zz, sigma[0, 1], 0., 0.])


def make_patch(mode: str, *, n_side: int = 4, dt: float = .01,
               method: str = 'cpdi', grid_step: float = GRID_STEP):
    """Contiguous square particle domains; prescribed affine velocity field."""
    A, t_end, drift = prescribed_rate(mode)
    if n_side < 2:
        raise ValueError('Patch requires n_side >= 2.')
    xy = (np.arange(n_side)-(n_side-1)*.5)*SIDE
    X = np.array([[CENTER[0]+x, CENTER[1]+y] for y in xy for x in xy], dtype=float)
    n = len(X)
    vol = np.full(n, SIDE**2)
    p = ParticleSet(X, mass=vol*1200., volume=vol,
                    velocity=(X-CENTER) @ A.T + np.array([drift, 0.]),
                    domain_size=np.full((n,2), SIDE))
    sig0 = INITIAL_STRESS[mode]
    p.stress[:, [0, 1, 3, 2]] = sig0
    count = round(12./grid_step)
    if not np.isclose(count*grid_step, 12., rtol=0., atol=1e-12):
        raise ValueError('grid_step must divide 12 m exactly.')
    grid = BackgroundGrid((0.,0.), (grid_step, grid_step), (count,count))
    grid.velocity_bc_mask[:] = True
    grid.velocity_bc_value[:] = (grid.node_coordinates-CENTER) @ A.T + np.array([drift,0.])
    material = LinearElastic(1200., .25)
    solver = ExplicitMPMSolver(
        p, grid, material, dt=dt, cfl=.65, pic_fraction=0.,
        interpolation=method, stress_update='corotational',
        kinematics_update='exponential', advection_update='affine_exponential',
    )
    return X, p, grid, solver, A, t_end, drift, material


def _rectangles(position: np.ndarray, full_width: np.ndarray) -> np.ndarray:
    """Axis-aligned GIMP/cpGIMP four corner coordinates, counterclockwise."""
    d = .5*full_width
    return np.stack([position + np.array([-1,-1])*d,
                     position + np.array([ 1,-1])*d,
                     position + np.array([ 1, 1])*d,
                     position + np.array([-1, 1])*d], axis=1)


def shared_edge_mismatch(corners: np.ndarray, n_side: int) -> float:
    """Largest shared-edge endpoint separation between initially adjacent domains."""
    error = 0.
    for j in range(n_side):
        for i in range(n_side):
            p = j*n_side + i
            if i+1 < n_side:
                q = p+1
                error = max(error, float(np.linalg.norm(corners[p, 1]-corners[q, 0])),
                            float(np.linalg.norm(corners[p, 2]-corners[q, 3])))
            if j+1 < n_side:
                q = p+n_side
                error = max(error, float(np.linalg.norm(corners[p, 3]-corners[q, 0])),
                            float(np.linalg.norm(corners[p, 2]-corners[q, 1])))
    return error


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.
    return .5*abs(float(np.dot(poly[:,0],np.roll(poly[:,1],-1)) -
                          np.dot(poly[:,1],np.roll(poly[:,0],-1))))


def overlap_area_convex(P: np.ndarray, Q: np.ndarray) -> float:
    """Convex polygon overlap using Sutherland-Hodgman; no Shapely dependency.

    Both input polygons are ordered counterclockwise. Numerical area below
    1e-12 is ignored to prevent adjacent shared edges from appearing as overlap.
    """
    poly = [np.asarray(x, dtype=float) for x in P]
    tol = 1e-12
    for j in range(len(Q)):
        a, b = Q[j], Q[(j+1)%len(Q)]
        e = b-a
        def side(x):
            v = x-a
            return e[0]*v[1]-e[1]*v[0]
        if not poly:
            return 0.
        old = poly
        poly = []
        previous = old[-1]
        previous_side = side(previous)
        for current in old:
            current_side = side(current)
            if (previous_side >= -tol) != (current_side >= -tol):
                denom = previous_side - current_side
                if abs(denom)>1e-20:
                    alpha = previous_side/denom
                    poly.append(previous+alpha*(current-previous))
            if current_side >= -tol:
                poly.append(current)
            previous, previous_side = current, current_side
    area = polygon_area(np.array(poly))
    return area if area>tol else 0.


def pairwise_overlap_area(corners: np.ndarray) -> float:
    """Sum of pairwise overlaps (not exact union overlap for triple intersections)."""
    acc=0.
    for i in range(len(corners)):
        for j in range(i+1,len(corners)):
            # AABB broad-phase rejection
            if np.any(corners[i].max(axis=0)<corners[j].min(axis=0)-1e-12) or np.any(
                corners[j].max(axis=0)<corners[i].min(axis=0)-1e-12):
                continue
            acc += overlap_area_convex(corners[i],corners[j])
    return float(acc)


def expected_positions(X: np.ndarray, A: np.ndarray, t: float, drift: float):
    F=expm_reference(A,t)
    # particular solution for constant drift: integral_0^t exp(A s) U ds
    U=np.array([drift,0.])
    if drift == 0.:
        shift=np.zeros(2)
    else:
        pts,w=np.polynomial.legendre.leggauss(32)
        shift=sum(.5*t*wi*(expm_reference(A,.5*t*(zi+1.))@U)
                  for zi,wi in zip(pts,w))
    return CENTER+(X-CENTER)@F.T+shift,F


def reference_corners(position: np.ndarray, F: np.ndarray) -> np.ndarray:
    # Reference domain axes in initial square patch; F is global affine.
    a=F[:,0]*(SIDE*.5)
    b=F[:,1]*(SIDE*.5)
    return np.stack([position-a-b,position+a-b,position+a+b,position-a+b],axis=1)


def run_affine(mode: str, *, dt: float = .01, n_side: int = 4,
               grid_step: float = GRID_STEP):
    X,p,grid,solver,A,t_end,drift,mat=make_patch(mode,n_side=n_side,dt=dt,
                                                 grid_step=grid_step)
    steps=round(t_end/dt)
    if not np.isclose(steps*dt,t_end,rtol=0,atol=1e-12):
        raise ValueError('dt must divide end time exactly for this reproducible benchmark.')
    history=[]
    initial_mass=float(p.mass.sum())
    sig0=np.array(INITIAL_STRESS[mode],dtype=float)
    for step in range(1,steps+1):
        solver.step()
        if step % max(1,steps//20) != 0 and step != steps:
            continue
        t=step*dt
        ref_pos,ref_F=expected_positions(X,A,t,drift)
        ref_stress=stress_reference(A,t,mat,sig0)
        actual_corners,_,_,_=candidate_cpdi_geometry(p,grid)
        ref_corners=reference_corners(ref_pos,ref_F)
        sigma=p.stress
        numerical_sig=sigma[:,[0,1,2,3]]
        stress_err=float(np.max(np.abs(numerical_sig-ref_stress[:4])))
        edge_err=shared_edge_mismatch(actual_corners,n_side)
        # Geometry between neighbors is independent from particle volume bookkeeping.
        J=float(np.linalg.det(ref_F))
        domain_area=np.array([polygon_area(c) for c in actual_corners])
        record={
            'mode':mode,'step':step,'time_s':float(t),
            'max_F_error':float(np.max(np.abs(p.deformation_gradient-ref_F))),
            'max_position_error_m':float(np.max(np.linalg.norm(p.position-ref_pos,axis=1))),
            'max_corner_error_m':float(np.max(np.linalg.norm(actual_corners-ref_corners,axis=2))),
            'max_stress_error_pa':stress_err,
            'reference_stress_norm_pa':float(np.linalg.norm(ref_stress[:4])),
            'J_reference':J,
            'max_J_error':float(np.max(abs(np.linalg.det(p.deformation_gradient)-J))),
            'max_volume_relative_error':float(np.max(abs(p.volume/(SIDE**2*J)-1.))),
            'max_domain_area_relative_error':float(np.max(abs(domain_area/p.volume-1.))),
            'shared_edge_mismatch_m':edge_err,
            'mass_relative_error':abs(float(p.mass.sum())-initial_mass)/initial_mass,
            'maximum_stretch_ratio':float(max(np.linalg.svd(ref_F,compute_uv=False))),
        }
        history.append(record)
    end=history[-1].copy()
    end.update({'method':'cpdi','n_side':n_side,'dt_s':dt,'grid_step_m':grid_step,
                'particle_cell_index_changes':int(np.count_nonzero(np.floor((p.position-grid.origin)/grid.cell_size) !=
                                         np.floor((X-grid.origin)/grid.cell_size)))})
    return end,history


def geometry_comparison(*, n_side: int = 4, shear: float = 2.0):
    """All methods use the SAME analytical centers and F at large shear.

    uGIMP/cpGIMP geometries are rectangles (same for pure shear). MPM has
    no explicit particle interpolation domains; thus not part of this metric.
    """
    A=np.array([[0.,1.],[0.,0.]])
    xy=(np.arange(n_side)-(n_side-1)*.5)*SIDE
    X=np.array([[CENTER[0]+x,CENTER[1]+y] for y in xy for x in xy])
    position,F=expected_positions(X,A,shear,0.)
    cpdi=reference_corners(position,F)
    u=_rectangles(position,np.full_like(position,SIDE))
    cp=_rectangles(position,np.tile(SIDE*np.diag(F),(len(X),1)))
    return {mode:{'edge_mismatch_m':shared_edge_mismatch(verts,n_side),
                  'pairwise_overlap_area_m2':pairwise_overlap_area(verts),
                  'area_relative_error':float(abs(sum(polygon_area(v) for v in verts)-
                      (SIDE**2*len(verts)*np.linalg.det(F)))/
                      (SIDE**2*len(verts)*np.linalg.det(F)))}
            for mode,verts in [('ugimp',u),('cpgimp',cp),('cpdi',cpdi)]}


def nonaffine_domain_diagnostic(*, n_side: int = 4, coefficient: float = 1.0):
    """Locally affine CPDI polygons for the smooth nonlinear map

    x'=x+k*(x-cx)*(y-cy)^2, y'=y. Its varying deformation gradient
    F=[[1+k*v^2, 2*k*u*v],[0,1]] remains orientation preserving.
    The local linear approximation can generate BOTH edge mismatch and
    polygon overlaps although the exact nonlinear mapping is continuous.
    """
    xy=(np.arange(n_side)-(n_side-1)*.5)*SIDE
    X=np.array([[CENTER[0]+x,CENTER[1]+y] for y in xy for x in xy])
    u=X[:,0]-CENTER[0]
    v=X[:,1]-CENTER[1]
    pos=X.copy();pos[:,0]+=coefficient*u*v*v
    F=np.tile(np.eye(2),(len(X),1,1))
    F[:,0,0]+=coefficient*v*v
    F[:,0,1]=2*coefficient*u*v
    a=F[:,:,0]*(SIDE*.5);b=F[:,:,1]*(SIDE*.5)
    corners=np.stack([pos-a-b,pos+a-b,pos+a+b,pos-a+b],axis=1)
    return {'edge_mismatch_m':shared_edge_mismatch(corners,n_side),
            'pairwise_overlap_area_m2':pairwise_overlap_area(corners),
            'comment':'Local affine CPDI1 polygons approximate a nonlinear x+ k*x*y^2 map.'}


def run_refinement(output, *, n_side: int = 4):
    """Repeat shear using smaller dt and multiple grid sizes.

    For homogeneous imposed affine velocity, grid-size independence tests
    consistency only; it is NOT a force-driven spatial convergence study.
    """
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rows=[]
    for dt in (.02,.01,.005):
        result,_=run_affine('shear',dt=dt,n_side=n_side,grid_step=.5)
        rows.append({'study':'time','parameter':dt,**result})
    for h in (1.,.5,.25):
        result,_=run_affine('shear',dt=.01,n_side=n_side,grid_step=h)
        rows.append({'study':'grid','parameter':h,**result})
    with (out/'10D_refinement.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    time_rows=[r for r in rows if r['study']=='time']
    grid_rows=[r for r in rows if r['study']=='grid']
    checks={
        'temporal_stress_error_decreases':time_rows[0]['max_stress_error_pa']>
            time_rows[1]['max_stress_error_pa']>time_rows[2]['max_stress_error_pa'],
        'second_order_midpoint_stress_trend':(
            time_rows[0]['max_stress_error_pa']/time_rows[1]['max_stress_error_pa']>3.5
            and time_rows[1]['max_stress_error_pa']/time_rows[2]['max_stress_error_pa']>3.5),
        'grid_affine_kinematics_consistent':max(r['max_F_error'] for r in grid_rows)<5e-10,
        'grid_affine_stress_consistent':max(r['max_stress_error_pa'] for r in grid_rows)-
            min(r['max_stress_error_pa'] for r in grid_rows)<1e-7,
    }
    record={'checks':checks,'temporal':[{'dt_s':r['dt_s'],
         'max_stress_error_pa':r['max_stress_error_pa']} for r in time_rows],
         'grid':[{'cell_size_m':r['grid_step_m'],
         'max_F_error':r['max_F_error'],
         'max_stress_error_pa':r['max_stress_error_pa']} for r in grid_rows],
         'note':'These are prescribed affine-velocity tests, not dynamic spatial convergence.'}
    (out/'10D_refinement_summary.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    return record


def run(output='validation/results/stage10d', *, dt: float = .01,
        n_side: int = 4, grid_step: float = GRID_STEP, plot: bool = True):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    end_records=[]; history=[]
    for mode in ('shear','spin','combined'):
        rec,h=run_affine(mode,dt=dt,n_side=n_side,grid_step=grid_step)
        end_records.append(rec);history.extend(h)
    geometry=geometry_comparison(n_side=n_side)
    nonaffine=nonaffine_domain_diagnostic(n_side=n_side)
    # Guardrails validate independent reference, geometric integrity and finite stress.
    checks={
        'large_shear_gamma_2':end_records[0]['maximum_stretch_ratio']>2.,
        'half_turn_rotation':bool(np.isclose(expm_reference(prescribed_rate('spin')[0],2.)[0,0],-1.,atol=1e-12)),
        'large_affine_F':max(r['max_F_error'] for r in end_records)<5e-10,
        'large_affine_position':max(r['max_position_error_m'] for r in end_records)<5e-10,
        'large_affine_corners':max(r['max_corner_error_m'] for r in end_records)<5e-10,
        'finite_stress_reference_error':all(np.isfinite(r['max_stress_error_pa']) for r in end_records),
        'stress_accuracy':max(r['max_stress_error_pa'] for r in end_records)<1.0,
        'CPDI_conforming_affine_patch':max(r['shared_edge_mismatch_m'] for r in end_records)<1e-9,
        'CPDI_area_matches_volume':max(r['max_domain_area_relative_error'] for r in end_records)<1e-9,
        'mass_conserved':max(r['mass_relative_error'] for r in end_records)<1e-12,
        'particle_cell_crossing_detected':end_records[0]['particle_cell_index_changes']>0,
        'nonaffine_mismatch_detected':nonaffine['edge_mismatch_m']>1e-6,
        'nonaffine_overlap_detected':nonaffine['pairwise_overlap_area_m2']>1e-6,
        'ugimp_shear_geometry_mismatch':geometry['ugimp']['edge_mismatch_m']>1e-3,
    }
    summary={'stage':'10D','benchmark':'kinematically driven large-affine CPDI1',
             'parameters':{'dt_s':dt,'n_side':n_side,'particle_domain_full_side_m':SIDE,'grid_step_m':grid_step},
             'checks':checks,'affine_end':{r['mode']:r for r in end_records},
             'large_shear_geometry':geometry,
             'nonaffine_local_domain_warning':nonaffine,
             'limitations':[
                 'All active grid-node velocities constrained: no internal-force feedback.',
                 'Stress reference is the objective Jaumann hypoelastic rate, NOT a finite-strain hyperelastic law.',
                 'No force-driven finite-strain/landslide-runout benchmark performed.',
                 'Only prescribed-affine grid/particle resolution checks; no dynamic wave convergence or plasticity/contact.',
             ]}
    with (out/'10D_affine_history.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    with (out/'10D_end_summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(end_records[0]));writer.writeheader();writer.writerows(end_records)
    (out/'10D_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            for metric, title, filename in [
                ('max_F_error','F error','10D_F_error.png'),
                ('max_stress_error_pa','Stress error (Pa)','10D_stress_error.png'),
            ]:
                fig,ax=plt.subplots(figsize=(7,4))
                for mode in ('shear','spin','combined'):
                    rr=[r for r in history if r['mode']==mode]
                    ax.plot([r['time_s'] for r in rr],
                            [r[metric] for r in rr],label=mode)
                ax.set(xlabel='Time (s)',ylabel=title,title='Stage 10D: '+title)
                ax.legend();fig.tight_layout();fig.savefig(out/filename,dpi=160);plt.close(fig)
            # For visual inspection use the exact reference geometry at gamma=2.
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            xy=(np.arange(n_side)-(n_side-1)*.5)*SIDE
            X=np.array([[CENTER[0]+x,CENTER[1]+y] for y in xy for x in xy])
            pos,F=expected_positions(X,np.array([[0.,1.],[0.,0.]]),2.,0.)
            for ax, method, vertices in [
                (axes[0],'uGIMP',_rectangles(pos,np.full_like(pos,SIDE))),
                (axes[1],'CPDI1',reference_corners(pos,F)),
            ]:
                for c in vertices:
                    z=np.vstack((c,c[0]));ax.plot(z[:,0],z[:,1],lw=.9)
                ax.set(xlabel='x (m)',ylabel='y (m)',title=method+' at shear gamma=2')
                ax.axis('equal');ax.grid(alpha=.2)
            fig.tight_layout();fig.savefig(out/'10D_domain_comparison.png',dpi=160);plt.close(fig)
        except ImportError:
            pass
    print(json.dumps(summary,indent=2))
    print('Results:',out)
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description='Stage 10D large-affine CPDI1 verification')
    p.add_argument('--dt',type=float,default=.01)
    p.add_argument('--n-side',type=int,default=4)
    p.add_argument('--cell-size',type=float,default=GRID_STEP)
    p.add_argument('--output',default='validation/results/stage10d')
    p.add_argument('--no-plot',action='store_true')
    p.add_argument('--refinement',action='store_true')
    a=p.parse_args(argv)
    if not (0<a.dt<=.02) or a.n_side<2 or a.cell_size<=0:
        p.error('Require 0 < --dt <= 0.02 and --n-side >= 2.')
    result=run(a.output,dt=a.dt,n_side=a.n_side,grid_step=a.cell_size,plot=not a.no_plot)
    passed=all(result['checks'].values())
    if a.refinement:
        refine=run_refinement(a.output,n_side=a.n_side)
        print('Stage 10D refinement:',json.dumps(refine,indent=2))
        passed=passed and all(refine['checks'].values())
    return 0 if passed else 1

if __name__=='__main__':
    raise SystemExit(main())
