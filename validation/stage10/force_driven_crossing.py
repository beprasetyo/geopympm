"""Stage 10A-FD: genuinely force-driven elastic stress wave + cell crossing.

A small-amplitude right-going Gaussian displacement pulse propagates in a
homogeneous plane-strain bar while the complete bar is optionally translated
by a constant drift. Its infinite-bar linear-elastic d'Alembert solution is
independent of the GeoPyMPM P2G/G2P kernels (analytic stress/position/velocity
in MATERIAL coordinates). Both runs use the same stress initialization.

Importantly: only transverse grid velocities are constrained. Interior x
velocities are NOT prescribed. Particle stress -> nodal internal forces ->
nodal acceleration -> particle kinematics/stress/velocity/position.

This is a small-strain diagnostic, not a nonlinear finite-deformation exact
solution, and the comparison remains informative only before outgoing waves
reach physical bar ends. No a-priori superiority assertion is made for uGIMP.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver


# SI units and plane strain; chosen so wave does not reach the specimen ends.
L = 10.0
BAR_LEFT = 1.0
X_CENTER = 5.0
WIDTH = 0.42
AMP = 0.002
RHO = 1200.0
YOUNG = 1200.0
NU = 0.25
LAME = YOUNG * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
M = YOUNG * (1.0 - NU) / ((1.0 + NU) * (1.0 - 2.0 * NU))
WAVE_SPEED = np.sqrt(M / RHO)
DOMAIN_X = 14.0
DX = 0.25
DY = 0.25
NY = 2
STEP_DT = 0.002
STEPS = 450
PPC_X = 4
PPC_Y = 2
DRIFT = 1.0


def gaussian(x):
    z = np.asarray(x, dtype=float) / WIDTH
    return np.exp(-0.5 * z**2)


def gaussian_dx(x):
    return -np.asarray(x, dtype=float) * gaussian(x) / WIDTH**2


def analytical(X, t, drift):
    """Linear infinite-bar travelling wave; material labels X are fixed."""
    phase = X - X_CENTER - WAVE_SPEED * t
    strain_xx = AMP * gaussian_dx(phase)
    displacement = drift * t + AMP * gaussian(phase)
    v_x = drift - WAVE_SPEED * AMP * gaussian_dx(phase)
    sigma_xx = M * strain_xx
    return X + displacement, v_x, sigma_xx


def make_case(method, drift=DRIFT, ppc_x=PPC_X, *, dt=STEP_DT, cell_dx=DX,
              domain_factor=1.0, pic_fraction=0.0):
    if method not in ('mpm', 'gimp'):
        raise ValueError('method must be mpm or gimp')
    if ppc_x not in (2, 4, 8):
        raise ValueError('ppc_x must be 2, 4 or 8')
    if not (np.isfinite(cell_dx) and cell_dx in (0.125, 0.25, 0.5)):
        raise ValueError('cell_dx must be 0.125, 0.25, or 0.5 for this benchmark')
    if not (np.isfinite(domain_factor) and 0.0 < domain_factor <= ppc_x):
        raise ValueError('domain_factor must be positive and <= ppc_x (GIMP width <= cell).')
    if not (np.isfinite(pic_fraction) and 0.0 <= pic_fraction <= 1.0):
        raise ValueError('pic_fraction must be between 0 and 1.')
    if abs(drift) > 1.5:
        raise ValueError('drift exceeds supported validation window')
    xs = BAR_LEFT + (np.arange(round(L/cell_dx)*ppc_x)+0.5)*(cell_dx/ppc_x)
    ys = (np.arange(NY*PPC_Y)+0.5)*(DY/PPC_Y)
    X = np.array([(x, y) for y in ys for x in xs], dtype=float)
    n = len(X)
    V0 = cell_dx*DY/(ppc_x*PPC_Y)
    x0, v0, s0 = analytical(X[:,0], 0.0, drift)
    xi = X.copy()
    xi[:,0] = x0
    p = ParticleSet(
        xi, np.full(n, RHO*V0), np.full(n, V0),
        velocity=np.column_stack((v0, np.zeros(n))),
        domain_size=np.tile((cell_dx/ppc_x * domain_factor, DY/PPC_Y), (n, 1)),
    )
    exx = s0 / M
    p.deformation_gradient[:,0,0] = 1.0 + exx
    p.volume[:] = V0 * (1.0 + exx)
    p.strain[:,0] = exx
    p.stress[:,0] = s0
    p.stress[:,1] = LAME*exx
    p.stress[:,2] = LAME*exx

    grid = BackgroundGrid((0.0,0.0),(cell_dx,DY),(round(DOMAIN_X/cell_dx),NY))
    # Constrain ONLY y velocity (plane-strain uniaxial-wave kinematics).
    grid.velocity_bc_mask[:,1] = True
    solver = ExplicitMPMSolver(p, grid, LinearElastic(YOUNG,NU),
                               dt=dt, pic_fraction=pic_fraction,
                               interpolation=method)
    return X, p, grid, solver


def rms(x):
    a = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(a*a)))


def one_run(method, *, drift=DRIFT, ppc_x=PPC_X, steps=STEPS,
            dt=STEP_DT, sample_every=5, cell_dx=DX,
            domain_factor=1.0, pic_fraction=0.0):
    X, p, grid, solver = make_case(method, drift, ppc_x, dt=dt, cell_dx=cell_dx,
                                    domain_factor=domain_factor, pic_fraction=pic_fraction)
    sig_scale = rms(analytical(X[:,0], 0.0, drift)[2])
    if sig_scale <= 0:
        raise ValueError('Nonzero reference stress required')
    # Group a material point close to an x-grid line at each time, for an
    # objective crossing statistic based on each fixed material label.
    prev_cell = np.floor(p.position[:,0]/cell_dx).astype(int)
    crossing_events = 0
    # Per-particle temporal stress-error increment (not merely stress RMS).
    # These diagnostics separate instants when a material point crosses a
    # grid line from other instants, without treating the correlation as proof
    # of causation. Retain the entire cross/non-cross sample sets only as
    # squared sums and counts to avoid large diagnostics allocations.
    cross_error_squares = 0.0
    cross_error_count = 0
    no_cross_error_squares = 0.0
    no_cross_error_count = 0
    prev_error = p.stress[:,0] - analytical(X[:,0], 0.0, drift)[2]
    peak_reference_stress = M * AMP / WIDTH * np.exp(-0.5)
    first_mass = float(p.mass.sum())
    initial_momentum_x = float(p.mass @ p.velocity[:,0])
    momentum_scale = max(abs(initial_momentum_x),
                         first_mass * WAVE_SPEED * AMP / WIDTH)
    # zero-velocity reference used to check whether gravity or x-BC sneak in
    assert not np.any(grid.velocity_bc_mask[:,0])
    rows, probes = [], []
    # Plot one y-row only: duplicate x-labels from multiple identical y-rows
    # create misleading zig-zag lines even when the underlying 2D stress is
    # consistent; choose a reproducible x-material-point subset.
    ix = np.arange(len(X))
    particles_in_first_row = round(L/cell_dx)*ppc_x
    sampled = (ix < particles_in_first_row) & (ix % max(1,ppc_x//2) == 0)
    def record(i):
        t = i*dt
        xr, vr, sr = analytical(X[:,0], t, drift)
        se = p.stress[:,0] - sr
        verror = p.velocity[:,0] - vr
        xerror = p.position[:,0] - xr
        force_norm = float(np.linalg.norm(grid.force_internal[:,0]))
        rows.append(dict(method=method,drift_m_s=drift,ppc_x=ppc_x,cell_dx_m=cell_dx,
             domain_factor=domain_factor,pic_fraction=pic_fraction,
             step=i,time_s=t,n_crossings=crossing_events,
             stress_rms_error_pa=rms(se),stress_relative_l2=rms(se)/sig_scale,
             stress_max_error_pa=float(np.max(np.abs(se))),
             velocity_rms_error_m_s=rms(verror),position_rms_error_m=rms(xerror),
             momentum_x_relative_error=float(abs((p.mass@p.velocity[:,0])-initial_momentum_x)/
               momentum_scale),
             mass_relative_error=float(abs(p.mass.sum()-first_mass)/first_mass),
             nodal_force_l2_n=force_norm,
             max_particle_cell_move=float(np.max(np.abs(p.position[:,0]-X[:,0]-drift*t)/cell_dx))))
        if i in (0, steps//2, steps):
            for j in np.flatnonzero(sampled):
                probes.append(dict(method=method,drift_m_s=drift,step=i,time_s=t,
                    material_x_m=float(X[j,0]),current_x_m=float(p.position[j,0]),
                    stress_pa=float(p.stress[j,0]),analytic_stress_pa=float(sr[j]),
                    stress_error_pa=float(se[j])))
    record(0)
    for i in range(1, steps+1):
        solver.step()
        new_cell=np.floor(p.position[:,0]/cell_dx).astype(int)
        crossed = new_cell != prev_cell
        crossing_events += int(np.count_nonzero(crossed))
        ref_step = analytical(X[:,0],i*dt,drift)[2]
        current_error = p.stress[:,0] - ref_step
        increment_error = current_error - prev_error
        signal_active = np.abs(ref_step) >= (0.05*peak_reference_stress)
        cross_active = crossed & signal_active
        noncross_active = ~crossed & signal_active
        cross_error_squares += float(np.sum(increment_error[cross_active]**2))
        cross_error_count += int(np.count_nonzero(cross_active))
        no_cross_error_squares += float(np.sum(increment_error[noncross_active]**2))
        no_cross_error_count += int(np.count_nonzero(noncross_active))
        prev_error = current_error
        prev_cell=new_cell
        if i % sample_every == 0 or i==steps:
            record(i)
    summary = dict(method=method,drift_m_s=drift,ppc_x=ppc_x,cell_dx_m=cell_dx,
        domain_factor=domain_factor,pic_fraction=pic_fraction,
        n_particles=len(X),steps=steps,dt_s=dt,total_time_s=steps*dt,
        linear_wave_speed_m_s=float(WAVE_SPEED),
        stress_reference='dAlembert infinite homogeneous bar, small-strain Gaussian right-going wave',
        force_feedback=True,velocity_bc_only_y=True,
        n_particle_cell_crossings=crossing_events,
        stress_increment_crossing_rms_pa=(float(np.sqrt(cross_error_squares/cross_error_count))
                                         if cross_error_count else None),
        stress_increment_non_crossing_rms_pa=(float(np.sqrt(no_cross_error_squares/no_cross_error_count))
                                             if no_cross_error_count else None),
        crossing_stress_active_samples=cross_error_count,
        non_crossing_stress_active_samples=no_cross_error_count,
        final_stress_relative_l2=rows[-1]['stress_relative_l2'],
        time_rms_stress_relative_l2=rms([r['stress_relative_l2'] for r in rows]),
        max_stress_relative_l2=max(r['stress_relative_l2'] for r in rows),
        time_rms_stress_error_pa=rms([r['stress_rms_error_pa'] for r in rows]),
        final_stress_rms_error_pa=rows[-1]['stress_rms_error_pa'],
        max_mass_relative_error=max(r['mass_relative_error'] for r in rows),
        max_momentum_x_relative_error=max(r['momentum_x_relative_error'] for r in rows))
    # Retain stresses indexed by X (same labels) to assess Galilean invariance.
    return rows, probes, summary, p.stress[:,0].copy(), X[:,0].copy()


def save_csv(filename, rows):
    if rows:
        with open(filename,'w',encoding='utf8',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)


def make_plots(out, curves, snapshots, steps):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    pathlist=[]
    fig,ax=plt.subplots(figsize=(7.4,4.1))
    for method in ('mpm','gimp'):
        for drift in (0.0,DRIFT):
            r=curves[(method,drift)]
            ax.plot([v['time_s'] for v in r], [v['stress_relative_l2'] for v in r],
                label=f'{method.upper()} drift={drift:g} m/s',
                ls='-' if drift else '--', lw=1.5)
    ax.set(title='Force-driven travelling elastic wave: stress error vs analytic reference',
           xlabel='Time (s)', ylabel='Relative L2 stress error')
    ax.legend(); fig.tight_layout()
    path=out/'10A_FD_stress_error.png'; fig.savefig(path,dpi=160);pathlist.append(str(path));plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.4,4.1))
    for method in ('mpm','gimp'):
        sel=[v for v in snapshots if v['method']==method and v['drift_m_s']==DRIFT and
             v['step']==steps]
        if sel:
            ax.plot([v['material_x_m'] for v in sel], [v['stress_pa'] for v in sel],
                     marker='.',ls='-',lw=.9,label=f'{method.upper()}')
    sel=[v for v in snapshots if v['method']=='mpm' and v['drift_m_s']==DRIFT and
         v['step']==steps]
    if sel:
        ax.plot([v['material_x_m'] for v in sel],[v['analytic_stress_pa'] for v in sel],
                 ls='--',lw=1.5,color='black',label='Analytic stress')
    ax.set(title='Material-coordinate stress at final time, force-driven drifting bar',
          xlabel='Initial material coordinate X (m)',ylabel=r'$\sigma_{xx}$ (Pa)')
    ax.legend();fig.tight_layout()
    path=out/'10A_FD_stress_profile.png';fig.savefig(path,dpi=160);pathlist.append(str(path));plt.close(fig)
    return pathlist


def run(output='validation/results/stage10a_force_driven', *, ppc_x=PPC_X,
        steps=STEPS,dt=STEP_DT,plot=True,cell_dx=DX,refinement=False,
        domain_factor=1.0,pic_fraction=0.0):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    all_rows, snapshots, stats, final = [], [], {}, {}
    curves={}
    for method in ('mpm','gimp'):
        for drift in (0.0,DRIFT):
            r,s,m,stress,X = one_run(method,drift=drift,ppc_x=ppc_x,
                                      steps=steps,dt=dt,cell_dx=cell_dx,
                                      domain_factor=domain_factor,pic_fraction=pic_fraction)
            key=(method,drift)
            curves[key]=r
            all_rows.extend(r); snapshots.extend(s)
            stats[f'{method}_drift_{drift:g}']=m
            final[key]=stress
    for method in ('mpm','gimp'):
        err=final[(method,DRIFT)]-final[(method,0.0)]
        stats[f'{method}_galilean'] = dict(
            final_boost_vs_rest_stress_relative_l2=rms(err)/rms(analytical(X,steps*dt,0)[2]),
            final_boost_vs_rest_stress_rms_pa=rms(err))
    grid_refinement = []
    if refinement:
        for h in (0.5,0.25,0.125):
            for method in ('mpm','gimp'):
                if h == cell_dx:
                    result = stats[f'{method}_drift_{DRIFT:g}']
                else:
                    _,_,result,_,_=one_run(method,drift=DRIFT,ppc_x=ppc_x,
                                            steps=steps,dt=dt,cell_dx=h,
                                            domain_factor=domain_factor,pic_fraction=pic_fraction)
                grid_refinement.append(dict(method=method,cell_dx_m=h,
                     ppc_x=ppc_x,n_particles=result['n_particles'],
                     n_crossings=result['n_particle_cell_crossings'],
                     time_rms_stress_relative_l2=result['time_rms_stress_relative_l2'],
                     final_stress_relative_l2=result['final_stress_relative_l2']))
        save_csv(out/'10A_FD_grid_refinement.csv',grid_refinement)
    save_csv(out/'10A_FD_timeseries.csv',all_rows)
    save_csv(out/'10A_FD_stress_snapshots.csv',snapshots)
    plot_files=make_plots(out,curves,snapshots,steps) if plot else []
    if plot and grid_refinement:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax=plt.subplots(figsize=(6.6,4.2))
        for method in ('mpm','gimp'):
            entries=sorted([r for r in grid_refinement if r['method']==method],
                           key=lambda r:r['cell_dx_m'])
            ax.plot([r['cell_dx_m'] for r in entries],
                    [r['time_rms_stress_relative_l2'] for r in entries],
                    '-o',label=method.upper())
        ax.set(xlabel='Cell size in x (m)',ylabel='Time-RMS relative L2 stress error',
               title='Force-driven wave: spatial-grid refinement')
        ax.set_xticks([0.125,0.25,0.5])
        ax.legend();fig.tight_layout()
        file=out/'10A_FD_grid_refinement.png'
        fig.savefig(file,dpi=160); plt.close(fig)
        plot_files.append(str(file))
    checks=dict(
        no_axial_velocity_bcs=True,
        force_feedback_enabled=all(stats[f'{m}_drift_{DRIFT:g}']['force_feedback']
                                   for m in ('mpm','gimp')),
        drifting_particles_cross_cells=all(
            stats[f'{m}_drift_{DRIFT:g}']['n_particle_cell_crossings']>0
            for m in ('mpm','gimp')),
        mass_conserved=all(stats[f'{m}_drift_{v:g}']['max_mass_relative_error']<1e-12
                           for m in ('mpm','gimp') for v in (0.0,DRIFT)),
        axial_momentum_conserved=all(
            stats[f'{m}_drift_{v:g}']['max_momentum_x_relative_error']<1e-9
            for m in ('mpm','gimp') for v in (0.0,DRIFT)),
        stress_solutions_finite=all(
            np.isfinite(stats[f'{m}_drift_{v:g}']['final_stress_relative_l2'])
            for m in ('mpm','gimp') for v in (0.0,DRIFT)),
        grid_convergence_consistent=(
            all(next(r['time_rms_stress_relative_l2'] for r in grid_refinement
                     if r['method']==m and r['cell_dx_m']==0.125)
                < next(r['time_rms_stress_relative_l2'] for r in grid_refinement
                       if r['method']==m and r['cell_dx_m']==0.5)
                for m in ('mpm','gimp')) if grid_refinement else None),
    )
    result = dict(benchmark='Stage 10A force-driven stress crossing vs exact elastic wave',
        source='validation/stage10/force_driven_crossing.py',
        setup=dict(bar_length_m=L,bar_left_m=BAR_LEFT,domain_length_m=DOMAIN_X,
          cell_size_m=[cell_dx,DY],density_kg_m3=RHO,young_pa=YOUNG,poisson_ratio=NU,
          sigma_plane_strain_modulus_pa=M,transverse_displacement_bcs_only=True,
          gaussian_width_m=WIDTH,gaussian_displacement_amplitude_m=AMP,
          primary_boost_m_s=DRIFT,grid_move_per_step_for_boost=DRIFT*dt/cell_dx,
          time_step_s=dt,n_steps=steps,ppc_x=ppc_x,cell_dx_m=cell_dx,
          domain_factor=domain_factor,pic_fraction=pic_fraction),
        summary=stats,checks=checks,grid_refinement=grid_refinement,plots=plot_files,
        limitations=['Infinite bar reference is exact for infinitesimal linear elasticity, not the nonlinear updated formulation.',
                     'Wave must stay away from specimen ends until final time.',
                     'Only prescribed zero transverse velocities; free axial dynamics.',
                     'Numerical comparisons are case-specific and do not prove uGIMP superiority for all configurations.'])
    with open(out/'10A_FD_summary.json','w',encoding='utf8') as f:
        json.dump(result,f,indent=2)
    print(json.dumps(stats,indent=2))
    print('Results:',out)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='validation/results/stage10a_force_driven')
    parser.add_argument('--ppc-x',type=int,default=PPC_X,choices=(2,4,8))
    parser.add_argument('--steps',type=int,default=STEPS)
    parser.add_argument('--dt',type=float,default=STEP_DT)
    parser.add_argument('--cell-dx',type=float,default=DX,choices=(0.125,0.25,0.5))
    parser.add_argument('--domain-factor',type=float,default=1.0,
                        help='Full GIMP x-domain width = factor * (cell_dx / ppc_x).')
    parser.add_argument('--pic-fraction',type=float,default=0.0,
                        help='PIC/FLIP blend (0=FLIP, 1=PIC).')
    parser.add_argument('--refinement',action='store_true')
    parser.add_argument('--no-plot',action='store_true')
    args=parser.parse_args()
    run(args.output,ppc_x=args.ppc_x,steps=args.steps,dt=args.dt,plot=not args.no_plot,
        cell_dx=args.cell_dx,refinement=args.refinement,
        domain_factor=args.domain_factor,pic_fraction=args.pic_fraction)


if __name__=='__main__':
    main()
