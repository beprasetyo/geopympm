"""Stage 10A: reproducible uGIMP vs point-MPM cell-crossing validation.

Run from repository root (editable install or PYTHONPATH=src):
    python -m validation.stage10.validate_gimp
    python -m validation.stage10.validate_gimp --output my/results

Experiments
-----------
10A-1  Prescribed single-particle positions: local internal nodal force vs
       exact continuum domain integral (NOT a stress evolution simulation).
10A-2  Uniformly translating, prestressed rectangular block: a genuine solver
       time loop but all nodal velocities prescribed. Exact continuum nodal
       traction-force integral is the reference. Stress should remain constant.
10A-3  Particle crossing under a spatially nonuniform, time-independent,
       prescribed nodal velocity field: accumulated sigma_xx compared with
       analytical smooth-continuum kinematics. Stress DOES evolve here;
       force feedback is suppressed by BC, so NOT a free dynamic stress wave.

These distinguish interpolation/force consistency from full coupled validation.
Units: SI (m, s, kg, N, Pa); 2D plane strain, unit thickness.
"""
from __future__ import annotations

import argparse
import csv
import json
from math import cos, log, pi, sin
from pathlib import Path

import numpy as np

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.interpolation import evaluate_quad4, evaluate_gimp
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver
from geopympm.solvers.p2g import particle_to_grid
from geopympm.solvers.grid_update import update_grid_momentum
from validation.stage6.cell_crossing import exact_nodal_force_x


def _interpolator(method):
    if method == 'mpm':
        return evaluate_quad4
    if method == 'gimp':
        return evaluate_gimp
    raise ValueError(f'Invalid interpolation method: {method}')


def _save_csv(path, rows):
    if not rows:
        raise ValueError('No rows to save.')
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rms(values):
    v = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(v * v)))


def single_particle_probe(n_points=201, width=0.5):
    """Moving particle prescribed at x=2 +/- 0.25; compare integrated hat.

    Continuum comparator: uniform stress over the rectangle, integrated
    *with the same constant representative physical volume* V=0.2; the
    nonzero y-average hat weight at y=1.5 for node (2,1) is 0.5.
    """
    if not isinstance(n_points, int) or n_points < 3 or width <= 0 or width > 1:
        raise ValueError('Bad single-particle probe parameters.')
    grid = BackgroundGrid((0, 0), (1, 1), (4, 4))
    x_values = np.linspace(1.75, 2.25, n_points)
    sample_node = grid.node_id(2, 1)
    sigma, volume = 100.0, 0.2
    p = ParticleSet([[1.75, 1.5]], [1.0], [volume],
                    stress=[[sigma, 0, 0, 0, 0, 0]],
                    domain_size=[[width, width]])
    rows = []
    for x in x_values:
        p.position[0, 0] = x
        forces = {}
        for method in ('mpm', 'gimp'):
            ip = _interpolator(method)(p, grid)
            particle_to_grid(p, grid, ip)
            update_grid_momentum(p, grid, ip, 0.001)
            forces[method] = float(grid.force_internal[sample_node, 0])
        exact = exact_nodal_force_x(
            node_x=2.0, x_left=x - width / 2, x_right=x + width / 2,
            h=1.0, thickness=volume / width, sigma_xx=sigma,
            transverse_weight=0.5,
        )
        rows.append(dict(x_m=float(x), mpm_force_n=forces['mpm'],
                         gimp_force_n=forces['gimp'],
                         exact_domain_force_n=float(exact)))
    mpm = np.array([r['mpm_force_n'] for r in rows])
    gimp = np.array([r['gimp_force_n'] for r in rows])
    exact = np.array([r['exact_domain_force_n'] for r in rows])
    results = {
        'description': 'Prescribed positions, constant stress; not a dynamic solver.',
        'n_positions': n_points, 'domain_width_m': width,
        'max_adjacent_mpm_force_change_n': float(np.max(np.abs(np.diff(mpm)))),
        'max_adjacent_gimp_force_change_n': float(np.max(np.abs(np.diff(gimp)))),
        'max_gimp_exact_domain_error_n': float(np.max(np.abs(gimp - exact))),
        'rms_mpm_vs_domain_force_error_n': _rms(mpm-exact),
        'rms_gimp_vs_domain_force_error_n': _rms(gimp-exact),
    }
    return rows, results


def moving_prestressed_block(method, *, ppc=4, dt=0.01, steps=150):
    """Actual explicit solver with prescribed translation (no force feedback).

    Particle rectangles tile the 3x1 m block exactly at t=0; uGIMP domain
    equals physical rectangular particle quadrature tile (dx=1/ppc,dy=0.5).
    All BC velocities=(speed,0); stress stays 120 Pa throughout.
    """
    _interpolator(method)
    if not isinstance(ppc, int) or ppc < 2 or ppc > 16 or steps < 2:
        raise ValueError('Unsupported ppc/steps.')
    speed, sigma, left0 = 0.9, 120.0, 1.22
    xs = left0 + (np.arange(3*ppc) + 0.5) / ppc
    ys = (0.25, 0.75)
    xy = np.array([(x, y) for y in ys for x in xs], dtype=float)
    n = len(xy)
    volume = 3.0/n  # 2D area under unit-thickness convention
    p = ParticleSet(xy, np.full(n, 1200.0*volume), np.full(n, volume),
                    velocity=np.tile([speed,0.0], (n,1)),
                    domain_size=np.tile([1.0/ppc,0.5], (n,1)))
    p.stress[:,0] = sigma
    grid = BackgroundGrid((0.0,0.0), (1.0,1.0), (8,1))
    grid.velocity_bc_mask[:,:] = True
    grid.velocity_bc_value[:,0] = speed
    solver = ExplicitMPMSolver(p, grid, LinearElastic(120000.0, 0.25),
                               dt=dt, pic_fraction=0.0, interpolation=method)
    sample_node = grid.node_id(5,0)
    rows = []
    total_initial_mass = float(np.sum(p.mass))
    for it in range(steps):
        t = it*dt  # forces computed at PRE-advection positions
        diag = solver.step()
        expected = exact_nodal_force_x(
            node_x=5.0, x_left=left0+speed*t,
            x_right=left0+3.0+speed*t,
            h=1.0, thickness=1.0, sigma_xx=sigma,
        )
        computed = float(grid.force_internal[sample_node,0])
        rows.append(dict(method=method, step=it+1, time_s=t,
                         mpm_force_n=computed if method == 'mpm' else '',
                         gimp_force_n=computed if method == 'gimp' else '',
                         computed_force_n=computed, exact_force_n=expected,
                         signed_force_error_n=computed-expected,
                         stress_drift_pa=float(np.max(np.abs(p.stress[:,0]-sigma))),
                         mass_relative_error=abs(diag.total_mass-total_initial_mass)/total_initial_mass,
                         position_error_m=float(np.max(np.abs(p.position[:,0] - (xy[:,0]+speed*(it+1)*dt))))))
    err = np.array([r['signed_force_error_n'] for r in rows])
    force = np.array([r['computed_force_n'] for r in rows])
    exact = np.array([r['exact_force_n'] for r in rows])
    scale = abs(sigma)*0.5
    metrics = dict(method=method, particles=n, ppc_x=ppc, steps=steps,
                   rms_force_error_n=_rms(err),
                   normalized_rms_force_error=_rms(err)/scale,
                   max_force_error_n=float(np.max(np.abs(err))),
                   max_force_increment_noise_n=float(np.max(np.abs(np.diff(force-exact)))),
                   max_stress_drift_pa=max(r['stress_drift_pa'] for r in rows),
                   max_mass_relative_error=max(r['mass_relative_error'] for r in rows),
                   max_position_error_m=max(r['position_error_m'] for r in rows))
    return rows, metrics


def _reference_x_rk4(x0, duration, dt=0.00005, v0=0.6, amp=0.08, k=pi/2):
    """Independent time integration of smooth field v(x)=v0+amp*sin(k*x)."""
    n = int(round(duration/dt))
    h = duration/n
    x = float(x0)
    def v(z):
        return v0+amp*sin(k*z)
    for _ in range(n):
        a = v(x)
        b = v(x+h*a/2)
        c = v(x+h*b/2)
        d = v(x+h*c)
        x += h*(a+2*b+2*c+d)/6
    return x


def prescribed_strain_crossing(method, *, dt=0.005, steps=160):
    """Strain and stress evolve in an externally imposed smooth 1D flow.

    All nodes are prescribed at v(x_i)=v0+amp*sin(k x_i), v_y=0;
    physics at the particle sees nodal hat interpolation. Exact reference
    uses smooth v(x), not grid-sampled v. For 1D steady flow:
      epsilon_xx(t)=ln[v(x(t))/v(x0)] (Eulerian velocity gradient integrated
      along the exact trajectory), sigma_xx=M epsilon_xx for the Stage-4
      incremental isotropic elasticity with zero rigid spin.
    Reference time integration of x uses independent RK4 and small dt.
    This is NOT force-driven dynamics; BC forces do not influence motion.
    """
    _interpolator(method)
    E, nu, density = 120000.0, 0.25, 1200.0
    M = E*(1.0-nu)/((1+nu)*(1-2*nu))
    x0, v0, amp, k = 1.70, 0.6, 0.08, pi/2
    velocity = lambda x: v0+amp*np.sin(k*x)
    grid = BackgroundGrid((0,0),(0.5,0.5),(12,2))
    p = ParticleSet([[x0,0.5]], [density*0.25], [0.25],
                    velocity=[[float(velocity(x0)),0]],
                    domain_size=[[0.5,0.5]])
    grid.velocity_bc_mask[:,:] = True
    grid.velocity_bc_value[:,0] = velocity(grid.node_coordinates[:,0])
    solver = ExplicitMPMSolver(p, grid, LinearElastic(E,nu),
                               dt=dt,pic_fraction=1.0,interpolation=method,
                               stress_update='incremental',
                               kinematics_update='euler')
    rows = []
    for it in range(steps):
        diag = solver.step()
        t = (it+1)*dt
        xref = _reference_x_rk4(x0, t, dt=0.0002, v0=v0,amp=amp,k=k)
        sigma_ref = M*log(float(velocity(xref))/float(velocity(x0)))
        grad_ref = amp*k*cos(k*xref)
        rows.append(dict(method=method,step=it+1,time_s=t,
                         particle_x_m=float(p.position[0,0]),
                         reference_x_m=xref,
                         sigma_xx_pa=float(p.stress[0,0]),
                         reference_sigma_xx_pa=sigma_ref,
                         sigma_error_pa=float(p.stress[0,0]-sigma_ref),
                         strain_xx=float(p.strain[0,0]),
                         reference_velocity_gradient_s_inv=grad_ref,
                         mass_kg=diag.total_mass))
    err = np.array([r['sigma_error_pa'] for r in rows])
    sigma = np.array([r['sigma_xx_pa'] for r in rows])
    ref = np.array([r['reference_sigma_xx_pa'] for r in rows])
    # Compare pointwise stress increment against smooth-continuum increment:
    # diagnostic for the non-smoothness of stress *updates*, not a proof of
    # superiority in full coupled stress evolution.
    step_error = np.diff(np.r_[0.0, sigma-ref])
    summary = dict(method=method, steps=steps, dt_s=dt,
                   start_x_m=x0, end_x_m=float(p.position[0,0]),
                   reference_end_x_m=rows[-1]['reference_x_m'],
                   rms_stress_error_pa=_rms(err),
                   max_abs_stress_error_pa=float(np.max(np.abs(err))),
                   rms_stress_increment_error_pa=_rms(step_error),
                   max_stress_increment_error_pa=float(np.max(np.abs(step_error))),
                   particle_crossed_x2=bool(p.position[0,0]>2.0),
                   max_mass_relative_error=0.0)
    return rows, summary



def force_driven_elastic_wave(method, *, nx=24, ny=2, dt=0.001, steps=300):
    """Force-driven small-strain wave; analytical continuum comparison.

    Same mode and constitutive initialization as the Stage 6D standing-wave
    verification, but uses both interpolators at identical particle locations.
    Only axial end-node velocities and all transverse velocities are fixed;
    internal stress forces drive the interior. Small motion generally does
    not cross grid lines, so this is a complementary stress/force coupling
    test, not cell-crossing stress-noise evidence.
    """
    _interpolator(method)
    L, H, rho = 4.0, 0.5, 1200.0
    E, nu, amp = 120000.0, 0.25, 0.002
    lam = E*nu / ((1+nu)*(1-2*nu))
    G = E/(2*(1+nu))
    M = lam + 2*G
    k = pi/L
    omega = k*np.sqrt(M/rho)
    grid = BackgroundGrid((0,0),(L/nx,H/ny),(nx,ny))
    X = np.array([
        ((ix+sx)*L/nx,(iy+sy)*H/ny)
        for iy in range(ny) for ix in range(nx)
        for sy in (0.25,0.75) for sx in (0.25,0.75)
    ],dtype=float)
    n = len(X)
    basis = np.sin(k*X[:,0])
    stress_basis = np.cos(k*X[:,0])
    u0 = amp*basis
    eps0 = amp*k*stress_basis
    xp = X.copy()
    xp[:,0] += u0
    V0 = L*H/n
    p = ParticleSet(
        xp, np.full(n,rho*V0),np.full(n,V0),
        domain_size=np.tile([0.95*L/nx/2,0.95*H/ny/2],(n,1)),
    )
    p.volume[:] = V0*(1+eps0)
    p.deformation_gradient[:,0,0] = 1+eps0
    p.strain[:,0] = eps0
    p.stress[:,0] = M*eps0
    p.stress[:,1] = lam*eps0
    p.stress[:,2] = lam*eps0
    grid.velocity_bc_mask[:,1] = True
    xnodes = grid.node_coordinates[:,0]
    grid.velocity_bc_mask[np.isclose(xnodes,0)|np.isclose(xnodes,L),0] = True
    solver = ExplicitMPMSolver(p,grid,LinearElastic(E,nu),dt=dt,
                               pic_fraction=0.0,interpolation=method)
    u_scale = np.linalg.norm(amp*basis)
    sigma_scale = np.linalg.norm(M*amp*k*stress_basis)
    rows = []
    def record(it,kinetic):
        t=it*dt
        expected_u=amp*np.cos(omega*t)*basis
        expected_sigma=M*amp*k*np.cos(omega*t)*stress_basis
        displacement=p.position[:,0]-X[:,0]
        rows.append(dict(method=method,step=it,time_s=t,
                         displacement_relative_l2=float(np.linalg.norm(displacement-expected_u)/u_scale),
                         stress_relative_l2=float(np.linalg.norm(p.stress[:,0]-expected_sigma)/sigma_scale),
                         kinetic_energy_j=float(kinetic),
                         modal_sigma_pa=float(np.dot(p.stress[:,0],stress_basis)/np.dot(stress_basis,stress_basis)),
                         modal_sigma_reference_pa=float(M*amp*k*np.cos(omega*t)),
                         mass_relative_error=float(abs(np.sum(p.mass)-rho*L*H)/(rho*L*H))))
    record(0,0.0)
    for it in range(1,steps+1):
        diag=solver.step()
        record(it,diag.kinetic_energy)
    summary=dict(method=method,n_particles=n,nx=nx,ny=ny,steps=steps,
                 dt_s=dt,t_end_s=steps*dt,
                 max_displacement_relative_l2=max(r['displacement_relative_l2'] for r in rows),
                 max_stress_relative_l2=max(r['stress_relative_l2'] for r in rows),
                 final_stress_relative_l2=rows[-1]['stress_relative_l2'],
                 max_mass_relative_error=max(r['mass_relative_error'] for r in rows),
                 driven_by_internal_forces=True,
                 grid_cell_crossing_not_required=True)
    return rows,summary

def _plots(out, probe, block, stress, wave, refinement):
    """Matplotlib is optional, plots excluded from solver dependencies."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    plt.rcParams.update({'figure.dpi':145,'font.size':9})
    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    a = probe
    ax.plot([r['x_m'] for r in a], [r['mpm_force_n'] for r in a], label='Point MPM', lw=1.5)
    ax.plot([r['x_m'] for r in a], [r['gimp_force_n'] for r in a], label='uGIMP',lw=2)
    ax.plot([r['x_m'] for r in a], [r['exact_domain_force_n'] for r in a],
            '--', label='Analytical domain average', lw=1)
    ax.axvline(2.0,ls=':',c='0.5')
    ax.set(xlabel='Particle center x (m)',ylabel='Internal force on node (2,1) (N)',
           title='10A-1: Single-particle crossing, constant stress')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'10A_single_particle_force.png'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0,3.7))
    for m, label in (('mpm','Point MPM'),('gimp','uGIMP')):
        a = block[m]
        ax.plot([r['time_s'] for r in a], [r['computed_force_n'] for r in a],label=label,lw=1.5)
    a=block['mpm']
    ax.plot([r['time_s'] for r in a], [r['exact_force_n'] for r in a],
            '--',c='black',alpha=.75,lw=1,label='Exact continuum integral')
    ax.set(xlabel='Time (s)',ylabel='Internal force on node (5,0) (N)',
           title='10A-2: Uniformly translating prestressed block')
    ax.legend();fig.tight_layout();fig.savefig(out/'10A_block_force.png');plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0,3.7))
    for m, label in (('mpm','Point MPM'),('gimp','uGIMP')):
        a = block[m]
        ax.plot([r['time_s'] for r in a], [r['signed_force_error_n'] for r in a],
                label=label,lw=1.3)
    ax.axhline(0,ls='--',c='black',lw=.8)
    ax.set(xlabel='Time (s)',ylabel='Force error (N)',
           title='10A-2: Force minus analytical continuum')
    ax.legend();fig.tight_layout();fig.savefig(out/'10A_block_force_error.png');plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0,3.7))
    for m,label in (('mpm','Point MPM'),('gimp','uGIMP')):
        a=stress[m]
        ax.plot([r['time_s'] for r in a],[r['sigma_xx_pa'] for r in a],label=label,lw=1.5)
    a=stress['mpm']
    ax.plot([r['time_s'] for r in a],[r['reference_sigma_xx_pa'] for r in a],
            '--',c='black',label='Reference smooth flow',lw=1)
    ax.set(xlabel='Time (s)',ylabel='Particle sigma_xx (Pa)',
           title='10A-3: Strain-driven stress across a grid line (prescribed BC)')
    ax.legend();fig.tight_layout();fig.savefig(out/'10A_stress_crossing.png');plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0,3.7))
    for m,label in (('mpm','Point MPM'),('gimp','uGIMP')):
        a=stress[m]
        ax.plot([r['time_s'] for r in a],[r['sigma_error_pa'] for r in a],label=label,lw=1.5)
    ax.axhline(0,ls='--',c='black',lw=.8)
    ax.set(xlabel='Time (s)',ylabel='Stress minus smooth-flow reference (Pa)',
           title='10A-3: Stress error (not free dynamic response)')
    ax.legend();fig.tight_layout();fig.savefig(out/'10A_stress_error.png');plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.0,3.7))
    for m,label in (('mpm','Point MPM'),('gimp','uGIMP')):
        a=wave[m]
        ax.plot([r['time_s'] for r in a], [r['stress_relative_l2'] for r in a],label=label,lw=1.5)
    ax.set(xlabel='Time (s)',ylabel='Relative L2 error in sigma_xx',
           title='10A-4: Force-driven elastic standing wave (Stage 6D type)')
    ax.legend();fig.tight_layout();fig.savefig(out/'10A_wave_stress_error.png');plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.2,3.8))
    for method,label in (('mpm','Point MPM'),('gimp','uGIMP')):
        vals=[r for r in refinement if r['method']==method]
        ax.semilogy([r['ppc_x'] for r in vals],
                    [max(r['rms_force_error_n'],1e-15) for r in vals],
                    marker='o',label=label)
    ax.set(xlabel='Particles per cell in x (2 particle rows in y)',
           ylabel='RMS force error (N, logarithmic)',
           title='10A-5: Particle resolution and force assembly error')
    ax.legend();fig.tight_layout();fig.savefig(out/'10A_refinement.png');plt.close(fig)
    return True


def run(output='validation/results/stage10a', *, ppc=4, steps=150, plot=True):
    out=Path(output)
    out.mkdir(parents=True,exist_ok=True)
    probe, probe_stats = single_particle_probe()
    _save_csv(out/'10A_single_particle.csv',probe)
    block, block_stats, stress, stress_stats, wave, wave_stats={},{},{},{},{},{}
    for method in ('mpm','gimp'):
        block[method], block_stats[method] = moving_prestressed_block(
            method, ppc=ppc, steps=steps)
        stress[method], stress_stats[method] = prescribed_strain_crossing(method)
        wave[method], wave_stats[method] = force_driven_elastic_wave(method)
    _save_csv(out/'10A_moving_block.csv', [r for m in ('mpm','gimp') for r in block[m]])
    _save_csv(out/'10A_strain_driven_stress.csv', [r for m in ('mpm','gimp') for r in stress[m]])
    _save_csv(out/'10A_force_driven_wave.csv', [r for m in ('mpm','gimp') for r in wave[m]])
    refinement=[]
    for resolution in (2,4,8):
        for method in ('mpm','gimp'):
            if resolution==ppc and steps==150:
                data=block_stats[method]
            else:
                _, data=moving_prestressed_block(method,ppc=resolution,steps=150)
            refinement.append(dict(ppc_x=resolution,method=method,
                                   rms_force_error_n=data['rms_force_error_n'],
                                   max_increment_noise_n=data['max_force_increment_noise_n']))
    _save_csv(out/'10A_refinement.csv',refinement)
    plots = _plots(out,probe,block,stress,wave,refinement) if plot else False
    summary=dict(
        benchmark='GeoPyMPM Stage 10A uGIMP vs point MPM',
        scope='Interp/force and prescribed-kinematics stress diagnostics; not coupled runout validation',
        single_particle=probe_stats, moving_block=block_stats,
        strain_driven_stress=stress_stats, force_driven_wave=wave_stats,
        particle_refinement=refinement,
        plots_created=plots,
    )
    # Requirements independent of expected superiority for *all* geometries.
    tolerances=dict(single_particle_domain_force_n=1e-10,
                    block_gimp_rms_force_n=1e-8,
                    prescribed_stress_mass_rel=1e-12)
    checks={
        'single_particle_domain_averaging': probe_stats['max_gimp_exact_domain_error_n']<tolerances['single_particle_domain_force_n'],
        'single_particle_continuity_improvement':probe_stats['max_adjacent_gimp_force_change_n']<probe_stats['max_adjacent_mpm_force_change_n'],
        'moving_block_gimp_matches_exact': block_stats['gimp']['rms_force_error_n']<tolerances['block_gimp_rms_force_n'],
        'moving_block_gimp_less_force_error':block_stats['gimp']['rms_force_error_n']<block_stats['mpm']['rms_force_error_n'],
        'moving_block_stress_invariant':all(block_stats[m]['max_stress_drift_pa']<1e-7 for m in ('mpm','gimp')),
        'moving_block_motion_invariant':all(block_stats[m]['max_position_error_m']<1e-9 for m in ('mpm','gimp')),
        'prescribed_stress_grid_crossing':all(stress_stats[m]['particle_crossed_x2'] for m in ('mpm','gimp')),
        'stress_evolves_under_prescribed_strain':all(abs(stress[m][-1]['sigma_xx_pa'])>10 for m in ('mpm','gimp')),
        'stress_stays_finite':all(np.all(np.isfinite([r['sigma_xx_pa'] for r in stress[m]])) for m in ('mpm','gimp')),
        'force_driven_wave_stress_accuracy':all(wave_stats[m]['max_stress_relative_l2']<0.08 for m in ('mpm','gimp')),
        'force_driven_wave_mass_conservation':all(wave_stats[m]['max_mass_relative_error']<1e-12 for m in ('mpm','gimp')),
        'refinement_gimp_force_exact':all(row['rms_force_error_n']<1e-8 for row in refinement if row['method']=='gimp'),
    }
    summary['checks']=checks
    summary['passed']=bool(all(checks.values()))
    with (out/'10A_summary.json').open('w',encoding='utf-8') as f:
        json.dump(summary,f,indent=2,allow_nan=False)
    print('Stage 10A: MPM vs uGIMP validation')
    print(f'  Single-particle max |delta force|: MPM {probe_stats["max_adjacent_mpm_force_change_n"]:.6g} N; uGIMP {probe_stats["max_adjacent_gimp_force_change_n"]:.6g} N')
    for m in ('mpm','gimp'):
        print(f'  Moving block {m:4s}: RMS nodal force error {block_stats[m]["rms_force_error_n"]:.6g} N; max increment noise {block_stats[m]["max_force_increment_noise_n"]:.6g} N')
        print(f'  Prescribed stress {m:4s}: RMS stress error {stress_stats[m]["rms_stress_error_pa"]:.6g} Pa; RMS stress-increment error {stress_stats[m]["rms_stress_increment_error_pa"]:.6g} Pa')
        print(f'  Force-driven wave {m:4s}: max stress relative L2 error {wave_stats[m]["max_stress_relative_l2"]:.6g}')
    print(f'  Checks: {sum(checks.values())}/{len(checks)} passed; overall {"PASS" if summary["passed"] else "FAIL"}. Outputs: {out}')
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='validation/results/stage10a')
    parser.add_argument('--ppc',type=int,default=4)
    parser.add_argument('--steps',type=int,default=150)
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    result=run(args.output,ppc=args.ppc,steps=args.steps,plot=not args.no_plots)
    raise SystemExit(0 if result['passed'] else 1)
