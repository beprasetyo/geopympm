"""Stage 13D_V1: 2D self-weight cantilever, explicit MPM numerical experiment.

Reference: Iaconeta et al. 2020, Sec. 4.5.1, 8x1x1 m cantilever.
Important: the published value is a nonlinear quasi-static reference.
This GeoPyMPM experiment is explicit dynamics with optional velocity relaxation;
it is not an identical solution procedure or constitutive law.

All units SI. The MPM tip monitor is a near-tip particle strip, not the exact
geometric end; its position is reported in summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.materials import LinearElastic, CompressibleNeoHookean
from geopympm.solvers import ExplicitMPMSolver

LENGTH_M = 8.0
HEIGHT_M = 1.0
THICKNESS_M = 1.0
DENSITY_KG_M3 = 1000.0
YOUNG_PA = 90.0e6
POISSON_RATIO = 0.0
GRAVITY_M_S2 = 9.81
REFERENCE_BEAM_M = -0.67806
REFERENCE_FINE_M = -0.67433


@dataclass(frozen=True)
class BeamConfig:
    cell_size: float = 0.5
    particles_per_axis: int = 2
    dt: float = 0.00020
    ramp_time: float = 1.5
    gravity: float = GRAVITY_M_S2 / 10.0  # 10% load: small-deflection validation mode
    # Damping is applied as post-step particle velocity scaling,
    # for numerical dynamic relaxation (not a material constitutive property).
    velocity_damping_rate: float = 2.0  # inverse seconds; 0 => undamped dynamics
    pic_fraction: float = 0.0  # FLIP by default; damping is explicit and recorded
    scheme: str = "usl"
    interpolation: str = "mpm"
    material: str = "linear"  # "cnh" optional; not asserted equal to published hyperelastic law
    cfl: float = 0.25

    def validate(self):
        if not (math.isfinite(self.cell_size) and self.cell_size > 0
                and math.isclose(HEIGHT_M / self.cell_size, round(HEIGHT_M / self.cell_size), abs_tol=1e-12)
                and math.isclose(LENGTH_M / self.cell_size, round(LENGTH_M / self.cell_size), abs_tol=1e-12)):
            raise ValueError("cell_size must divide both 1 m and 8 m exactly")
        if type(self.particles_per_axis) is not int or not 1 <= self.particles_per_axis <= 8:
            raise ValueError("particles_per_axis must be an integer in [1,8]")
        for name in ("dt", "ramp_time", "gravity", "velocity_damping_rate", "pic_fraction", "cfl"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(name + " must be finite")
        if self.dt <= 0 or self.ramp_time < 0 or self.gravity <= 0 or self.velocity_damping_rate < 0:
            raise ValueError("invalid dt, ramp_time, gravity or damping")
        if not 0 <= self.pic_fraction <= 1 or not 0 < self.cfl <= 1:
            raise ValueError("invalid pic_fraction or cfl")
        if self.scheme not in ("usl", "usf", "musl"):
            raise ValueError("invalid scheme")
        if self.interpolation not in ("mpm", "gimp", "cpgimp", "cpdi"):
            raise ValueError("invalid interpolation")
        if self.material not in ("linear", "cnh"):
            raise ValueError("material must be linear or cnh")


def analytical_deflection_at(x, *, g=GRAVITY_M_S2):
    """Timoshenko small-deflection result at x for uniform self-weight.

    Clamped at x=0, free at x=L, 0<=x<=L.
    No material point / MPM solver operations enter this reference.
    """
    if not (math.isfinite(x) and 0 <= x <= LENGTH_M):
        raise ValueError("x must be within [0,L]")
    if not (math.isfinite(g) and g > 0):
        raise ValueError("g must be finite and positive")
    area=THICKNESS_M*HEIGHT_M
    inertia=THICKNESS_M*HEIGHT_M**3/12.0
    shear_modulus=YOUNG_PA/(2.0*(1.0+POISSON_RATIO))
    shear_area=(5.0/6.0)*area
    q=DENSITY_KG_M3*g*area
    bending=-q*x*x*(6*LENGTH_M*LENGTH_M-4*LENGTH_M*x+x*x)/(24.0*YOUNG_PA*inertia)
    shear=-q*(LENGTH_M*x-0.5*x*x)/(shear_modulus*shear_area)
    return dict(bending_m=bending,shear_m=shear,total_m=bending+shear,
                inertia_m4=inertia,shear_area_m2=shear_area,x_m=x)


def analytical_beam_deflection(*, g=GRAVITY_M_S2):
    """Small-deflection Timoshenko theory at the EXACT geometric tip."""
    return analytical_deflection_at(LENGTH_M, g=g)


def build_solver(config: BeamConfig):
    """Create particle-seeded beam plus padded grid and a clamped left edge."""
    config.validate()
    h = config.cell_size
    dx = h / config.particles_per_axis
    nx = int(round(LENGTH_M/dx))
    ny = int(round(HEIGHT_M/dx))
    xs = (np.arange(nx) + 0.5)*dx
    ys = (np.arange(ny) + 0.5)*dx
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    xy = np.stack([xx.ravel(), yy.ravel()], axis=1)
    volume = np.full(len(xy), dx*dx*THICKNESS_M)
    particle = ParticleSet(xy, DENSITY_KG_M3*volume, volume,
                           domain_size=np.full_like(xy, dx))
    # Grid padding: x[-0.5,9], y[-1.5,2.5] for h=0.5, extends below beam.
    origin = np.array([-h, -1.5], dtype=float)
    extent = np.array([LENGTH_M+2*h, 4.0], dtype=float)
    n_cells = tuple(int(round(span/h)) for span in extent)
    grid = BackgroundGrid(origin, (h,h), n_cells)
    x, y = grid.node_coordinates.T
    support = (np.abs(x) <= 1.0e-12) & (y >= -h-1e-12) & (y <= HEIGHT_M+h+1e-12)
    grid.velocity_bc_mask[support] = True
    grid.velocity_bc_value[support] = 0.0
    if np.count_nonzero(support) == 0:
        raise RuntimeError("No clamped grid nodes selected")
    if config.interpolation == "cpdi":
        # Particle initial domains are consistent with F=I.
        pass
    model = (LinearElastic(YOUNG_PA, POISSON_RATIO) if config.material == "linear"
             else CompressibleNeoHookean.from_young_poisson(YOUNG_PA, POISSON_RATIO))
    solver = ExplicitMPMSolver(particle, grid, model, dt=config.dt, cfl=config.cfl,
                               gravity=(0.0, 0.0), scheme=config.scheme,
                               interpolation=config.interpolation,
                               pic_fraction=config.pic_fraction,
                               stress_update=("incremental" if config.material=="linear" else "hyperelastic"),
                               kinematics_update="exponential")
    last_x = float(np.max(particle.position0[:,0]))
    monitor = np.isclose(particle.position0[:,0], last_x, rtol=0, atol=dx*1e-5)
    return solver, monitor


def _ramp_fraction(t, ramp_time):
    if ramp_time == 0.0:
        return 1.0
    ratio = min(1.0, max(0.0, t/ramp_time))
    # C1 smoothstep start/stop reduces impulsive excitation.
    return ratio*ratio*(3.0-2.0*ratio)


def _observation(solver, monitor, load_factor, damping_rate):
    p=solver.particles
    displ=p.position-p.position0
    vel=p.velocity
    ke=float(0.5*np.sum(p.mass*np.sum(vel**2,axis=1)))
    return dict(step=int(solver.step_count), time_s=float(solver.time),
                gravity_factor=float(load_factor),
                near_tip_x0_m=float(np.mean(p.position0[monitor,0])),
                near_tip_uy_m=float(np.mean(displ[monitor,1])),
                near_tip_ux_m=float(np.mean(displ[monitor,0])),
                mean_clamp_side_uy_m=float(np.mean(displ[p.position0[:,0]<0.5,1])),
                max_particle_speed_m_s=float(np.max(np.linalg.norm(vel,axis=1))),
                kinetic_energy_j=ke,
                min_J=float(np.min(np.linalg.det(p.deformation_gradient))),
                max_abs_sigma_xx_pa=float(np.max(np.abs(p.stress[:,0]))),
                damping_rate_per_s=float(damping_rate))


def run_case(*, output_dir: Path, config: BeamConfig, steps: int,
             sample_every: int = 50, vtu_every: int = 0, plot: bool = False,
             progress_every: int = 500):
    """Execute dynamics, write numerical evidence, never mislabel as static validated.

    Default outputs are diagnostics.csv / summary.json; VTU requires vtu_every>0.
    Output directory must not have pre-existing content.
    """
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be nonnegative integer")
    if type(sample_every) is not int or sample_every <= 0:
        raise ValueError("sample_every must be positive integer")
    if type(vtu_every) is not int or vtu_every < 0:
        raise ValueError("vtu_every must be nonnegative integer")
    if type(progress_every) is not int or progress_every < 0:
        raise ValueError("progress_every must be a nonnegative integer")
    solver, monitor = build_solver(config)
    out=Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing results in {out}")
    out.mkdir(parents=True, exist_ok=True)
    from geopympm.io import OutputManager, ProgressReporter
    manager=(OutputManager(out/'visualization',every_n_steps=vtu_every,
                           particles=True,grid=False,cpdi_domains=False)
             if vtu_every else None)
    if manager:
        manager.write_initial(solver)
    observations=[_observation(solver, monitor, 0, config.velocity_damping_rate)]
    failure_reason = None
    failure_requested_step = None
    progress = ProgressReporter(total_steps=steps, every=progress_every,
                                label="Stage 13D_V1")
    progress.start()
    for i in range(steps):
        frac=_ramp_fraction(solver.time+solver.dt, config.ramp_time)
        # Applied acceleration, not direct modification of force kernel.
        solver.gravity[:] = [0.0, -config.gravity*frac]
        try:
            diag=solver.step()
        except (ValueError, FloatingPointError, ArithmeticError) as exc:
            # Preserve evidence for unstable full-gravity experiments. Explicit
            # solver.step() already rolls back the failed step atomically.
            failure_reason=f"{type(exc).__name__}: {exc}"
            failure_requested_step=i+1
            progress.finish(step=solver.step_count, time_s=solver.time,
                            status="NUMERICAL FAILURE", detail=failure_reason)
            if observations[-1]["step"] != solver.step_count:
                observations.append(_observation(solver,monitor,frac,config.velocity_damping_rate))
            break
        if config.velocity_damping_rate:
            # Deliberate numerical relaxation, NOT a new solver constitutive term.
            solver.particles.velocity *= math.exp(-config.velocity_damping_rate*config.dt)
        if manager:
            manager.on_step_completed(solver, diag)
        current_obs = None
        if (i+1)%sample_every == 0 or i == steps-1:
            current_obs = _observation(solver,monitor,frac,config.velocity_damping_rate)
            observations.append(current_obs)
        if progress.should_report(solver.step_count):
            obs = (current_obs if current_obs is not None else
                   _observation(solver,monitor,frac,config.velocity_damping_rate))
            progress.update(step=solver.step_count, time_s=solver.time,
                            metrics={"tip_uy(m)": obs["near_tip_uy_m"],
                                     "vmax(m/s)": obs["max_particle_speed_m_s"],
                                     "Jmin": obs["min_J"]})
    if not failure_reason:
        progress.finish(step=solver.step_count, time_s=solver.time)
    with (out/'diagnostics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(observations[0]))
        writer.writeheader();writer.writerows(observations)
    tip=observations[-1]['near_tip_uy_m']
    analytic=analytical_beam_deflection(g=config.gravity)
    analytic_near_tip=analytical_deflection_at(float(observations[-1]['near_tip_x0_m']),g=config.gravity)
    # Export the final committed particle state for reproducible shape plots.
    p=solver.particles
    displ=p.position-p.position0
    with (out/'final_particles.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['x0_m','y0_m','x_m','y_m','ux_m','uy_m','sigma_xx_pa','sigma_yy_pa','sigma_xy_pa','J'])
        js=np.linalg.det(p.deformation_gradient)
        for i in range(p.n_particles):
            writer.writerow([p.position0[i,0],p.position0[i,1],p.position[i,0],p.position[i,1],
                             displ[i,0],displ[i,1],p.stress[i,0],p.stress[i,1],p.stress[i,3],js[i]])
    # A single last-step velocity is not a defensible static-equilibrium test.
    # Use full load + a 1 s history window and monitor velocity/displacement changes.
    final=observations[-1]
    window=[o for o in observations if o['time_s'] >= final['time_s']-1.0-1e-12]
    max_v=max(o['max_particle_speed_m_s'] for o in window)
    drift=(max(o['near_tip_uy_m'] for o in window) - min(o['near_tip_uy_m'] for o in window))
    has_hold=(final['time_s']-config.ramp_time >= 1.0 and
              all(abs(o['gravity_factor']-1.0)<1e-10 for o in window))
    # A toleranced diagnostic (not force-equilibrium proof). Full convergence also
    # requires grid refinement, longer run and damping sensitivity.
    settled=bool(failure_reason is None and has_hold and max_v < 0.002 and drift < 0.004)
    summary=dict(stage='13D_V1', benchmark='Iaconeta cantilever self-weight',
                 status=('numerical failure, last committed state reported' if failure_reason
                         else 'dynamic-relaxed candidate; NOT fully validated' if settled
                         else 'dynamic / quasi-static NOT established'),
                 failure_reason=failure_reason,
                 failed_requested_step=failure_requested_step,
                 quasi_static_screen_passed=settled,
                 config=dict(vars(config)) if hasattr(config,'__dict__') else
                     {name:getattr(config,name) for name in config.__dataclass_fields__},
                 geometry=dict(length_m=LENGTH_M,height_m=HEIGHT_M,thickness_m=THICKNESS_M),
                 reference=dict(scaled_beam_formula_m=analytic['total_m'],
                                scaled_beam_formula_at_monitor_m=analytic_near_tip['total_m'],
                                published_beam_full_gravity_m=REFERENCE_BEAM_M,
                                published_fine_mesh_m=REFERENCE_FINE_M,
                                source='Iaconeta et al. (2020), section 4.5.1'),
                 n_particles=int(solver.particles.n_particles),
                 total_mass_kg=float(np.sum(solver.particles.mass)),
                 clamped_nodal_count=int(np.count_nonzero(solver.grid.velocity_bc_mask[:,0])),
                 last_particle_monitor_x_m=final['near_tip_x0_m'],
                 steps_requested=steps, steps=int(solver.step_count), physical_time_s=float(solver.time),
                 tip_uy_m=tip,
                 difference_from_scaled_beam_formula_m=(tip-analytic['total_m'] if settled else None),
                 difference_from_monitor_analytical_m=(tip-analytic_near_tip['total_m'] if settled else None),
                 relative_difference_at_monitor=(abs(tip-analytic_near_tip['total_m'])/abs(analytic_near_tip['total_m']) if settled else None),
                 relative_difference_to_scaled_beam_formula=(abs(tip-analytic['total_m'])/abs(analytic['total_m']) if settled else None),
                 difference_from_published_static_m=(tip-REFERENCE_FINE_M if settled and math.isclose(config.gravity,GRAVITY_M_S2) else None),
                 relative_difference_to_published_static=(abs(tip-REFERENCE_FINE_M)/abs(REFERENCE_FINE_M) if settled and math.isclose(config.gravity,GRAVITY_M_S2) else None),
                 tip_drift_last_window_m=float(drift),
                 max_particle_speed_last_window_m_s=float(max_v),
                 min_J=float(min(o['min_J'] for o in observations)))
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    if plot:
        import matplotlib.pyplot as plt
        fig, ax=plt.subplots(figsize=(8,4.5))
        ax.plot([o['time_s'] for o in observations], [o['near_tip_uy_m'] for o in observations],label='GeoPyMPM near-tip (dynamic)')
        ax.axhline(analytic_near_tip['total_m'], linestyle='--',label='Beam analytical at monitored x (static)')
        if math.isclose(config.gravity, GRAVITY_M_S2):
            ax.axhline(REFERENCE_FINE_M,linestyle=':',label='Published full-gravity fine-mesh (static)')
        ax.set(xlabel='Physical time (s)',ylabel='Vertical displacement (m)',title='Stage 13D_V1 — cantilever self-weight')
        ax.legend();fig.tight_layout();fig.savefig(out/'tip_history.png',dpi=150);plt.close(fig)
        fig, ax=plt.subplots(figsize=(9,3.4))
        scat=ax.scatter(p.position[:,0],p.position[:,1],c=displ[:,1],s=12)
        ax.plot([0, LENGTH_M], [0,0],linestyle='--',color='gray',alpha=.3)
        ax.plot([0, LENGTH_M], [HEIGHT_M,HEIGHT_M],linestyle='--',color='gray',alpha=.3)
        ax.set_aspect('equal',adjustable='box')
        ax.set(xlabel='x (m)',ylabel='y (m)',title='Particle configuration at last committed step')
        fig.colorbar(scat,ax=ax,label='Vertical displacement (m)')
        fig.tight_layout();fig.savefig(out/'beam_shape.png',dpi=160);plt.close(fig)
        fig, ax=plt.subplots(figsize=(8,4.5))
        ax.plot([o['time_s'] for o in observations],[o['max_particle_speed_m_s'] for o in observations],label='Maximum particle speed')
        ax.set(xlabel='Physical time (s)',ylabel='Speed (m/s)');fig.tight_layout();fig.savefig(out/'max_speed_history.png',dpi=150);plt.close(fig)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('results/stage13d_v1_cantilever'))
    parser.add_argument('--steps',type=int,default=30000)
    parser.add_argument('--cell-size',type=float,default=0.5)
    parser.add_argument('--ppc',type=int,default=2)
    parser.add_argument('--dt',type=float,default=0.0002)
    parser.add_argument('--ramp-time',type=float,default=1.5)
    parser.add_argument('--gravity-scale',type=float,default=0.1,help='0.1 = stable small-load verification; 1.0 = full published gravity (challenging)')
    parser.add_argument('--material',choices=['linear','cnh'],default='linear')
    parser.add_argument('--damping-rate',type=float,default=2.0)
    parser.add_argument('--pic-fraction',type=float,default=0.0)
    parser.add_argument('--scheme',choices=['usl','usf','musl'],default='usl')
    parser.add_argument('--interpolation',choices=['mpm','gimp','cpgimp','cpdi'],default='mpm')
    parser.add_argument('--sample-every',type=int,default=50)
    parser.add_argument('--progress-every',type=int,default=500,
                        help='Print live progress every N completed steps (0=off; default 500)')
    parser.add_argument('--vtu-every',type=int,default=0)
    parser.add_argument('--plot',action='store_true')
    a=parser.parse_args(argv)
    conf=BeamConfig(cell_size=a.cell_size,particles_per_axis=a.ppc,dt=a.dt,
                    ramp_time=a.ramp_time,velocity_damping_rate=a.damping_rate,
                    pic_fraction=a.pic_fraction,scheme=a.scheme,interpolation=a.interpolation,
                    gravity=GRAVITY_M_S2*a.gravity_scale,material=a.material)
    summary=run_case(output_dir=a.output_dir,config=conf,steps=a.steps,
                     sample_every=a.sample_every,vtu_every=a.vtu_every,plot=a.plot,
                     progress_every=a.progress_every)
    print(f"Stage 13D_V1: {summary['status']}")
    print(f"time={summary['physical_time_s']:.6f} s, near-tip uy={summary['tip_uy_m']:.8f} m")
    print(f"small-deflection beam formula at applied g: {analytical_beam_deflection(g=conf.gravity)['total_m']:.6f} m")
    print(f"published full-gravity fine-mesh STATIC uy={REFERENCE_FINE_M:.5f} m (different loading if gravity-scale!=1)")
    print(f"wrote: {a.output_dir}")
    if summary["failure_reason"]:
        print("NUMERICAL FAILURE (run not validated):", summary["failure_reason"])
        raise SystemExit(2)
    return summary


if __name__=='__main__':
    main()
