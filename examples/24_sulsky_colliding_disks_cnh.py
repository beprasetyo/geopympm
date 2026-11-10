"""Stage 11E: Sulsky colliding disks with *compressible Neo-Hookean* elasticity.

Matches elastic/isothermal stress branch of Uintah UCNH (not solver equivalence):
  Uintah-master/src/StandAlone/inputs/UintahRelease/MPM/disks_sulsky.ups
  Uintah-master/src/CCA/Components/MPM/Materials/ConstitutiveModel/UCNH.cc

Run from repository root:
  python examples/24_sulsky_colliding_disks_cnh.py --steps 50 --no-vtu
  python examples/24_sulsky_colliding_disks_cnh.py --plot

Output time is in seconds, not step/cycle count. No dedicated multimaterial contact,
Uintah's artificial viscosity, 3D through-thickness kinematics, or adaptive dt.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.io import OutputManager
from geopympm.materials import CompressibleNeoHookean, neo_hookean_strain_energy
from geopympm.solvers import ExplicitMPMSolver


RADIUS = 0.2
CELL = 0.05
THICKNESS = 0.05
RHO = 1000.0
INITIAL_V = 0.1
E = 1000.0
NU = 0.3
DT = 0.0005  # CNH: 0.001 generated strong late-time energy drift in this solver


def make_solver(*, dt: float = DT, ppc: int = 2, interpolation: str = "mpm"):
    """Use a shared velocity field (like Uintah's null contact example)."""
    if not isinstance(ppc, int) or isinstance(ppc, bool) or ppc < 1:
        raise ValueError("ppc must be positive integer")
    if interpolation not in ("mpm", "gimp", "cpgimp", "cpdi"):
        raise ValueError("Unknown interpolation method")
    step = CELL / ppc
    coords = (np.arange(20*ppc, dtype=float) + 0.5)*step
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    a = np.sum((xy - (0.25,0.25))**2, axis=1) <= RADIUS**2
    b = np.sum((xy - (0.75,0.75))**2, axis=1) <= RADIUS**2
    n_a, n_b = int(a.sum()), int(b.sum())
    if not n_a or not n_b or np.any(a & b):
        raise ValueError("Disk seeding failed")
    position = np.concatenate((xy[a], xy[b]), axis=0)
    vel = np.empty_like(position)
    vel[:n_a] = INITIAL_V
    vel[n_a:] = -INITIAL_V
    vol = np.full(n_a+n_b, step*step*THICKNESS)
    parts = ParticleSet(position, RHO*vol, vol, velocity=vel,
                        material_id=np.array([0]*n_a+[1]*n_b, dtype=np.int32),
                        domain_size=np.full_like(position, step))
    grid = BackgroundGrid((0.,0.), (CELL,CELL), (20,20))
    boundary = np.any(np.isclose(grid.node_coordinates, 0., atol=1e-14)
                      | np.isclose(grid.node_coordinates, 1., atol=1e-14), axis=1)
    grid.velocity_bc_mask[boundary] = True
    mat = CompressibleNeoHookean.from_young_poisson(E,NU)
    return ExplicitMPMSolver(parts, grid, {0:mat,1:mat}, dt=dt, cfl=0.2,
                             stress_update="hyperelastic", pic_fraction=0.0,
                             kinematics_update="exponential", interpolation=interpolation)


def snapshot(solver):
    p = solver.particles
    ma = p.material_id == 0
    mb = ~ma
    momentum = p.mass[:,None]*p.velocity
    pa = momentum[ma].sum(axis=0)
    pb = momentum[mb].sum(axis=0)
    ke = float(0.5*np.sum(p.mass * np.sum(p.velocity**2,axis=1)))
    elastic = neo_hookean_strain_energy(p, solver.materials)
    com_a = (p.mass[ma,None]*p.position[ma]).sum(axis=0)/p.mass[ma].sum()
    com_b = (p.mass[mb,None]*p.position[mb]).sum(axis=0)/p.mass[mb].sum()
    return dict(step=solver.step_count, time_s=solver.time,
                kinetic_energy_j=ke, hyperelastic_energy_j=elastic,
                total_mechanical_energy_j=ke+elastic,
                momentum_a_x=pa[0], momentum_a_y=pa[1],
                momentum_b_x=pb[0], momentum_b_y=pb[1],
                momentum_total_x=pa[0]+pb[0], momentum_total_y=pa[1]+pb[1],
                center_distance_m=float(np.linalg.norm(com_b-com_a)),
                min_J=float(np.min(np.linalg.det(p.deformation_gradient))))


def run(*, directory, steps=6000, dt=DT, output_every=50, write_vtu=True, plot=False,
        interpolation="mpm"):
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if type(output_every) is not int or output_every < 1:
        raise ValueError("output_every must be positive")
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Refusing non-empty results folder: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    solver = make_solver(dt=dt, interpolation=interpolation)
    manager = (OutputManager(directory/"visualization", every_n_steps=output_every,
                             particles=True, grid=False, cpdi_domains=False)
               if write_vtu else None)
    if manager is not None:
        manager.write_initial(solver)
    rows = [snapshot(solver)]
    for j in range(steps):
        diagnostic = solver.step()
        if manager is not None:
            manager.on_step_completed(solver, diagnostic)
        rows.append(snapshot(solver))
        if (j+1) % 500 == 0:
            print(f"step={j+1:5d}, time={solver.time:.4f} s, E={rows[-1]['total_mechanical_energy_j']:.6g} J")
    with (directory/"diagnostics.csv").open("w", encoding="utf8", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    p0 = np.array([rows[0]["momentum_total_x"], rows[0]["momentum_total_y"]])
    p_drift = max(float(np.linalg.norm(np.array([r["momentum_total_x"],
                                              r["momentum_total_y"]]) - p0)) for r in rows)
    summary = dict(model="isothermal elastic UCNH", method=interpolation,
                   steps=steps, final_time_s=solver.time, dt_s=dt,
                   reference="Uintah disks_sulsky.ups / UCNH.cc (not code-to-code validated)",
                   n_particles=solver.particles.n_particles,
                   total_mass_kg=float(solver.particles.mass.sum()),
                   initial_total_energy_j=rows[0]["total_mechanical_energy_j"],
                   final_total_energy_j=rows[-1]["total_mechanical_energy_j"],
                   max_total_momentum_drift=p_drift,
                   min_J=min(row["min_J"] for row in rows),
                   rebound_a=any(row["momentum_a_x"] < 0 for row in rows),
                   rebound_b=any(row["momentum_b_x"] > 0 for row in rows))
    (directory/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf8")
    if plot:
        import matplotlib.pyplot as plt
        t = [r['time_s'] for r in rows]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for field, label in (("kinetic_energy_j","Kinetic"),
                             ("hyperelastic_energy_j","CNH elastic"),
                             ("total_mechanical_energy_j","Total")):
            ax.plot(t, [r[field] for r in rows], label=label)
        ax.set(xlabel="Time (s)", ylabel="Energy (J)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory/"energy.png", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for field, label in (("momentum_a_x","Disk A px"), ("momentum_b_x","Disk B px"),
                             ("momentum_total_x","Total px")):
            ax.plot(t, [r[field] for r in rows], label=label)
        ax.set(xlabel="Time (s)", ylabel="Momentum (kg m/s)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory/"momentum.png", dpi=150)
        plt.close(fig)
    print("Result:", directory/"summary.json", "momentum drift:",p_drift)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/sulsky_cnh"))
    parser.add_argument("--steps", type=int, default=None,
                        help="Default: round(3.0 / dt), total physical time 3 seconds")
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--output-every", type=int, default=50)
    parser.add_argument("--interpolation", choices=("mpm","gimp","cpgimp","cpdi"), default="mpm")
    parser.add_argument("--no-vtu", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    run(directory=args.output_dir, steps=(int(round(3.0/args.dt)) if args.steps is None else args.steps), dt=args.dt,
        output_every=args.output_every, write_vtu=not args.no_vtu,
        plot=args.plot, interpolation=args.interpolation)


if __name__ == "__main__":
    main()
