"""Sulsky / Uintah colliding elastic disks, a **surrogate** validation example.

Reference: Uintah User Guide Sec. 7.6, input disks_sulsky.ups, and
Sulsky, Chen & Schreyer (1994), Sec. 7.3.

Run from repository root:
    python examples/23_sulsky_colliding_disks.py --steps 20 --output-dir results/sulsky_smoke
    python examples/23_sulsky_colliding_disks.py --output-dir results/sulsky_full

IMPORTANT: GeoPyMPM Stage 10G-D4 uses incremental linear elasticity, whereas
Uintah disks_sulsky.ups uses compressible Neo-Hookean elasticity. The input
geometry and moduli are matched; this is NOT quantitative code-to-code
validation and not an exact reproduction of the Uintah constitutive response.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.io import OutputManager
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver


DOMAIN = 1.0  # m (each horizontal direction)
THICKNESS = 0.05  # m; plane-strain cylinder thickness in Uintah
CELL = 0.05  # m
N_CELLS = 20
RADIUS = 0.2  # m
RHO = 1000.0  # kg/m^3
YOUNG = 1000.0  # Pa
NU = 0.30
CENTER_A = (0.25, 0.25)
CENTER_B = (0.75, 0.75)
INITIAL_V = (0.1, 0.1)
DEFAULT_DT = 0.001  # s, Sulsky (1994) fixed dt, not Uintah adaptive dt
DEFAULT_DURATION = 3.0  # s, Uintah guide


def seed_disks(*, particles_per_cell_axis: int = 2) -> ParticleSet:
    """Seed two disks on uniform sub-cell centers with correct 2D mass/volume.

    For the Uintah res=[2,2,2] in a single thickness cell, integrate the
    two through-thickness layers into a single 2D particle of full thickness.
    Geometric inclusion is by particle CENTER (as in regular point seeding),
    so finite particle count approximates, rather than exactly equals, pi*r^2.

    ``material_id`` 0 / 1 is a diagnostic label for disk identity. Both IDs
    receive identical elastic properties and share one nodal velocity field.
    Uintah defines both bodies in one material; these labels do NOT introduce
    a contact/multi-field algorithm in GeoPyMPM.
    """
    ppc = particles_per_cell_axis
    if isinstance(ppc, bool) or not isinstance(ppc, int) or ppc < 1:
        raise ValueError("particles_per_cell_axis must be a positive integer")
    subcell = CELL / ppc
    coord = (np.arange(N_CELLS * ppc, dtype=np.float64) + 0.5) * subcell
    xx, yy = np.meshgrid(coord, coord, indexing="xy")
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    a = np.sum((xy - CENTER_A) ** 2, axis=1) <= RADIUS**2
    b = np.sum((xy - CENTER_B) ** 2, axis=1) <= RADIUS**2
    if not np.any(a) or not np.any(b) or np.any(a & b):
        raise ValueError("Disks could not be sampled without overlap")
    pos = np.concatenate((xy[a], xy[b]), axis=0)
    count_a, count_b = int(a.sum()), int(b.sum())
    n = count_a + count_b
    velocity = np.zeros((n, 2), dtype=np.float64)
    velocity[:count_a] = INITIAL_V
    velocity[count_a:] = -np.asarray(INITIAL_V, dtype=np.float64)
    volume = np.full(n, subcell**2 * THICKNESS, dtype=np.float64)
    mass = volume * RHO
    material_id = np.concatenate((np.zeros(count_a, dtype=np.int32),
                                  np.ones(count_b, dtype=np.int32)))
    return ParticleSet(pos, mass, volume, velocity=velocity, material_id=material_id,
                       domain_size=np.full((n, 2), subcell, dtype=np.float64))


def make_solver(*, dt: float = DEFAULT_DT, interpolation: str = "mpm",
                pic_fraction: float = 0.0, particles_per_cell_axis: int = 2
                ) -> ExplicitMPMSolver:
    """Return initialized solver with fixed boundary nodes on x/y faces.

    `mpm` matches Uintah's linear basis. `gimp` is an optional GeoPyMPM
    sensitivity study and should not be presented as an identical Uintah run.
    FLIP (`pic_fraction=0`) is used as an approximately non-dissipative baseline.
    """
    if interpolation not in ("mpm", "gimp"):
        raise ValueError("Use 'mpm' for reference or 'gimp' for sensitivity")
    particles = seed_disks(particles_per_cell_axis=particles_per_cell_axis)
    grid = BackgroundGrid((0.0, 0.0), (CELL, CELL), (N_CELLS, N_CELLS))
    at_boundary = np.any(
        np.isclose(grid.node_coordinates, 0.0, rtol=0.0, atol=1e-14)
        | np.isclose(grid.node_coordinates, DOMAIN, rtol=0.0, atol=1e-14),
        axis=1,
    )
    grid.velocity_bc_mask[at_boundary, :] = True
    material = LinearElastic(YOUNG, NU)
    return ExplicitMPMSolver(
        particles, grid, {0: material, 1: material}, dt=dt,
        interpolation=interpolation, pic_fraction=pic_fraction,
        gravity=(0.0, 0.0), stress_update="corotational",
        kinematics_update="euler", advection_update="euler",
    )


def approximate_elastic_energy(p: ParticleSet) -> float:
    """Approximate small-strain elastic recoverable energy, NOT Neo-Hookean W.

    Uses current Cauchy stresses and isotropic linear elastic compliance:
       W = 0.5 * sigma : C^-1 : sigma, integrated over current volumes.
    Valid only as a diagnostic of a *linear* material with small strains.
    """
    s = p.stress
    x, y, z, xy = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    normal = ((x * x + y * y + z * z)
              - 2.0 * NU * (x * y + y * z + z * x)) / (2.0 * YOUNG)
    shear = xy * xy / (2.0 * (YOUNG / (2.0 * (1.0 + NU))))
    return float(np.sum((normal + shear) * p.volume))


def snapshot(solver: ExplicitMPMSolver) -> dict[str, float | int]:
    """Scalar diagnostics including momentum of EACH disk and COM separation."""
    p = solver.particles
    a = p.material_id == 0
    b = p.material_id == 1
    pa = (p.mass[a, None] * p.velocity[a]).sum(axis=0)
    pb = (p.mass[b, None] * p.velocity[b]).sum(axis=0)
    ma = float(p.mass[a].sum())
    mb = float(p.mass[b].sum())
    ca = (p.mass[a, None] * p.position[a]).sum(axis=0) / ma
    cb = (p.mass[b, None] * p.position[b]).sum(axis=0) / mb
    ke = float(0.5 * np.sum(p.mass * np.sum(p.velocity ** 2, axis=1)))
    ee = approximate_elastic_energy(p)
    return {
        "step": solver.step_count, "time_s": float(solver.time),
        "mass_a_kg": ma, "mass_b_kg": mb,
        "com_a_x_m": float(ca[0]), "com_a_y_m": float(ca[1]),
        "com_b_x_m": float(cb[0]), "com_b_y_m": float(cb[1]),
        "p_a_x_kg_m_s": float(pa[0]), "p_a_y_kg_m_s": float(pa[1]),
        "p_b_x_kg_m_s": float(pb[0]), "p_b_y_kg_m_s": float(pb[1]),
        "p_total_x_kg_m_s": float(pa[0] + pb[0]),
        "p_total_y_kg_m_s": float(pa[1] + pb[1]),
        "kinetic_energy_j": ke,
        "approx_linear_elastic_energy_j": ee,
        "approx_mechanical_energy_j": ke + ee,
        "com_distance_m": float(np.linalg.norm(cb - ca)),
        "peak_von_mises_pa": float(np.sqrt(np.max(
            0.5 * ((s[:, 0] - s[:, 1]) ** 2 +
                   (s[:, 1] - s[:, 2]) ** 2 +
                   (s[:, 2] - s[:, 0]) ** 2) + 3.0 * s[:, 3] ** 2
        ))) if (s := p.stress).size else 0.0,
    }


def run_case(*, output_dir: str | Path, steps: int, dt: float = DEFAULT_DT,
             interpolation: str = "mpm", pic_fraction: float = 0.0,
             particles_per_cell_axis: int = 2, output_every: int = 25,
             write_vtu: bool = True, make_plot: bool = False) -> dict:
    """Execute reproducible case, saving CSV/JSON diagnostics and optional VTU."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if isinstance(output_every, bool) or not isinstance(output_every, int) or output_every <= 0:
        raise ValueError("output_every must be a positive integer")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}; use a fresh directory")
    solver = make_solver(dt=dt, interpolation=interpolation,
                         pic_fraction=pic_fraction, particles_per_cell_axis=particles_per_cell_axis)
    output_dir.mkdir(parents=True, exist_ok=True)
    manager = (OutputManager(output_dir / "visualization", every_n_steps=output_every,
                             particles=True, grid=False, cpdi_domains=False)
               if write_vtu else None)
    if manager is not None:
        manager.write_initial(solver)
    rows = [snapshot(solver)]
    for step in range(steps):
        diagnostics = solver.step()
        if manager is not None:
            manager.on_step_completed(solver, diagnostics)
        rows.append(snapshot(solver))
        if (step + 1) % max(1, min(500, output_every * 5)) == 0:
            print(f"step={diagnostics.step:5d} t={diagnostics.time:8.4f} "
                  f"KE={rows[-1]['kinetic_energy_j']:.6g} J "
                  f"pa_x={rows[-1]['p_a_x_kg_m_s']:.5g}")
    diag_file = output_dir / "diagnostics.csv"
    with diag_file.open("w", encoding="utf-8", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    p0 = np.asarray([rows[0]["p_total_x_kg_m_s"], rows[0]["p_total_y_kg_m_s"]])
    last = rows[-1]
    momentum_error = max(float(np.linalg.norm(np.asarray([
        row["p_total_x_kg_m_s"], row["p_total_y_kg_m_s"]]) - p0)) for row in rows)
    initial_e = rows[0]["kinetic_energy_j"]
    summary = {
        "reference": "Uintah User Guide 7.6 / disks_sulsky.ups; Sulsky (1994) 7.3",
        "fidelity": "2D linear-elastic surrogate; NOT the compressible Neo-Hookean Uintah solution",
        "interpolation": interpolation, "pic_fraction": pic_fraction,
        "time_step_s": dt, "steps": steps, "final_time_s": solver.time,
        "particles_a": int((solver.particles.material_id == 0).sum()),
        "particles_b": int((solver.particles.material_id == 1).sum()),
        "total_mass_kg": float(solver.particles.mass.sum()),
        "analytical_disk_mass_kg": float(RHO * np.pi * RADIUS**2 * THICKNESS),
        "initial_kinetic_energy_j": initial_e,
        "final_kinetic_energy_j": last["kinetic_energy_j"],
        "final_approx_elastic_energy_j": last["approx_linear_elastic_energy_j"],
        "maximum_total_momentum_drift_kg_m_s": momentum_error,
        "minimum_kinetic_energy_j": min(row["kinetic_energy_j"] for row in rows),
        "observed_a_reversal": any(row["p_a_x_kg_m_s"] < 0.0 and row["p_a_y_kg_m_s"] < 0.0 for row in rows),
        "observed_b_reversal": any(row["p_b_x_kg_m_s"] > 0.0 and row["p_b_y_kg_m_s"] > 0.0 for row in rows),
        "output_directory": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if make_plot:
        plot_diagnostics(rows, output_dir)
    print(f"Completed: {steps} steps, {solver.time:.5f} s, "
          f"{summary['particles_a']} + {summary['particles_b']} particles")
    print(f"Momentum drift max: {momentum_error:.5g} kg m/s")
    print(f"A/B reversal observed: {summary['observed_a_reversal']} / {summary['observed_b_reversal']}")
    print(f"Diagnostics: {diag_file}")
    if manager is not None:
        print(f"ParaView: {manager.directory / 'particles.pvd'}")
        print(f"VisIt: {manager.directory / 'particles.visit'}")
    return summary


def plot_diagnostics(rows: list[dict], directory: Path) -> None:
    """Two separate optional energy and momentum plots (requires matplotlib)."""
    import matplotlib.pyplot as plt

    t = np.array([r["time_s"] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, name in (("kinetic_energy_j", "Kinetic"),
                      ("approx_linear_elastic_energy_j", "Approx. linear elastic"),
                      ("approx_mechanical_energy_j", "Sum (approx.)")):
        ax.plot(t, [r[key] for r in rows], label=name)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (J)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "energy.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, name in (("p_a_x_kg_m_s", "Disk A $p_x$"),
                      ("p_a_y_kg_m_s", "Disk A $p_y$"),
                      ("p_b_x_kg_m_s", "Disk B $p_x$"),
                      ("p_b_y_kg_m_s", "Disk B $p_y$")):
        ax.plot(t, [r[key] for r in rows], label=name)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Momentum (kg m/s)")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(directory / "momentum.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("results/sulsky_colliding_disks"))
    parser.add_argument("--steps", type=int, help="Number of steps; default round(3.0/dt)")
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--interpolation", choices=("mpm", "gimp"), default="mpm")
    parser.add_argument("--pic-fraction", type=float, default=0.0,
                        help="0.0=FLIP; 1.0=PIC. Uintah has a distinct velocity update.")
    parser.add_argument("--particles-per-cell-axis", type=int, default=2)
    parser.add_argument("--output-every", type=int, default=50)
    parser.add_argument("--no-vtu", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.dt) or args.dt <= 0.0:
        parser.error("--dt must be positive and finite")
    nsteps = args.steps if args.steps is not None else int(round(DEFAULT_DURATION / args.dt))
    run_case(output_dir=args.output_dir, steps=nsteps, dt=args.dt,
             interpolation=args.interpolation, pic_fraction=args.pic_fraction,
             particles_per_cell_axis=args.particles_per_cell_axis,
             output_every=args.output_every, write_vtu=not args.no_vtu,
             make_plot=args.plot)


if __name__ == "__main__":
    main()
