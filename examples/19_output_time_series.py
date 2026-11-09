"""Stage 10G-A3: force-driven CPDI solver with automated VisIt/ParaView series.

The numerical state is produced by ExplicitMPMSolver.step(), not manually
prescribed affine motion. Time-series writer is outside the solver loop.

Run: python examples/19_output_time_series.py --output-dir results/stage10g_a3
Then open visualization/particles.visit or particles.pvd; grid and domains
have their own .visit/.pvd files.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.io import OutputManager
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver


def build_solver():
    xs = np.linspace(1.125, 2.375, 6)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    coords = np.column_stack((X.ravel(), Y.ravel()))
    n = len(coords)
    particles = ParticleSet(
        coords, np.full(n, 1.0), np.full(n, .25**2),
        velocity=np.column_stack((np.full(n, .04), np.zeros(n))),
        domain_size=np.full((n, 2), .20),
    )
    particles.stress[:, 0] = -25.0
    grid = BackgroundGrid((0., 0.), (.25, .25), (14, 14))
    return ExplicitMPMSolver(
        particles, grid, LinearElastic(1200., .25), dt=.002,
        interpolation="cpdi", pic_fraction=0.,
        stress_update="corotational", kinematics_update="exponential",
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir", type=Path, default=Path("results/stage10g_a3"))
    cli.add_argument("--steps", type=int, default=8)
    args = cli.parse_args()
    solver = build_solver()
    output = OutputManager(
        args.output_dir / "visualization",
        every_n_steps=2, every_dt=.005,
        particles=True, grid=True, cpdi_domains=True,
    )
    print(f"Initial: {list(output.write_initial(solver))}")
    for _ in range(args.steps):
        diagnostics = solver.step()
        files = output.on_step_completed(solver, diagnostics)
        if files:
            print(f"Step {diagnostics.step:3d}, t={diagnostics.time:.6f} s: "
                  f"{', '.join(p.name for p in files.values())}")
    print("\nOpen these animated time series in VisIt or ParaView:")
    for kind in output.enabled:
        print("  ", output.directory / f"{kind}.visit")
        print("  ", output.directory / f"{kind}.pvd")


if __name__ == "__main__":
    main()
