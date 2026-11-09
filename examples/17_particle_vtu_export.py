"""Stage 10G-A1: write particle states for VisIt/ParaView (no scheduler yet)."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.io import write_particles_vtu
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage10g_a1"))
    args = parser.parse_args()

    # A small 12x12 particle patch so the exported cloud is visible in VisIt.
    x = np.linspace(2.125, 4.875, 12)
    X, Y = np.meshgrid(x, x, indexing="xy")
    coords = np.column_stack([X.ravel(), Y.ravel()])
    n = len(coords)
    p = ParticleSet(
        position=coords,
        mass=np.full(n, 1.0),
        volume=np.full(n, 0.25 ** 2),
        velocity=np.column_stack([np.full(n, 0.1), np.zeros(n)]),
        material_id=(coords[:, 1] >= 3.5).astype(np.int32),
        domain_size=np.full((n, 2), 0.25),
    )
    p.stress[:, :3] = [-100.0, -60.0, -40.0]  # Pa, core is tension-positive
    p.stress[:, 3] = 10.0 + 5.0 * (coords[:, 1] - 3.5)  # sigma_xy Pa
    g = BackgroundGrid((0.0, 0.0), (0.25, 0.25), (29, 29))
    solver = ExplicitMPMSolver(
        p, g, {0: LinearElastic(1200.0, 0.25), 1: LinearElastic(1200.0, 0.25)},
        dt=0.002, interpolation="cpdi", pic_fraction=0.0,
        stress_update="corotational", kinematics_update="exponential",
    )
    for frame in range(6):
        filename = args.output_dir / f"particles_{solver.step_count:06d}.vtu"
        write_particles_vtu(p, filename, time=solver.time, step=solver.step_count, overwrite=True)
        print(f"Wrote {filename} ({p.n_particles} particle vertices)")
        if frame < 5:
            solver.step()
    print("Open an individual .vtu file in VisIt/ParaView. Automatic time series is Stage 10G-A3.")


if __name__ == "__main__":
    main()
