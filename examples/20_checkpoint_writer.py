"""Stage 10G-B1: write inspectable full-state checkpoints from CPDI solver.

Run:
    python examples/20_checkpoint_writer.py

This example ONLY WRITES and VERIFIES the snapshots. Stage 10G-B2 will
implement reconstruction and continuation from these archives.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.io import OutputManager, inspect_checkpoint, write_checkpoint
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver


def build_solver():
    xs = np.linspace(1.125, 2.375, 6)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    coords = np.column_stack((X.ravel(), Y.ravel()))
    n = coords.shape[0]
    particles = ParticleSet(
        coords, np.ones(n), np.full(n, .25**2),
        velocity=np.column_stack((np.full(n, .04), np.zeros(n))),
        domain_size=np.full((n, 2), .20),
    )
    particles.stress[:, 0] = -25.
    grid = BackgroundGrid((0., 0.), (.25, .25), (14, 14))
    return ExplicitMPMSolver(
        particles, grid, LinearElastic(1200., .25), dt=.002,
        interpolation="cpdi", pic_fraction=0.,
        stress_update="corotational", kinematics_update="exponential",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/stage10g_b1"))
    args = parser.parse_args()
    solver = build_solver()
    output = OutputManager(args.output_dir / "visualization",
                           every_n_steps=2, particles=True,
                           grid=True, cpdi_domains=True)
    output.write_initial(solver)
    for _ in range(6):
        diag = solver.step()
        output.on_step_completed(solver, diag)
        if diag.step in (3, 6):
            record = write_checkpoint(solver, args.output_dir / "checkpoints",
                                      output_manager=output)
            metadata = inspect_checkpoint(record.manifest_path)
            print(f"Checkpoint at step {metadata['step']}, t={metadata['time']:.6f}s:")
            print(f"  {record.data_path} ({record.bytes_written} bytes pair)")
            print(f"  {record.manifest_path} ({record.array_count} arrays)")
    print("B1 verifies disk data; restart is Stage 10G-B2.")


if __name__ == "__main__":
    main()
