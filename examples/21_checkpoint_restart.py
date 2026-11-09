"""Stage 10G-B2: reconstruct and continue a CPDI simulation from a B1 archive.

Three modes (run from the repository root):

    python examples/21_checkpoint_restart.py --mode compare
    python examples/21_checkpoint_restart.py --mode save --output-dir results/b2_demo
    python examples/21_checkpoint_restart.py --mode resume --output-dir results/b2_demo

Compare mode performs a 12-step continuous simulation and a 6+6 split
simulation, asserting *exact* particle, grid, and diagnostics equivalence.
Save and resume modes run as separate Python processes and continue the
existing VisIt/ParaView output series. New output directory required for save.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.io import (
    OutputManager, load_checkpoint, restore_output_manager, write_checkpoint,
)
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver


def build_solver() -> ExplicitMPMSolver:
    xs = np.linspace(1.125, 2.375, 6)
    x, y = np.meshgrid(xs, xs, indexing="xy")
    coords = np.column_stack((x.ravel(), y.ravel()))
    n = len(coords)
    particles = ParticleSet(
        coords, np.ones(n), np.full(n, .25**2),
        velocity=np.column_stack((np.full(n, .04), np.zeros(n))),
        domain_size=np.full((n, 2), .20),
    )
    particles.stress[:, 0] = -25.
    grid = BackgroundGrid((0., 0.), (.25, .25), (14, 14))
    return ExplicitMPMSolver(
        particles, grid, LinearElastic(1200., .25),
        dt=.002, interpolation="cpdi", pic_fraction=0.,
        stress_update="corotational", kinematics_update="exponential",
    )


def create_output(solver: ExplicitMPMSolver, folder: Path) -> OutputManager:
    out = OutputManager(folder / "visualization", every_n_steps=2, every_dt=.005,
                        particles=True, grid=True, cpdi_domains=True)
    out.write_initial(solver)
    return out


def advance(solver: ExplicitMPMSolver, output: OutputManager, count: int):
    diagnostics = []
    for _ in range(count):
        d = solver.step()
        output.on_step_completed(solver, d)
        diagnostics.append(d)
    return diagnostics


def compare_states(a: ExplicitMPMSolver, b: ExplicitMPMSolver) -> None:
    assert a.time == b.time and a.step_count == b.step_count
    for name in vars(a.particles):
        np.testing.assert_array_equal(getattr(a.particles, name),
                                      getattr(b.particles, name), err_msg=name)
    for name, original in vars(a.grid).items():
        after = getattr(b.grid, name)
        if isinstance(original, np.ndarray):
            np.testing.assert_array_equal(original, after, err_msg=name)
        else:
            assert original == after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("compare", "save", "resume"), default="compare")
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage10g_b2"))
    parser.add_argument("--steps-after-restart", type=int, default=6)
    args = parser.parse_args()
    if args.steps_after_restart < 1:
        parser.error("--steps-after-restart must be >= 1")
    root = args.output_dir
    checkpoint = root / "checkpoints" / "state_00000006.json"

    if args.mode == "resume":
        solver = load_checkpoint(checkpoint)
        out = restore_output_manager(checkpoint, solver)
        advance(solver, out, args.steps_after_restart)
        print(f"Restart from step 6: now step={solver.step_count}, time={solver.time:.6f} s")
        print(f"VisIt series: {out.directory / 'particles.visit'}")
        return

    if root.exists():
        parser.error(f"Output directory already exists: {root}; choose a NEW --output-dir")

    if args.mode == "save":
        solver = build_solver()
        out = create_output(solver, root)
        advance(solver, out, 6)
        record = write_checkpoint(solver, root / "checkpoints", output_manager=out)
        print(f"Committed checkpoint: {record.manifest_path}")
        print("Now run this command in a separate Python process:")
        print(f"  python examples/21_checkpoint_restart.py --mode resume --output-dir {root}")
        return

    continuous = build_solver()
    split = build_solver()
    continuous_out = create_output(continuous, root / "continuous")
    split_out = create_output(split, root / "checkpoint_restart")
    advance(continuous, continuous_out, 6)
    advance(split, split_out, 6)
    record = write_checkpoint(split, root / "checkpoint_restart" / "checkpoints",
                              output_manager=split_out)
    restarted = load_checkpoint(record.manifest_path)
    recovered_out = restore_output_manager(record.manifest_path, restarted)
    compare_states(split, restarted)
    diag_a = advance(continuous, continuous_out, args.steps_after_restart)
    diag_b = advance(restarted, recovered_out, args.steps_after_restart)
    assert diag_a == diag_b, "Timestep diagnostics differ after restart"
    compare_states(continuous, restarted)
    assert continuous_out.frames == recovered_out.frames
    for stream in recovered_out.enabled:
        for extension in ("visit", "pvd"):
            assert (continuous_out.directory / f"{stream}.{extension}").read_bytes() == (
                recovered_out.directory / f"{stream}.{extension}").read_bytes()
    print(f"PASS: 6+{args.steps_after_restart} restart = continuous (bitwise-equal state)")
    print(f"Checkpoint: {record.manifest_path}")
    print(f"Particles: {restarted.particles.n_particles}, method={restarted.interpolation}")
    print(f"Final time: {restarted.time:.6f} s, step={restarted.step_count}")
    print(f"Restored animation: {recovered_out.directory / 'particles.visit'}")


if __name__ == "__main__":
    main()
