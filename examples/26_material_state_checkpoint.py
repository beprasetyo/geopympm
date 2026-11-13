"""Stage 13A: constitutive history storage + lossless restart (NOT plasticity).

Run from repository root:
    python examples/26_material_state_checkpoint.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from geopympm.core import ParticleSet, BackgroundGrid
from geopympm.materials import LinearElastic, MaterialState, StateFieldSpec
from geopympm.solvers import ExplicitMPMSolver
from geopympm.io import inspect_checkpoint, load_checkpoint, write_checkpoint


def main() -> None:
    particles = ParticleSet(
        [[0.35, 0.42], [0.65, 0.57]],
        [0.1, 0.1], [0.002, 0.002],
        velocity=[[0.02, -0.01], [-0.02, 0.01]],
    )
    grid = BackgroundGrid([0.0, 0.0], [0.1, 0.1], (10, 10))
    state = MaterialState(particles.n_particles, [
        StateFieldSpec("plastic_strain", (6,)),
        StateFieldSpec("hardening_parameter"),
    ])
    # These values are USER-DEFINED INITIAL DATA, NOT computed plastic strains.
    state["hardening_parameter"][:] = [0.25, 0.5]

    solver = ExplicitMPMSolver(
        particles, grid, LinearElastic(1000.0, 0.3),
        dt=0.0005, scheme="musl", pic_fraction=0,
        material_state=state,
    )
    solver.run(5)
    with tempfile.TemporaryDirectory() as folder:
        record = write_checkpoint(solver, Path(folder))
        meta = inspect_checkpoint(record.manifest_path)
        restored = load_checkpoint(record.manifest_path)
        print("Checkpoint schema:", meta["schema_version"])
        print("Restored step:", restored.step_count)
        print("Restored history fields:", restored.material_state.names)
        for name in state.names:
            np.testing.assert_array_equal(state[name], restored.material_state[name])
        restored.run(2)
        print("Restarted to step:", restored.step_count)

    print("History arrays are conserved, but LinearElastic does not evolve them.")
    print("Yield surfaces, return mapping, and plastic updates are Stage 13B onward.")


if __name__ == "__main__":
    main()
