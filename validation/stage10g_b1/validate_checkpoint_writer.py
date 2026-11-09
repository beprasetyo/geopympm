"""Validate Stage 10G-B1 checkpoint data against in-memory solver state.

Run: python -m validation.stage10g_b1.validate_checkpoint_writer
Optionally: --output-dir results/stage10g_b1_validation

This checks serialization and numeric equality, NOT solver restart.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.io import write_checkpoint, inspect_checkpoint
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver


def solver_for_mode(mode: str) -> ExplicitMPMSolver:
    x, y = np.meshgrid([.625, .875], [.625, .875], indexing="xy")
    p = ParticleSet(np.column_stack((x.ravel(), y.ravel())),
                    np.ones(4), np.full(4, .0625),
                    domain_size=np.full((4, 2), .2),
                    velocity=np.tile([.02, -.01], (4, 1)))
    p.stress[:, 0] = -10.
    g = BackgroundGrid((0., 0.), (.25, .25), (8, 8))
    g.velocity_bc_mask[g.node_id(0, 0), 1] = True
    return ExplicitMPMSolver(p, g, LinearElastic(1000., .25),
                             dt=.001, interpolation=mode, pic_fraction=0,
                             stress_update="corotational",
                             kinematics_update="exponential")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=Path("results/stage10g_b1_validation"))
    args = ap.parse_args()
    per_method = {}
    for method in ("mpm", "gimp", "cpgimp", "cpdi"):
        solver = solver_for_mode(method)
        for _ in range(4):
            solver.step()
        particle_before = {key: value.copy() for key, value in vars(solver.particles).items()}
        grid_before = {key: value.copy() for key, value in vars(solver.grid).items()
                       if isinstance(value, np.ndarray)}
        record = write_checkpoint(solver, args.output_dir / method)
        manifest = inspect_checkpoint(record.manifest_path)
        with np.load(record.data_path, allow_pickle=False) as arrays:
            checks = {
                "particle_arrays_identical": all(np.array_equal(arrays[f"particle.{k}"], v)
                                                 for k, v in particle_before.items()),
                "grid_arrays_identical": all(np.array_equal(arrays[f"grid.{k}"], v)
                                             for k, v in grid_before.items()),
                "time_and_step_saved": manifest["step"] == solver.step_count
                                       and manifest["time"] == solver.time,
                "material_saved": manifest["materials"]["kind"] == "single",
                "solver_configuration_saved": manifest["config"]["interpolation"] == method,
                "intact_after_write": all(np.array_equal(getattr(solver.particles, k), v)
                                          for k, v in particle_before.items()),
                "only_numeric_arrays": all(arrays[k].dtype.kind != "O" for k in arrays.files),
                "integrity_verified": manifest["schema_version"] == 1,
            }
        per_method[method] = {"checks": checks, "all_passed": all(checks.values()),
                              "size_bytes": record.bytes_written,
                              "array_count": record.array_count,
                              "step": record.step, "time": record.time}
    result = {"stage": "10G-B1", "checkpoint_schema": 1,
              "methods": per_method,
              "checks_passed": sum(sum(v["checks"].values()) for v in per_method.values()),
              "checks_total": sum(len(v["checks"]) for v in per_method.values()),
              "all_passed": all(v["all_passed"] for v in per_method.values()),
              "caveat": "Writer/inspection validation only; NO restart is implemented."}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
