"""Independent Stage 10G-B2: bitwise continuous-vs-restart validation.

Run from repo root:
    python -m validation.stage10g_b2.validate_restart

Checks direct-vs-restored 5+5 timestep trajectories, material mapping,
transient grid fields, BCs, external forces and independent output indexes.
Writes a short JSON report; B1 checkpoint schema remains version 1.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.io import (
    OutputManager, load_checkpoint, restore_output_manager, write_checkpoint,
)
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver

METHODS = ("mpm", "gimp", "cpgimp", "cpdi")


def make_solver(method):
    x, y = np.meshgrid(np.array([.625, .875]), np.array([.625, .875]), indexing="xy")
    pos = np.column_stack((x.ravel(), y.ravel()))
    p = ParticleSet(pos, np.ones(4), np.full(4, .0625),
                    material_id=np.array([0, 1, 1, 0]),
                    velocity=np.tile((.03, -.01), (4, 1)),
                    domain_size=np.full((4, 2), .2))
    p.stress[:, 0] = -15.
    g = BackgroundGrid((0., 0.), (.25, .25), (8, 8))
    g.velocity_bc_mask[g.node_id(0, 0), 0] = True
    g.velocity_bc_value[g.node_id(0, 0), 0] = 0.
    f = np.zeros((g.n_nodes, 2))
    f[g.node_id(3, 3), 0] = .01
    return ExplicitMPMSolver(
        p, g, {0: LinearElastic(1000., .25), 1: LinearElastic(1500., .28)},
        dt=.001, pic_fraction=0., gravity=(0., -.02),
        nodal_external_force=f, stress_update="corotational",
        kinematics_update="exponential", advection_update="affine_exponential",
        interpolation=method,
    )


def new_output(solver, directory):
    out = OutputManager(directory, every_n_steps=2, every_dt=.0015,
                        particles=True, grid=True,
                        cpdi_domains=(solver.interpolation == "cpdi"))
    out.write_initial(solver)
    return out


def equal_all_arrays(a, b, container):
    return all(np.array_equal(getattr(a, key), getattr(b, key))
               for key, value in vars(a).items() if isinstance(value, np.ndarray))


def run_validation(folder: Path) -> dict:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    summary = {"stage": "10G-B2", "schema": 1, "methods": {},
               "caveats": "Only 2D LinearElastic Stage-10G-B1 schema; no plastic state, authentication, or cross-platform bitwise guarantee."}
    for method in METHODS:
        root = folder / method
        full = make_solver(method)
        split = make_solver(method)
        a_out = new_output(full, root / "continuous")
        b_out = new_output(split, root / "restarted")
        for _ in range(5):
            a_out.on_step_completed(full, full.step())
            b_out.on_step_completed(split, split.step())
        cp = write_checkpoint(split, root / "checkpoints", output_manager=b_out)
        fresh = load_checkpoint(cp.manifest_path)
        b_out = restore_output_manager(cp.manifest_path, fresh)
        before_particle = equal_all_arrays(split.particles, fresh.particles, "particle")
        before_grid = equal_all_arrays(split.grid, fresh.grid, "grid")
        diagnostics_match = True
        for _ in range(5):
            d1 = full.step()
            d2 = fresh.step()
            diagnostics_match &= d1 == d2
            a_out.on_step_completed(full, d1)
            b_out.on_step_completed(fresh, d2)
        particle_equal = equal_all_arrays(full.particles, fresh.particles, "particle")
        grid_equal = equal_all_arrays(full.grid, fresh.grid, "grid")
        series_equal = a_out.frames == b_out.frames
        indices_equal = all(
            (a_out.directory / f"{kind}.{ext}").read_bytes() ==
            (b_out.directory / f"{kind}.{ext}").read_bytes()
            for kind in b_out.enabled for ext in ("visit", "pvd")
        )
        config_equal = (full.interpolation == fresh.interpolation and
                        full.materials == fresh.materials and
                        np.array_equal(full.nodal_external_force, fresh.nodal_external_force) and
                        np.array_equal(full.gravity, fresh.gravity) and
                        full.step_count == fresh.step_count and full.time == fresh.time)
        checks = {
            "initial_particle_state_exact": before_particle,
            "initial_grid_state_exact": before_grid,
            "subsequent_diagnostics_exact": diagnostics_match,
            "final_particle_state_exact": particle_equal,
            "final_grid_state_exact": grid_equal,
            "materials_forces_clock_exact": config_equal,
            "output_frames_exact": series_equal,
            "visit_and_pvd_indexes_exact": indices_equal,
        }
        summary["methods"][method] = {
            "checks": checks,
            "all_passed": bool(all(checks.values())),
            "restart_step": cp.step,
            "final_step": fresh.step_count,
            "final_time": fresh.time,
        }
    summary["checks_total"] = 8 * len(METHODS)
    summary["checks_passed"] = sum(sum(v["checks"].values()) for v in summary["methods"].values())
    summary["all_passed"] = summary["checks_passed"] == summary["checks_total"]
    return summary


def main():
    dest = Path("validation/results/stage10g_b2")
    output = run_validation(dest / "cases")
    (dest / "restart_validation.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    if not output["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
