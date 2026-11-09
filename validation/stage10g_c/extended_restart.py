"""Stage 10G-C: extended checkpoint/restart verification (NO solver changes).

Run from repository root:

    python -m validation.stage10g_c.extended_restart --steps 90
    python -m validation.stage10g_c.extended_restart --steps 120 --preserve

The central oracle is a continuous solver advanced IN LOCKSTEP with a chain of
freshly reconstructed solvers.  Every committed step and every scheduler frame
is compared, not merely the last snapshot.  Checkpoint archives remain schema 1.

Scope: deterministic 2D LinearElastic MPM/uGIMP/cpGIMP/CPDI.  Not a physical
validation of general nonlinear constitutive behavior or HPC crash recovery.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.io import (OutputManager, load_checkpoint, restore_output_manager,
                         write_checkpoint, inspect_checkpoint)
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver

METHODS = ("mpm", "gimp", "cpgimp", "cpdi")


def make_case(method: str, *, dynamic: bool = False) -> ExplicitMPMSolver:
    """A small two-component motion with a nonuniform stress field.

    The dynamic scenario tests LARGE(r) geometry changes and nonzero rotation,
    but is a reproducibility test, NOT a calibrated physical benchmark.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown interpolation: {method}")
    xs = np.linspace(1.125, 2.375, 6)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    x = np.column_stack((X.ravel(), Y.ravel()))
    n = x.shape[0]
    vx = .045 + (1.8 if dynamic else .2) * (x[:, 1] - 1.75)
    vy = -.025 - (.7 if dynamic else .08) * (x[:, 0] - 1.75)
    velocity = np.column_stack((vx, vy))
    particle = ParticleSet(
        x, np.ones(n), np.full(n, .25**2),
        velocity=velocity,
        domain_size=np.full((n, 2), .20),
        material_id=np.asarray([i % 2 for i in range(n)], dtype=np.int32),
    )
    particle.stress[:, 0] = -20 + 2.0 * (x[:, 0] - 1.75)
    particle.stress[:, 1] = -10 - 1.5 * (x[:, 1] - 1.75)
    particle.stress[:, 3] = 1.5 * (x[:, 0] - 1.75)
    grid = BackgroundGrid((0., 0.), (.25, .25), (18, 18))
    grid.velocity_bc_mask[grid.node_id(0, 0), 0] = True
    grid.velocity_bc_value[grid.node_id(0, 0), 0] = 0.
    force = np.zeros((grid.n_nodes, 2))
    force[grid.node_id(7, 7), 1] = -.02
    return ExplicitMPMSolver(
        particle, grid, {0: LinearElastic(1200., .25), 1: LinearElastic(1500., .28)},
        dt=.002, pic_fraction=0., gravity=(0., -.015),
        nodal_external_force=force, interpolation=method,
        stress_update="corotational", kinematics_update="exponential",
        advection_update="affine_exponential",
    )


def _compare_states(a: ExplicitMPMSolver, b: ExplicitMPMSolver) -> None:
    if (a.step_count != b.step_count or a.time != b.time or
            a.materials != b.materials or a.interpolation != b.interpolation or
            a.dt != b.dt or a.pic_fraction != b.pic_fraction or
            a.cfl != b.cfl or a.stress_update != b.stress_update or
            a.kinematics_update != b.kinematics_update or
            a.advection_update != b.advection_update):
        raise AssertionError("solver configuration or time mismatches")
    for category in ("particles", "grid"):
        for name, value in vars(getattr(a, category)).items():
            other = getattr(getattr(b, category), name)
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, other, err_msg=f"{category}.{name}")
            elif other != value:
                raise AssertionError(f"{category}.{name} differs")
    np.testing.assert_array_equal(a.gravity, b.gravity)
    if a.nodal_external_force is None:
        assert b.nodal_external_force is None
    else:
        np.testing.assert_array_equal(a.nodal_external_force, b.nodal_external_force)


def _make_output(solver, path: Path):
    out = OutputManager(path, every_n_steps=8, every_dt=.014,
                        particles=True, grid=True,
                        cpdi_domains=solver.interpolation == "cpdi")
    out.write_initial(solver)
    return out


def _check_output(a: OutputManager, b: OutputManager) -> None:
    if a.enabled != b.enabled or a.frames != b.frames:
        raise AssertionError("Time series frames differ")
    if a._next_time != b._next_time:
        raise AssertionError("Physical time scheduling differs")
    for stream in a.enabled:
        for suffix in ("visit", "pvd"):
            left = (a.directory / f"{stream}.{suffix}").read_bytes()
            right = (b.directory / f"{stream}.{suffix}").read_bytes()
            if left != right:
                raise AssertionError(f"{stream}.{suffix} differs")
        if {p.name for p in a.directory.glob(f"{stream}_*.vtu")} != {
                p.name for p in b.directory.glob(f"{stream}_*.vtu")}:
            raise AssertionError(f"{stream} snapshots differ")


def run_chain(method: str, workdir: Path, *, steps: int = 90,
              breaks: tuple[int, ...] | None = None,
              dynamic: bool = False) -> dict:
    """Compare continuous execution to a multi-restart chain at *every* step."""
    if type(steps) is not int or steps < 6:
        raise ValueError("steps must be >= 6")
    breaks = tuple(sorted({steps // 4, steps // 2, 3 * steps // 4})
                   if breaks is None else breaks)
    if (not breaks or any(type(v) is not int or v < 1 or v >= steps for v in breaks)
            or tuple(sorted(set(breaks))) != breaks):
        raise ValueError("breaks must be sorted unique integers within (0, steps)")
    workdir = Path(workdir)
    if workdir.exists():
        raise FileExistsError(f"Use an empty/new validation directory: {workdir}")
    workdir.mkdir(parents=True)
    continuous = make_case(method, dynamic=dynamic)
    resumed = make_case(method, dynamic=dynamic)
    direct_out = _make_output(continuous, workdir / "continuous")
    chain_out = _make_output(resumed, workdir / "restart")
    _compare_states(continuous, resumed)
    cp_info = []
    max_Fxy = 0.
    max_Fyx = 0.
    max_angle_deg = 0.
    min_det_f = math.inf
    checked_vtu_snapshots = 0
    for i in range(1, steps + 1):
        diag_a = continuous.step()
        diag_b = resumed.step()
        if diag_a != diag_b:
            raise AssertionError(f"Step diagnostics differ at step {i}")
        a_written = direct_out.on_step_completed(continuous, diag_a)
        b_written = chain_out.on_step_completed(resumed, diag_b)
        if a_written.keys() != b_written.keys():
            raise AssertionError(f"Different VTU emission streams at step {i}")
        for kind in a_written:
            if a_written[kind].read_bytes() != b_written[kind].read_bytes():
                raise AssertionError(f"{kind} VTU snapshot content differs at step {i}")
            checked_vtu_snapshots += 1
        _compare_states(continuous, resumed)
        _check_output(direct_out, chain_out)
        F = resumed.particles.deformation_gradient
        max_Fxy = max(max_Fxy, float(np.max(np.abs(F[:, 0, 1]))))
        max_Fyx = max(max_Fyx, float(np.max(np.abs(F[:, 1, 0]))))
        min_det_f = min(min_det_f, float(np.min(np.linalg.det(F))))
        theta = np.arctan2(F[:, 1, 0]-F[:, 0, 1], F[:, 0, 0]+F[:, 1, 1])
        max_angle_deg = max(max_angle_deg, math.degrees(float(np.max(np.abs(theta)))))
        if i in breaks:
            record = write_checkpoint(
                resumed, workdir / "checkpoints", output_manager=chain_out)
            restored = load_checkpoint(record.manifest_path)
            recovered_out = restore_output_manager(record.manifest_path, restored)
            _compare_states(resumed, restored)
            _check_output(chain_out, recovered_out)
            if inspect_checkpoint(record.manifest_path)["step"] != i:
                raise AssertionError("Checkpoint archive step mismatch")
            resumed, chain_out = restored, recovered_out
            cp_info.append({"step": i, "time": resumed.time,
                            "bytes": record.bytes_written,
                            "arrays": record.array_count,
                            "path": str(record.manifest_path)})
    _compare_states(continuous, resumed)
    _check_output(direct_out, chain_out)
    final = resumed.particles.deformation_gradient
    result = {
        "method": method, "dynamic": dynamic, "steps": steps,
        "checkpoints": cp_info, "continuous_equals_restarted": True,
        "diagnostics_match_every_step": True,
        "vtu_snapshots_byte_identical": True,
        "checked_vtu_snapshots": checked_vtu_snapshots,
        "time_series_match_every_step": True,
        "frame_counts": {k: len(v) for k, v in chain_out.frames.items()},
        "max_abs_Fxy": max_Fxy, "max_abs_Fyx": max_Fyx,
        "max_polar_angle_deg": max_angle_deg,
        "min_det_F": min_det_f,
        "final_time_s": resumed.time,
        "final_max_abs_F_minus_I": float(np.max(np.abs(final-np.eye(2)))),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--preserve", action="store_true", help="Preserve all sample VTK and checkpoint files")
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage10g_c"))
    parser.add_argument("--report", type=Path, default=Path("validation/results/stage10g_c/extended_restart.json"))
    args = parser.parse_args()
    if args.steps < 6:
        parser.error("--steps must be >= 6")
    report = {"stage": "10G-C", "schema": 1, "methods": {},
              "scope": "2D LinearElastic exact restart, not proof of physical accuracy or fault tolerance"}
    if args.preserve:
        if args.output_dir.exists():
            parser.error(f"Choose a new --output-dir (already exists: {args.output_dir})")
        args.output_dir.mkdir(parents=True)
        base = args.output_dir
        for method in METHODS:
            report["methods"][method] = run_chain(method, base / method,
                    steps=args.steps, dynamic=(method == "cpdi"))
        report["kept_results"] = str(base)
    else:
        with tempfile.TemporaryDirectory(prefix="geopympm_10gc_") as temp:
            for method in METHODS:
                report["methods"][method] = run_chain(method, Path(temp) / method,
                        steps=args.steps, dynamic=(method == "cpdi"))
        for r in report["methods"].values():
            for rec in r["checkpoints"]:
                rec.pop("path")
    report["all_passed"] = True
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
