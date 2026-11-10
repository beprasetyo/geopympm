"""Stage 10G-D baseline benchmark. Does not modify numerical solver code.

Run (repository root):
    python -m validation.stage10g_d.audit --output-dir results/stage10g_d_audit

The first step for each interpolation is timed separately from warmed steps.
Results are hardware/runtime dependent; there are no invented pass thresholds.
Files are written into a NEW output directory, never overwriting prior runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import statistics
import sys
try:
    import resource
except ImportError:  # Available on POSIX, not Windows
    resource = None
import time

import numpy as np
import numba

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.io import OutputManager, inspect_checkpoint, load_checkpoint, write_checkpoint
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver

METHODS = ("mpm", "gimp", "cpgimp", "cpdi")


def make_case(method: str, cells: int) -> ExplicitMPMSolver:
    """Uniform 2D block, 4 PPC, one inactive border cell, gravity loaded.

    The benchmark measures algorithm overhead, not physical predictive accuracy.
    Four particle positions per cell are used for each method for comparability.
    """
    if method not in METHODS or not isinstance(cells, int) or cells < 2:
        raise ValueError("Unknown interpolation or cells < 2")
    h = 0.25
    ii, jj = np.meshgrid(np.arange(cells), np.arange(cells), indexing="ij")
    pos = np.concatenate([
        np.column_stack(((ii.ravel() + dx + 1) * h,
                         (jj.ravel() + dy + 1) * h))
        for dx in (0.25, 0.75) for dy in (0.25, 0.75)
    ])
    n = len(pos)
    density = 1000.0
    volume = np.full(n, h*h / 4.0)
    particles = ParticleSet(pos, density * volume, volume,
                            domain_size=np.full((n, 2), h / 2.0))
    grid = BackgroundGrid((0., 0.), (h, h), (cells + 2, cells + 2))
    return ExplicitMPMSolver(
        particles, grid, LinearElastic(1200., .25),
        dt=0.001, gravity=(0., -0.02),
        interpolation=method, stress_update="corotational",
        kinematics_update="exponential", advection_update="affine_exponential",
    )


def _timer(fn):
    start = time.perf_counter()
    result = fn()
    return time.perf_counter() - start, result


def _state_bytes(solver):
    return sum(v.nbytes for obj in (solver.particles, solver.grid)
               for v in vars(obj).values() if isinstance(v, np.ndarray))


def run_case(method: str, cells: int, steps: int, repeats: int, folder: Path, *, skip_vtk=False) -> dict:
    solver = make_case(method, cells)
    cold_step_sec, _ = _timer(solver.step)
    # The first timed step may include Numba JIT compilation or Numba cache load.
    # All following timings exclude JIT initialization for this method/signature.
    warm_steps = []
    for _ in range(repeats):
        duration, _ = _timer(lambda: [solver.step() for _ in range(steps)])
        warm_steps.append(duration / steps)
    raw_bytes = _state_bytes(solver)
    record_dir = folder / "checkpoints"
    write_times = []
    records = []
    for i in range(repeats):
        duration, record = _timer(lambda i=i: write_checkpoint(
            solver, record_dir, prefix=f"trial{i}"))
        write_times.append(duration)
        records.append(record)
    verify_times, load_times = [], []
    for record in records:
        duration, _ = _timer(lambda rec=record: inspect_checkpoint(rec.manifest_path))
        verify_times.append(duration)
        duration, restored = _timer(lambda rec=record: load_checkpoint(rec.manifest_path))
        if restored.step_count != solver.step_count or restored.time != solver.time:
            raise AssertionError("Checkpoint restore lost solver chronology")
        for name, val in vars(solver.particles).items():
            np.testing.assert_array_equal(val, getattr(restored.particles, name))
        load_times.append(duration)
    # VTU output may dwarf the solver, so large-scale cases can opt out.
    vtk_times = []
    viz_bytes = 0
    if not skip_vtk:
        vtk_solver = make_case(method, cells)
        viz_dir = folder / "visualization"
        manager = OutputManager(viz_dir, every_n_steps=1,
                                particles=True, grid=True,
                                cpdi_domains=(method == "cpdi"))
        duration, _ = _timer(lambda: manager.write_initial(vtk_solver))
        vtk_times.append(duration)
        for _ in range(min(steps, 8)):
            diag = vtk_solver.step()
            duration, _ = _timer(lambda d=diag: manager.on_step_completed(vtk_solver, d))
            vtk_times.append(duration)
        viz_bytes = sum(path.stat().st_size for path in viz_dir.iterdir() if path.is_file())
    return {
        "method": method, "cells_per_axis": cells, "particles": solver.particles.n_particles,
        "grid_nodes": solver.grid.n_nodes, "time_steps_per_repeat": steps,
        "repeats": repeats, "cold_first_step_sec": cold_step_sec,
        "warm_step_median_sec": statistics.median(warm_steps),
        "warm_steps_per_second": 1.0 / statistics.median(warm_steps),
        "state_array_bytes": raw_bytes,
        "checkpoint_bytes": records[0].bytes_written,
        "checkpoint_bytes_per_particle": records[0].bytes_written / solver.particles.n_particles,
        "checkpoint_to_state_ratio": records[0].bytes_written / raw_bytes,
        "checkpoint_write_median_sec": statistics.median(write_times),
        "checkpoint_verify_median_sec": statistics.median(verify_times),
        "checkpoint_load_median_sec": statistics.median(load_times),
        "vtk_frames_per_stream": len(vtk_times),
        "vtk_total_bytes": viz_bytes,
        "vtk_initial_sec": vtk_times[0] if vtk_times else None,
        "vtk_followup_median_sec": statistics.median(vtk_times[1:]) if len(vtk_times) > 1 else None,
    }


def run_audit(*, methods=METHODS, sizes=(4, 8, 16), steps=8,
              repeats=3, output_dir=Path("results/stage10g_d_audit"), skip_vtk=False):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Audit refuses to overwrite previous results: {output_dir}")
    output_dir.mkdir(parents=True)
    results = []
    for method in methods:
        for cells in sizes:
            print(f"Auditing {method}: {4*cells*cells} particles", flush=True)
            results.append(run_case(method, cells, steps, repeats,
                                    output_dir / method / f"cells_{cells}", skip_vtk=skip_vtk))
    meta = {
        "stage": "10G-D preliminary baseline audit",
        "baseline": "10G-C; no changes to src/geopympm",
        "checkpoint_schema": 1,
        "platform": platform.platform(), "python": sys.version,
        "numpy": np.__version__, "numba": numba.__version__,
        "cpu_count": os.cpu_count(),
        "peak_process_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if (resource is not None and sys.platform.startswith("linux")) else None,
        "notes": [
            "Cold first step includes Numba initialization; warmed steps do not.",
            "Peak process RSS is cumulative across cases, not per-case and may include JIT caches.",
            "The I/O benchmark uses compressible near-static fields; real heterogeneous states may be larger.",
            "No hardware-independent pass/fail throughput criterion has been set.",
            "VTK times are null for skip-vtk cases.",
            "No contact, elastoplastic material, or 3D support is implied.",
        ],
        "measurements": results,
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(meta, indent=2) + "\n")
    with (output_dir / "measurements.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage10g_d_audit"))
    parser.add_argument("--sizes", type=int, nargs="+", default=(4, 8, 16))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-vtk", action="store_true", help="Avoid many large VTK files")
    args = parser.parse_args()
    if not args.steps > 0 or not args.repeats > 0:
        parser.error("steps and repeats must be positive")
    meta = run_audit(methods=args.methods, sizes=args.sizes, steps=args.steps,
                     repeats=args.repeats, output_dir=args.output_dir, skip_vtk=args.skip_vtk)
    print("Audit summary:", args.output_dir / "audit_summary.json")
    print("Results:", len(meta["measurements"]), "cases")


if __name__ == "__main__":
    main()
