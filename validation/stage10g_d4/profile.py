"""Stage 10G-D4: profiling harness (no solver or checkpoint modifications).

Usage from the repository root::

    python -m validation.stage10g_d4.profile --sizes 8 16 --steps 8 --repeats 3 \
        --output-dir results/stage10g_d4

Run on an otherwise idle machine; compare baseline and patched runs in separate
new output directories with the *same* options, Python/Numba, and hardware.
Timings are not pass/fail numerical-validation criteria. Cold JIT is isolated.
"""
from __future__ import annotations

import argparse
import cProfile
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import pstats
import statistics
import sys
import time

import numba
import numpy as np

from geopympm.io import (OutputManager, inspect_checkpoint, load_checkpoint,
                         write_checkpoint, write_particles_vtu)
from validation.stage10g_d.audit import METHODS, make_case


def _elapsed(call):
    start = time.perf_counter()
    value = call()
    return time.perf_counter() - start, value


def _hash_vtu(directory: Path) -> tuple[float, int, int]:
    """Explicitly re-read all VTU bytes; no unsafe mtime/size cache."""
    start = time.perf_counter()
    files = sorted(directory.glob('*.vtu'))
    total = 0
    for file in files:
        digest = hashlib.sha256()
        with file.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                total += len(chunk)
                digest.update(chunk)
        digest.hexdigest()
    return time.perf_counter() - start, len(files), total


def run_case(method: str, cells: int, *, steps: int, repeats: int,
             directory: Path, profile_steps: int = 5, skip_vtk: bool = False) -> dict:
    """Warm solver, profile representative steps, time checkpoint and VTU I/O."""
    if method not in METHODS or cells < 2 or steps < 1 or repeats < 1 or profile_steps < 0:
        raise ValueError('Invalid method/cells/steps/repeats/profile_steps')
    directory.mkdir(parents=True, exist_ok=False)
    solver = make_case(method, cells)
    cold, _ = _elapsed(solver.step)
    warm = []
    for _ in range(repeats):
        dt, _ = _elapsed(lambda: solver.run(steps))
        warm.append(dt / steps)
    # CProfile counts and aggregate time of Python call paths. It is not the
    # basis for wall-clock speedup claims; profiling perturbs runtime.
    profiler = cProfile.Profile()
    if profile_steps:
        profiler.enable()
        solver.run(profile_steps)
        profiler.disable()
    stats_file = directory / 'solver_profile.txt'
    stats_buffer = io.StringIO()
    if profile_steps:
        pstats.Stats(profiler, stream=stats_buffer).sort_stats('cumulative').print_stats(32)
    else:
        stats_buffer.write('Python function profiling disabled (--profile-steps 0)\n')
    stats_file.write_text(stats_buffer.getvalue(), encoding='utf-8')

    cdir = directory / 'checkpoints'
    checkpoint_write, record = _elapsed(lambda: write_checkpoint(solver, cdir))
    checkpoint_verify, _ = _elapsed(lambda: inspect_checkpoint(record.manifest_path))
    checkpoint_load, restored = _elapsed(lambda: load_checkpoint(record.manifest_path))
    np.testing.assert_array_equal(restored.particles.position, solver.particles.position)
    np.testing.assert_array_equal(restored.particles.stress, solver.particles.stress)
    assert restored.step_count == solver.step_count

    vtk_write, vtk_hash, vtk_count, vtk_bytes = None, None, 0, 0
    checkpoint_with_vtu = None
    if not skip_vtk:
        # Fresh solver so OutputManager can observe the entire time sequence.
        viz_solver = make_case(method, cells)
        viz = OutputManager(directory / 'visualization', every_n_steps=1,
                            particles=True, grid=True,
                            cpdi_domains=(method == 'cpdi'))
        vtutimes = []
        t, _ = _elapsed(lambda: viz.write_initial(viz_solver))
        vtutimes.append(t)
        for _ in range(2):
            diagnostics = viz_solver.step()
            t, _ = _elapsed(lambda d=diagnostics: viz.on_step_completed(viz_solver, d))
            vtutimes.append(t)
        vtk_write = statistics.median(vtutimes)
        vtk_hash, vtk_count, vtk_bytes = _hash_vtu(viz.directory)
        checkpoint_with_vtu, combined_record = _elapsed(
            lambda: write_checkpoint(viz_solver, cdir, prefix='with_vtu', output_manager=viz))
        # Check that the manifest including the Stage-D2 VTU hashes remains valid.
        inspect_checkpoint(combined_record.manifest_path)
        # Isolate the particle VTK writer cost (no collection-index rewrites).
        single_times = []
        for i in range(repeats):
            vtkpath = directory / f'single_particle_{i}.vtu'
            t, _ = _elapsed(lambda path=vtkpath: write_particles_vtu(
                solver.particles, path, time=solver.time, step=solver.step_count))
            single_times.append(t)
            vtkpath.unlink()
        standalone_particle_write = statistics.median(single_times)
    else:
        standalone_particle_write = None

    return {
        'method': method, 'cells_per_axis': cells,
        'particles': int(solver.particles.n_particles),
        'warm_step_median_sec': statistics.median(warm),
        'cold_step_sec': cold,
        'checkpoint_write_sec': checkpoint_write,
        'checkpoint_verify_sec': checkpoint_verify,
        'checkpoint_load_sec': checkpoint_load,
        'vtk_frame_median_sec': vtk_write,
        'vtk_particle_writer_median_sec': standalone_particle_write,
        'vtu_hash_all_sec': vtk_hash,
        'vtu_hashed_files': vtk_count,
        'vtu_hashed_bytes': vtk_bytes,
        'checkpoint_with_vtu_sec': checkpoint_with_vtu,
        'profile_file': str(stats_file.relative_to(directory.parent.parent)),
    }


def run_profile(*, methods=METHODS, sizes=(8, 16), steps=8, repeats=3,
                profile_steps=5, output_dir=Path('results/stage10g_d4'), skip_vtk=False):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Will not overwrite existing results: {output_dir}')
    if not methods or not sizes or steps < 1 or repeats < 1 or profile_steps < 0:
        raise ValueError('Invalid profile configuration')
    output_dir.mkdir(parents=True)
    data = []
    for method in methods:
        for cells in sizes:
            print(f'Profiling {method} / {4*cells*cells} particles', flush=True)
            data.append(run_case(method, cells, steps=steps, repeats=repeats,
                                 profile_steps=profile_steps,
                                 directory=output_dir / method / f'cells_{cells}',
                                 skip_vtk=skip_vtk))
    meta = {
        'stage': '10G-D4',
        'platform': platform.platform(), 'python': sys.version,
        'numpy': np.__version__, 'numba': numba.__version__,
        'cpu_count': os.cpu_count(),
        'settings': {'methods': list(methods), 'sizes': list(sizes),
                     'steps': steps, 'repeats': repeats,
                     'profile_steps': profile_steps, 'skip_vtk': skip_vtk},
        'notes': [
            'All warm timings exclude the initial Numba JIT/cache step for each case.',
            'Python cProfile results are stored separately and perturb timings.',
            'Default VTK fields, ASCII Float64, and output-index updates are included.',
            'Checkpoint-with-VTU timing includes full D2 SHA-256 rehashing of every snapshot.',
            'No hardware-independent speedup or accuracy threshold is claimed.',
            'Results are from an elastic, nearly-static synthetic case, not a slope failure.',
        ],
        'measurements': data,
    }
    (output_dir / 'profile_summary.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    with (output_dir / 'measurements.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('results/stage10g_d4'))
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    parser.add_argument('--sizes', type=int, nargs='+', default=(8, 16))
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--profile-steps', type=int, default=5)
    parser.add_argument('--skip-vtk', action='store_true')
    args = parser.parse_args()
    if not args.sizes or any(size < 2 for size in args.sizes):
        parser.error('All sizes must be >= 2')
    if args.steps < 1 or args.repeats < 1 or args.profile_steps < 0:
        parser.error('Invalid steps/repeats/profile-steps')
    results = run_profile(methods=args.methods, sizes=args.sizes, steps=args.steps,
                          repeats=args.repeats, profile_steps=args.profile_steps,
                          output_dir=args.output_dir, skip_vtk=args.skip_vtk)
    print('Cases:', len(results['measurements']))
    print('Results:', args.output_dir / 'measurements.csv')


if __name__ == '__main__':
    main()
