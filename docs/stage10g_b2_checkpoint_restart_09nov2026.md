# GeoPyMPM Stage 10G-B2: Checkpoint Restart Loader

Date: 9 November 2026  
Baseline: Stage 10G-B1, checkpoint schema 1 (2D, LinearElastic only).  
Status: Implemented, validated with 29 new pytest tests and 32 independent method-specific checks.

## Scope, boundaries and motivation

Stage 10G-B1 already wrote an `.npz` archive of all 12 particle arrays, 11 grid arrays and 1–2 solver arrays; its `.json` manifest stores material and solver configuration, clock, simulation step, optional `OutputManager` state and SHA-256 of the NPZ. B2 is the inverse operation: construct a runnable, independent solver from the disk checkpoint and, optionally, reconnect the existing visualization scheduler.

This is not a native Uintah `.uda` reader, and it does not introduce HPC parallel restart, checkpoint retention/rotation, plastic variables, arbitrary materials, or cross-platform reproducibility guarantees. It does not alter the numerical kernels.

## New implementation

New module `src/geopympm/io/restart.py` exports:

```python
from geopympm.io import load_checkpoint, restore_output_manager

solver = load_checkpoint("results/run/checkpoints/state_00000006.json")
output = restore_output_manager(
    "results/run/checkpoints/state_00000006.json", solver
)
for _ in range(1000):
    diagnostics = solver.step()
    output.on_step_completed(solver, diagnostics)
```

`load_checkpoint(manifest_path)`:

1. Calls `inspect_checkpoint(..., verify=True)` to enforce the B1 schema and NPZ SHA-256, array field inventory, dtype and shape signatures, and finiteness.
2. Checks manifest clock, particle count, 2D structured-grid dimensions, material specifications, solver configuration and strict `float64`/`int32`/`bool` internal types. No pickle is used.
3. Reconstructs `ParticleSet` from saved initial coordinates, masses and volumes and then restores all 12 saved particle arrays, including stress, strain, F, current coordinates and cpGIMP/CPDI domain fields.
4. Reconstructs `BackgroundGrid` and checks that the node coordinates match the structured-grid geometry exactly; restores all 11 grid arrays including transient nodal fields and prescribed velocity BCs.
5. Constructs `LinearElastic` or the saved material-ID mapping and then `ExplicitMPMSolver` with exact algorithm settings, gravity and optional nodal force.
6. Restores committed `time` and `step_count`, reserializes the freshly built solver in-memory and checks it matches the archived numerical arrays and configuration exactly.
7. Returns an independent, runnable solver; the original checkpoint files are left untouched.

`restore_output_manager(manifest_path, solver, directory=None)`:

- Requires the checkpoint to contain `output_manager` metadata and the solver to still be at the saved step and time.
- Restores selected streams, fields, schedule, next target physical time, initial-written flag, last observed step/time, and all `SeriesFrame` entries.
- Requires existing `.vtu` snapshots and `.visit`/`.pvd` index bytes to match the recorded lists **exactly**. It explicitly refuses an output folder containing future snapshots or modified indexes. It may also restore a scheduler that has never emitted a frame.
- Does not modify any files during restoration and does not overwrite existing frames. When a continuation step triggers output, the existing A3 `OutputManager` writes the next snapshot and updates indexes normally.
- `directory` may point to a **copied checkpoint-era visualization folder** for a branch/restart in another location. A directory override does not mean start a new, empty animation series.

### Important ordering constraint

Save checkpoints **after** processing the last committed solver step through the `OutputManager`, as in:

```python
diagnostics = solver.step()
output.on_step_completed(solver, diagnostics)
record = write_checkpoint(solver, "results/run/checkpoints", output_manager=output)
```

A checkpoint without `output_manager` can still be restored as a numerical solver. Restoring an animation is optional and requires the saved scheduler metadata and its matching visualization files.

## Example 21: real process restart and continuous/split comparison

Default mode compares the same 36-particle CPDI elastic test run as a continuous 12-step simulation and a 6+6 checkpoint/restart trajectory:

```bash
python examples/21_checkpoint_restart.py --mode compare --output-dir results/stage10g_b2
```

Two independent Python process invocations demonstrate cold restart, assuming the `--output-dir` path has not been used before:

```bash
python examples/21_checkpoint_restart.py --mode save --output-dir results/b2_realprocess
python examples/21_checkpoint_restart.py --mode resume --output-dir results/b2_realprocess
```

The first command writes the checkpoint at step 6; the second reconstructs the solver and OutputManager, continues 6 more steps, and appends to the same `.visit`/`.pvd` series. The example refuses to clobber an already existing save output folder.

## Numerical validation

Command:

```bash
python -m validation.stage10g_b2.validate_restart
```

Four interpolations, each 8 checks = 32/32 passed. In each check, the solver is run to step 5, checkpointed, loaded, and compared against continuous evolution to step 10. The test verifies exact initial state restoration, end state, diagnostics, material/forces/clock, frame lists and both index formats.

New tests:

```bash
python -m pytest tests/test_stage10g_b2_restart.py -q -rs
```

**29 passed**. Includes equal 5-step continuation for all four interpolation methods, material-ID mapping, fixed nodal force, boundary conditions, CPDI domain state, zero-domain point MPM, no external nodal force, round-trip to a second checkpoint, missing/inconsistent VTK snapshots, checkpoint metadata corruption, unsupported constitutive model/schema/dtype, initial state, no-output-yet manager, copied visualization directory, and unchanged files on load.

Regression suite was run in three groups because a single-process run hit an environment execution limit around 79%, without a test assertion failure:

| Group | Result |
|---|---|
| Full suite excluding Stage 6E and Stage 6F | 581 passed |
| Stage 6E cell-crossing tests | 29 passed |
| Stage 6F energy-convergence tests | 27 passed |
| **Total** | **637 passed, 0 failed** |

Commands:

```bash
python -m pytest -q --ignore=tests/test_stage6e_cell_crossing.py --ignore=tests/test_stage6f_energy_convergence.py
python -m pytest tests/test_stage6e_cell_crossing.py -q
python -m pytest tests/test_stage6f_energy_convergence.py -q
```

## Known limitations and Stage 10G-C/D priorities

- The loader supports only the B1 **schema 1**, 2D `LinearElastic` model, and current stage arrays. New plastic state must **not** be silently ignored; add a versioned serializer for Stage 11 first.
- The SHA-256 checks the `.npz` bytes against the JSON manifest, detecting accidental damage, but the manifest itself is **not cryptographically authenticated**; both can be maliciously replaced. Treat checkpoint files as trusted simulation inputs.
- Existing VTK files are checked by presence and indexes, not by independently hashing each VTK snapshot. File content integrity and interrupted/wrong-version output must be audited in Stage 10G-D.
- A restart of an older checkpoint refuses a visualization folder containing output generated after that checkpoint; copy the matching checkpoint-era directory or start a branch with a deliberately designed truncation workflow. The loader never automatically deletes newer results.
- Restart tests show bitwise equality on this local environment and deterministic algorithm. Cross-architecture, compiler/Numba, thread-count or modified-code reproducibility is not guaranteed.
- No automatic checkpoint interval, retention pruning or fallback to last valid checkpoint has been implemented yet. These belong in Stage 10G-C/D.
- Support for MPI/distributed partitions and true Uintah `.uda` interoperability is outside scope.

## Stage integration decision

After the A1/A2/A3 visualization pipeline and B1/B2 lossless writer/loader, work on Stage 10G-C (longer restart-equivalence/corruption scenarios) and 10G-D (reliability/performance, retention/recovery), then Stage 11 constitutive framework with corresponding schema extension.
