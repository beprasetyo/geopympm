# GeoPyMPM Stage 10G-A3 — Time Series and OutputManager

**Baseline:** GeoPyMPM Stage 10G-A2; **scope:** I/O only. Core solver, interpolators, constitutive model, Numba kernels and numerical tolerances unchanged.

## Objective

Create time-aware multi-frame visualization for both **VisIt** and **ParaView**, and schedule particle/grid/CPDI output based on committed explicit MPM time steps.

## New Python modules

- `src/geopympm/io/time_series.py` — validates monotonic (step, physical time, file) records and writes `.visit` (`!NBLOCKS 1` and `!TIME`) and `.pvd` (VTK Collection) indexes; both use files in the same visualization folder.
- `src/geopympm/io/output_manager.py` — `OutputManager`; triggers writers after `solver.step()` has successfully committed; handles scheduling, stream selection, optional initial frame, rollback of newly created snapshots and restoration of index files if a writer fails.
- `examples/19_output_time_series.py` — 2D force-driven CPDI example demonstrating output under timestep and physical-time schedules.
- `validation/stage10g_a3/validate_time_series.py` — deterministic standalone numerical and format audit.
- `tests/test_stage10g_a3_time_series.py` — unit/regression tests.

## Usage

```python
from geopympm.io import OutputManager

output = OutputManager(
    'results/myrun/visualization',
    every_n_steps=100, every_dt=None,
    particles=True, grid=True,
    cpdi_domains=(solver.interpolation == 'cpdi'),
)
output.write_initial(solver)  # optional step 0 / time 0
for _ in range(1000):
    diagnostic = solver.step()  # may raise; no file written in that case
    written = output.on_step_completed(solver, diagnostic)
```

The `on_step_completed` method **must be called once for every successfully committed step** (including steps when no VTK file is due). This explicit callback design keeps I/O outside the mechanics kernel and avoids changing pre-existing `solver.run(n_steps)` semantics. To use it with `run`, call `step()` in your own loop as shown.

## Scheduling semantics

- `every_n_steps=100`: output at steps 100, 200, ... (plus optional step 0 via `write_initial`).
- `every_dt=0.2`: output after the first *actual* timestep that crosses t = 0.2, 0.4, 0.6, ... s. **No interpolation to those times** is performed. File metadata contains the actual committed simulation time.
- Both schedules present: logical OR; simultaneous triggers produce one snapshot, not duplicates.
- Either schedule may be `None`, but not both.
- `write_initial()` allowed only once at `(step,time)=(0,0)` before any other output-manager step.
- Mismatched time/step diagnostics, duplicate callback, skipped callback and CPDI domain export on non-CPDI solver are rejected.

## Produced files

```text
results/myrun/visualization/
    particles_000000.vtu
    particles_000100.vtu
    particles.visit
    particles.pvd
    grid_000000.vtu
    grid_000100.vtu
    grid.visit
    grid.pvd
    domains_000000.vtu        # only with cpdi_domains=True
    domains_000100.vtu
    domains.visit
    domains.pvd
```

Open `particles.visit`, `grid.visit`, or `domains.visit` in **VisIt** as a time-varying database. Open the corresponding `.pvd` in **ParaView**. One file per stream per frame; the three time series use the same recorded output steps. `!TIME` in `.visit` and `DataSet timestep` in `.pvd` reference physical time in **seconds**. `.pvd` uses standard VTK XML Collection syntax. The underlying A1/A2 `.vtu` writers retain their tensor/Voigt and Quad4/CPDI geometry conventions.

## Failure and data safety

- Writer is called only after `solver.step()` returns successfully; failed timesteps produce no snapshots.
- Existing snapshots and previous time series are **never silently overwritten**. Stage 10G-A3 disallows continuing into an existing run directory; real checkpoint/restart support belongs to Stage 10G-B.
- Each index is written via a temporary file then atomically replaced. If one writer errors, new .vtu snapshots are deleted and the indexes are rolled back where possible. This is not a single multi-file filesystem transaction against sudden power failure; orphan partial frames/indexes should be detected by future restart tooling.
- Snapshot output never modifies solver arrays or update order. Initial nodal transient grid fields may be zero; post-step grid nodal fields reflect the just-completed step's assembly (not an interpolated grid at the new particle positions).
- Output is currently ASCII VTK XML, so particle fields and full background grid may produce large data. Output interval and field selection should control storage.

## Validation

```bash
python examples/19_output_time_series.py
python -m validation.stage10g_a3.validate_time_series
python -m pytest tests/test_stage10g_a3_time_series.py -q -rs
python -m pytest -q
```

The standalone validation checks time series indexes and independent VTK readers (if `vtk` is installed), and runs two **identical** MPM simulations, one with the manager and one without; key particle arrays must match exactly. It does not launch VisIt GUI or ParaView GUI.

## Known limitations and Stage 10G-B dependency

1. No checkpoint/restart, resume, or append into an existing time series; writer state (scheduled time targets, step counters and file indexes) is in RAM.
2. No native Uintah `.uda` archive or persistent solver state.
3. `.visit`/`.pvd` indexes are individually atomic, but multi-file updates are not crash-transactional.
4. Field names are those implemented in A1/A2; elastoplastic internal state will require extending the field registry later.
5. ASCII writer scales poorly to high particle counts; test performance and optionally implement appended-binary VTK in a separate optimization stage.
6. A time-based schedule may skip several target times if a single solver timestep crosses more than one boundary; only **one** snapshot of the actual committed state is emitted.
7. Stage 10G-B checkpoint restore must explicitly reconstruct `OutputManager` scheduling and output-index state rather than pretending this session-only scheduler can resume itself.

## References

- VisIt User Manual, "Creating `.visit` Files": https://visit-sphinx-github-user-manual.readthedocs.io/en/v3.5.0/using_visit/WorkingWithFiles/Supported_File_Types.html
- ParaView discussion on `.pvd` VTK collection: https://discourse.paraview.org/t/time-support-in-vtk-paraview/6878/4
