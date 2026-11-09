# GeoPyMPM Stage 10G-B1 — Full-State Checkpoint Writer

**Baseline:** Stage 10G-A3 (Time Series / OutputManager), 9 November 2026.
**Scope:** archive + metadata + integrity checks, **not** restoration/restart.
**Compatibility:** the current explicit 2D MPM solver, LinearElastic materials, and the four interpolation choices `mpm`, `gimp`, `cpgimp`, `cpdi`.

## 1. Rationale and separation of concerns

Stage 10G-A3 produces `.vtu`, `.visit`, and `.pvd` files for visualization. These files contain selected fields, are not guaranteed to capture transient grid state, and **are not restart files**. This stage adds a distinct checkpoint format. It follows the principle of Uintah's versioned timestep archives, without implementing UDA or Uintah's parallel DataWarehouse.

## 2. API and files

```python
from geopympm.io import write_checkpoint, inspect_checkpoint

# Only after successful solver.step(), or at the initial step 0:
record = write_checkpoint(solver, "results/example/checkpoints")
metadata = inspect_checkpoint(record.manifest_path)
```

The files for step 25 are:

```
checkpoints/
├── state_00000025.npz  # compressed numeric NumPy arrays, no pickle
└── state_00000025.json # schema/version, configs, types, SHA-256
```

`write_checkpoint` returns a `CheckpointRecord` containing the two paths, step, simulation time, particle count, array count, and combined file size. It refuses to overwrite an existing snapshot. A safe optional ASCII `prefix` changes the stem. `inspect_checkpoint(path, verify=True)` verifies the SHA-256, the array names, dtypes, shapes, and finite values. It **does not reconstruct** a solver.

## 3. State coverage

### Particle arrays (12)

`position`, `position0`, `mass`, `volume`, `volume0`, `velocity`, `material_id`, `stress`, `strain`, `domain_size`, `domain_size0`, `deformation_gradient`.

* Stress and strain Voigt order: `[xx, yy, zz, xy, yz, xz]`.
* Strain shear terms use engineering shear; stress positive in tension.
* CPDI1 corners are reconstructed using `position`, `deformation_gradient`, `domain_size0`, not archived as a separate redundant field.
* Future plastic internal state is **not yet supported**. The writer fails on unexpected fields so as not to silently discard them.

### BackgroundGrid arrays (11)

`origin`, `cell_size`, `node_coordinates`, `mass`, `momentum`, `velocity`, `acceleration`, `force_internal`, `force_external`, `velocity_bc_mask`, `velocity_bc_value`. `n_cells`, `node_shape` and `n_nodes` are stored in JSON.

Although grid mass/momentum/forces are reassembled at each step, we store them in B1 so a checkpoint is a faithful complete snapshot of the current state. A future restart reader may safely choose to reset/reassemble transient fields according to the proven integration sequence.

### Solver arrays and JSON settings

`solver.gravity` is always archived. `solver.nodal_external_force` is archived when non-None. JSON includes `time`, `step`, `dt`, `cfl`, `pic_fraction`, interpolation, stress/kinematics/advection update selections and a flag indicating whether nodal external forces exist.

### Materials

`LinearElastic(E,nu)` is serialized in JSON as a single model or as an integer-ID mapping of models. Unrecognized material types are rejected. The format must evolve when Von Mises / Mohr–Coulomb internal variables are added in Stage 11.

### Optional OutputManager state

Pass `output_manager=output` **after** `output.on_step_completed(...)`. Its schedule, enabled streams, field selection, next time target, known frame entries, and last seen step/time are stored in JSON. This is an optional *record of scheduler state*, not automatic restoration; Stage 10G-B2 will reconstruct and reconcile existing `.visit` and `.pvd` series.

## 4. File publication and integrity

1. Validate registered solver, grid, particle, material types, shapes, and finite array values **before writing**.
2. Copy numeric arrays (`numpy.ndarray`) without changing the live solver.
3. Write a temporary compressed NPZ in the target folder; flush and `fsync`.
4. Compute SHA-256 for the resulting file; record data array names, dtypes, shapes, and config in a versioned JSON manifest.
5. Write and `fsync` a temporary JSON manifest.
6. Rename the NPZ to its final path; **rename the JSON last as the commit marker**. Delete NPZ and temporary files if an ordinary exception occurs.

This procedure avoids exposing a *completed manifest* before NPZ publication in ordinary single-writer operation. It is not a cross-file transaction or a promise of power-failure atomicity on every filesystem. A hard crash may leave an orphan NPZ without JSON. Concurrent writers, network filesystems, and distributed checkpointing require further design and stress testing.

The writer uses `np.savez_compressed` and the inspector loads with `allow_pickle=False`. It does not evaluate or deserialize arbitrary Python objects.

## 5. Test and validation commands

```bash
python examples/20_checkpoint_writer.py
python -m validation.stage10g_b1.validate_checkpoint_writer
python -m pytest tests/test_stage10g_b1_checkpoint.py -q -rs
python -m pytest -q
```

Example 20 runs six CPDI solver steps; after steps 3 and 6 it writes and verifies paired checkpoint files, including OutputManager scheduling metadata. The example also emits Stage 10G-A3 visualization series independently.

The separate validation script checks four interpolation modes, with eight checks per mode: exact serialized particle/grid arrays, consistent time and step, material and solver config, unchanged live state, safe numeric arrays, and integrity verification.

## 6. Explicit limitations and Stage 10G-B2 requirements

**B1 writes and inspects; it does not restart.** B2 must reconstruct ParticleSet, BackgroundGrid and materials using validated checkpoint data, restore preexisting arrays in-place, correctly handle cpGIMP/CPDI1 domain consistency and initial state, restore or safely reinitialize transient nodal state, and resume timestep/time without reapplying initializations. It must also restore OutputManager frame indexing if one was saved, with conflict checks, and prove uninterrupted-vs-restart equivalence for each interpolation.

B1 does **not** implement automatic scheduling or retention of checkpoints; these belong to later B2/C or a dedicated CheckpointManager. VTK output still uses OutputManager from A3; checkpoint writing is currently an explicit call. All file paths in example results should be stored on persistent disk for real long runs.
