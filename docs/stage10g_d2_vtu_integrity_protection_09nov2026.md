# GeoPyMPM — Stage 10G-D2: VTU Integrity Protection

**Date:** 9 November 2026  
**Input baseline:** Stage 10G-D1 audit patch applied to Stage 10G-C  
**Scope:** Checkpoint writer / output restart validation only; no solver, material, interpolation, kinematics, or VTU writer kernel changes  
**Status:** Implemented and verified with Stage 10G regression tests; full repository validation is a separate gate

## Motivation and design

Stage 10G-C verified that checkpoint NPZ bytes, output snapshot *filenames* and `.visit`/`.pvd` index bytes were consistent. It could not detect an edited `.vtu` whose filename was unchanged. The Stage 10G-D1 audit reproduced this gap.

Stage 10G-D2 extends the **existing checkpoint schema 1** with a *separately versioned* metadata section in `output_manager`, and verifies it before reconstructing the resumed `OutputManager`. This is a backwards-compatible **additive checkpoint format extension**: archived numerical arrays and `schema_version = 1` remain unchanged. No changes to the explicit MPM solver.

### New checkpoint metadata

New checkpoints written **with** `output_manager=manager` include:

```json
"output_manager": {
  "...": "existing scheduler fields unchanged",
  "vtu_integrity": {
    "version": 1,
    "algorithm": "sha256",
    "files": {
      "particles": {"particles_000000.vtu": "<64-digit lowercase SHA-256>"},
      "grid": {"grid_000000.vtu": "<64-digit lowercase SHA-256>"}
    }
  }
}
```

`files` includes every enabled output stream (`particles`, `grid`, or `domains`), and exactly one digest for **each frame** already emitted in that stream. If no frames have yet been emitted, the enabled stream has an empty mapping. `output_manager=None` keeps the existing checkpoint format with `output_manager: null`.

The hash records are keyed by *relative basenames* and are independent of the visualization directory's absolute path. Consequently `restore_output_manager(..., directory=copy)` works if the copied VTU contents and index files are identical.

### Checkpoint writing

Before publishing a new checkpoint, the writer checks all registered VTU files exist and are not symlinks, there are no unregistered VTU filenames for selected streams, and the `.visit`/`.pvd` indexes agree with the scheduler state. It hashes the original VTU bytes in streaming chunks and attaches hashes to the checkpoint manifest. If validation fails, no new checkpoint pair is published.

**Cost consideration:** every checkpoint rehashes the complete visualization series recorded so far, *not merely the latest frame*. This is a conservative correctness-first decision for D2. Large simulations with frequent VTU exports may see additional checkpoint latency proportional to accumulated VTU bytes. Stage 10G-D4 should measure the overhead and evaluate safe caching or incremental integrity records before changing it.

### Restoring / fail-closed policy

```python
from geopympm.io import load_checkpoint, restore_output_manager

solver = load_checkpoint("results/run/checkpoints/state_00000009.json")
manager = restore_output_manager(
    "results/run/checkpoints/state_00000009.json", solver
)
```

For a new D2 checkpoint, restore requires: an exact match of metadata version and algorithm, all streams and frame keys present with valid hex SHA-256 strings, all expected files/indexes present with no extra enabled-stream frames, and the **SHA-256 of every VTU matching its saved digest**. Missing/mismatched hashes or changed files cause `ValueError` before creating an OutputManager. The operation never repairs, rewrites or deletes output.

For a checkpoint from Stage 10G-B2/C without `vtu_integrity` but with **at least one VTU frame**, the default is a safe rejection. To continue a legacy visualization series after checking that the files are trusted:

```python
manager = restore_output_manager(
    old_manifest, solver,
    allow_unverified_legacy_vtu=True,
)
```

This option **only bypasses the missing-hash gate**. It does not disable checks on indexes/filenames and cannot disable SHA-256 verification when a D2 integrity section is present. A legacy checkpoint with *zero emitted frames* requires no override because no VTU exists to verify. `load_checkpoint(old_manifest)` continues to work for solver-only numerical restart without this option.

**Important:** `allow_unverified_legacy_vtu=True` is an explicit acceptance of **unverified** historical VTU contents. To establish protected output for a legacy run, create a newly hashed checkpoint after restoring a trusted legacy series.

## Reliability boundary and limitations

- SHA-256 detects accidental or unauthorized changes to a VTU **relative to the checkpoint's saved digest**. It does **not** authenticate the checkpoint JSON. Someone who can maliciously edit both the manifest and VTU can update the digest. Use trusted checkpoint storage, permissions, or externally signed manifests for adversarial integrity guarantees.
- Hashing and inspection cannot make the VTU files and checkpoint manifest one atomic disk transaction. Concurrent writes, race conditions and power-loss durability need their own controls; this project assumes a **single writer**.
- The writer can hash a VTU that was already corrupted **before** it created the checkpoint. Correctness of the original visualization remains the responsibility of the VTU writers and physics validation.
- Older checkpoints lack VTU hashes: this cannot be repaired retroactively without a trusted independent copy. The explicit override is not a guarantee.
- No automatic fallback or retention policy is added here. These are Stage 10G-D3 items.
- No attempt is made to validate VTK XML semantic correctness beyond byte-integrity and existing index semantics.

## Verification

From repository root with editable package installed, or `PYTHONPATH=src`:

```bash
python -m pytest -q tests/test_stage10g_d2_vtu_integrity.py tests/test_stage10g_d_audit.py
python -m pytest -q \
  tests/test_stage10g_a1_vtk_writer.py \
  tests/test_stage10g_a2_mesh_vtk.py \
  tests/test_stage10g_a3_time_series.py \
  tests/test_stage10g_b1_checkpoint.py \
  tests/test_stage10g_b2_restart.py \
  tests/test_stage10g_c_extended_restart.py
```

The D2 tests cover all four interpolation methods, particle/grid/domain hash sets, tampered earliest/later snapshots, copied visualization directories, missing/extra hashes, unsupported metadata version/algorithm, optional legacy handling, no-frame legacy handling, and refusing incomplete output when writing checkpoints. The previous strict `xfail` in `tests/test_stage10g_d_audit.py` is removed because tampered VTU bytes are now correctly rejected.

### Verified results in the patch preparation environment

- **142 passed**, 0 failed: combined Stage 10G A1/A2/A3/B1/B2/C, D1 and new D2 tests.
- **Four independent 120-step restart chains passed**, each with checkpoints at steps 30, 60, 90: MPM, uGIMP, cpGIMP and CPDI1. Total snapshot byte comparisons: 60, 60, 60 and 90, respectively.
- A wider regression-suite attempt (excluding Stage 6E and 6F) reached approximately **80% without a reported failure** before the environment execution limit interrupted it. This is **not** a full-suite pass. Stage 6E/6F and remaining tests should be rerun in the development environment before considering the project-wide audit closed.

## Patch inventory (relative to GeoPyMPM repository root)

```text
src/geopympm/io/checkpoint.py              MODIFIED (write hashes)
src/geopympm/io/restart.py                 MODIFIED (verify hashes + legacy option)
tests/test_stage10g_d_audit.py             MODIFIED (xfail -> passing regression)
tests/test_stage10g_d2_vtu_integrity.py    NEW (D2 tests)
docs/stage10g_d2_vtu_integrity_protection_09nov2026.md  NEW
```

### Integration order

Apply patches **10G-D1 first, then 10G-D2**. Both ZIPs contain paths *relative to repository root* and no enclosing `geopympm-main/` directory. Do not run D2 as a standalone Python module copied outside the repository.

Next planned: **Stage 10G-D3 — Safe checkpoint discovery/recovery and conservative retention policy.**
