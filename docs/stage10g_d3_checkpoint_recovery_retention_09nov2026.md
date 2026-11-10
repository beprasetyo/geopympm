# GeoPyMPM — Stage 10G-D3: Safe Checkpoint Recovery & Retention

**Date:** 9 November 2026  
**Baseline:** Stage 10G-C + incremental Stage 10G-D1 and D2 patches  
**Patch type:** incremental (only new/modified paths; no full repository)  
**Scope:** high-level checkpoint discovery/recovery and conservative retention. **No solver, interpolation, materials, GIMP/CPDI, VTK writer, checkpoint schema or output-manager kernel changes.**

## 1. Reliability objective

Stage D2 verifies VTU contents, but the user had to identify a valid checkpoint manifest manually. Stage D3 adds an opt-in API to:

1. Inspect committed checkpoint pairs in a directory, newest step first.
2. Recover the **newest fully reconstructable numerical state**; if the newest checkpoint is corrupt or incomplete, try earlier committed pairs without modifying them.
3. Optionally require exact D2 OutputManager/VTU compatibility as an additional condition for recovery.
4. Preview a disk-retention policy, and optionally remove older **verified** JSON/NPZ pairs while keeping the newest N verified numerical checkpoints.

No automatic deletion or modification takes place during scanning/recovery. Retention is **dry-run by default** and never touches VTU/.visit/.pvd files.

## 2. Public API

The functions are exported from `geopympm.io`:

```python
scan_checkpoints(directory, *, prefix="state") -> CheckpointInventory
recover_latest_checkpoint(directory, *, prefix="state", restore_output=False,
                          output_directory=None,
                          allow_unverified_legacy_vtu=False) -> RecoveryResult
prune_checkpoints(directory, *, prefix="state", keep_last=3,
                  dry_run=True) -> RetentionReport
```

Supporting types: `CheckpointIssue`, `CheckpointInventory`, `RecoveryResult`, `RetentionReport`, `CheckpointRecoveryError`.

The checkpoint path format is unchanged: `state_00000030.json` (commit marker) paired with `state_00000030.npz` (numeric data). An `npz` file without its manifest is an **orphan**, never a recovery candidate. File paths must not be symlinks; noncanonical prefixes are rejected. File name and saved solver step must agree. Recovery prioritizes the **numeric step**, not the manifest modification timestamp.

### Inspect status without modifying anything

```python
from geopympm.io import scan_checkpoints

inventory = scan_checkpoints("results/run/checkpoints")
for path in inventory.valid:
    print("VALID", path.name)
for issue in inventory.invalid:
    print("INVALID", issue.manifest_path.name, issue.reason)
for path in inventory.orphan_data:
    print("ORPHAN", path.name)
```

A valid pair is validated through `load_checkpoint`, including NPZ SHA-256 and complete array/schema and solver reconstruction checks. This means scanning a large set of large checkpoints can be I/O-intensive; Stage 10G-D4 will profile this overhead. A successful scan is **numerical validation**, not validation of the physical model or existing visualizations.

### Recover newest numerical checkpoint with automatic fallback

```python
from geopympm.io import recover_latest_checkpoint

result = recover_latest_checkpoint("results/run/checkpoints")
solver = result.solver
print("Recovered step:", result.step)
print("Checkpoint:", result.manifest_path)
for issue in result.skipped:
    print("Skipped:", issue.manifest_path.name, issue.reason)
# Resume numerical calculation: diagnostic = solver.step()
```

On corruption (`.json` invalid, NPZ checksum mismatch, missing archive, incompatible schema, failed reconstruction), D3 tries the next oldest canonical committed manifest. If nothing is eligible, it raises `CheckpointRecoveryError`. No new checkpoint files are written, and no files are repaired or removed.

### Recover the existing visualization series (strict)

```python
result = recover_latest_checkpoint(
    "results/run/checkpoints",
    restore_output=True,
    # output_directory="a/verified/copy/of/checkpoint-era/viz",  # optional
)
solver, output = result.solver, result.output_manager
# diagnostic = solver.step()
# output.on_step_completed(solver, diagnostic)
```

`restore_output=True` invokes the existing Stage D2 `restore_output_manager`: all expected `.vtu` and index files must **exactly match** that checkpoint. A corrupt latest numeric checkpoint may have older valid numeric checkpoints while the visualization directory contains extra later frames. In this common scenario, **numerical fallback succeeds but visualization fallback is refused**. This is by design: D3 does NOT automatically truncate/overwrite newer VTU data. A valid verified checkpoint-era visualization copy may be supplied via `output_directory`. If such a copy is not available, use numerical-only recovery and plan a separate new output series; do not force old output to overwrite later frames.

For pre-D2 checkpoints without recorded VTU hashes, D2's explicit `allow_unverified_legacy_vtu=True` opt-in remains available. It never disables verification for D2-hashed checkpoints.

### Retention: preview first, then explicitly opt in

```python
from geopympm.io import prune_checkpoints

preview = prune_checkpoints("results/run/checkpoints", keep_last=3)
print("Kept:", [p.name for p in preview.retained])
print("Would delete:", [p.name for p in preview.would_delete])
print("Corrupt (not touched):", preview.invalid)
print("Orphan data (not touched):", preview.orphan_data)

# Optional DESTRUCTIVE action; do only after reviewing preview and making
# backups, and with NO concurrent writer using the checkpoint directory:
report = prune_checkpoints(
    "results/run/checkpoints", keep_last=3, dry_run=False
)
print("Deleted verified JSON/NPZ pairs:", len(report.deleted))
```

**Safety policy:**

- `keep_last` is a positive integer (default **3**) and counts only **validated numerical checkpoint pairs**; a broken newer file does not displace a good older fallback.
- If there are fewer than N valid pairs, nothing is deleted. Invalid JSON/NPZ pairs and orphan NPZ files remain for human inspection. A nonexistent directory raises rather than being silently created.
- Only older **valid committed pairs** are eligible for deletion. For each deletion, revalidate the candidate, then remove its manifest **first** (commit marker) and its `.npz` second. A crash between unlinks may leave an orphan NPZ, which is not a committed checkpoint; D3 does not delete orphans automatically.
- Retention NEVER removes visualization files, time-series indexes, or checkpoint files of other prefixes.
- Keep_last protects numeric recovery points, **not independent copies of the visualization series** or guaranteed disaster recovery. Use separate backup/snapshot storage for that.
- Run only with a **single writer**, and do not prune while the solver is generating checkpoints or another tool is modifying them. A two-file pair and output directory are not an atomic transaction across concurrent processes or power failures.
- Do not use retention on your only copy of critical research results without an external backup. Deleted checkpoint pairs are not recoverable via D3.

## 3. Reproducible verification

```bash
python -m pytest -q tests/test_stage10g_d3_recovery_retention.py
python -m pytest -q \
  tests/test_stage10g_a1_vtk_writer.py \
  tests/test_stage10g_a2_mesh_vtk.py \
  tests/test_stage10g_a3_time_series.py \
  tests/test_stage10g_b1_checkpoint.py \
  tests/test_stage10g_b2_restart.py \
  tests/test_stage10g_c_extended_restart.py \
  tests/test_stage10g_d_audit.py \
  tests/test_stage10g_d2_vtu_integrity.py \
  tests/test_stage10g_d3_recovery_retention.py
```

Tests cover all four interpolation methods (MPM, uGIMP, cpGIMP, CPDI1); corrupt, missing or malformed newest pairs; unsupported metadata; orphan NPZ; filename identity; read-only recovery; prefix isolation; symlink rejection; D2 VTU tamper detection; correct/incorrect visualization fallback; numerical continuation equality after fallback; dry-run and explicit pair retention; and retention when newest pair is corrupt.

**Verification gate:** Tests passing on the patch-preparation environment do not replace running the suite on the developer's Ubuntu `.venv`. Broader project-wide tests (especially numerically expensive earlier stages) remain a separate verification gate.

## 4. Files included in the incremental ZIP

```text
src/geopympm/io/recovery.py                         NEW
src/geopympm/io/__init__.py                         MODIFIED (exports)
tests/test_stage10g_d3_recovery_retention.py        NEW
docs/stage10g_d3_checkpoint_recovery_retention_09nov2026.md   NEW
```

**Apply only after D1 and D2**, from the GeoPyMPM repository root:

```bash
unzip -o ~/Downloads/GeoPyMPM_Stage10G_D3_Checkpoint_Recovery_Retention_patch_09Nov2026.zip -d .
source .venv/bin/activate
python -m pytest -q tests/test_stage10g_d3_recovery_retention.py
```

## 5. Remaining limitations / future work

- This is a conservative single-process policy, not a background task, distributed coordinator, automatic VTK rollback, or filesystem crash-consistency protocol.
- `scan_checkpoints` loads each committed pair in full, which may be expensive with large particle arrays; quantify/propose improvements in Stage 10G-D4.
- No retention by elapsed time, file size, run ID, named milestones or backup promotion is implemented; oldest valid pairs are selected by **solver step** only.
- Data SHA-256 and VTU SHA-256 protect against accidental modification relative to the manifest; they are not signed against deliberate manifest manipulation.
- Schema remains **1**, and the supported exact numerical checkpoint model remains **2D LinearElastic**. Stage 11 will require a deliberate serialization schema/material-state strategy before introducing elastoplastic checkpoint support.

**Next planned stage:** 10G-D4 — performance profiling and controlled optimization, including P2G/G2P, GIMP/CPDI kernel hotspots, checkpoint hashing, and VTK export overhead.
