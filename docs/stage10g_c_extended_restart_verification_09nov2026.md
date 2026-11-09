# GeoPyMPM — Stage 10G-C: Extended Restart Verification

**Date:** 9 November 2026  
**Baseline:** Stage 10G-B2 (checkpoint schema 1)  
**Purpose:** Demonstrate that repeatedly restoring saved MPM state does not change the future numerical trajectory or VisIt/ParaView series.

## 1. Scope and non-goals

This stage **adds verification and example code only**. It does **not** modify `src/geopympm/` (interpolation, solver, particles, grid, materials, or I/O). Stage 10G-C is a *numerical reproducibility audit*, **not** a physical validation of finite-strain soil behavior or a guarantee of filesystem crash durability.

The checkpoint format remains the Stage 10G-B1 **schema version 1**, and the restart loader remains Stage 10G-B2. Only **2D LinearElastic** material(s) are currently supported. There is no support yet for adding/deleting particles, plastic history, or automatic recovery from the most recent valid archive.

## 2. Verification method

Two simulations start with the same 36 particles, structured 18×18-cell background grid, multiple linear elastic material IDs, gravity, a nodal external force, and a velocity boundary condition.

- **Continuous:** advance from step 0 to N without interruption.
- **Restart chain:** at N/4, N/2, 3N/4, write a full checkpoint and reconstruct the solver and `OutputManager` with `load_checkpoint` and `restore_output_manager`.
- Both paths advance **in lockstep**, verifying *every committed step*, including the returned diagnostics.
- Every NumPy array in `ParticleSet` and `BackgroundGrid`, solver configuration/time and forces are compared exactly (`np.testing.assert_array_equal`).
- Every updated index `.visit` and `.pvd` is byte-compared, and each newly emitted `.vtu` is also compared byte-for-byte between the two runs. Stream frame names and physical-time scheduling targets are compared.
- The CPDI dynamic scenario starts with a nonuniform 2-component velocity field and generates measurable shear/rotation. This tests reconstruction of the deforming $\mathbf F$, domain geometry, and corresponding animation. The high-amplitude scenario is a *restart stress test*, not a validated reference solution.

### Run commands

From the GeoPyMPM repository root:

```bash
source .venv/bin/activate
python examples/22_extended_restart_verification.py
python -m validation.stage10g_c.extended_restart --steps 120
python -m pytest tests/test_stage10g_c_extended_restart.py -q -rs
python -m pytest -q
```

The example leaves inspection artifacts under `results/stage10g_c_example/cpdi/`:

```text
results/stage10g_c_example/cpdi/
├── checkpoints/
│   ├── state_00000030.json + state_00000030.npz
│   ├── state_00000060.json + state_00000060.npz
│   └── state_00000090.json + state_00000090.npz
├── continuous/
│   ├── particles.visit, grid.visit, domains.visit
│   ├── particles.pvd, grid.pvd, domains.pvd
│   └── ... .vtu snapshots
└── restart/
    ├── particles.visit, grid.visit, domains.visit
    ├── particles.pvd, grid.pvd, domains.pvd
    └── ... .vtu snapshots
```

In **VisIt**, open `restart/domains.visit` to inspect the deforming CPDI quadrilaterals or `restart/particles.visit` to inspect particle fields. **Do not open `.npz` or `.json` as visualization.**

If that example output directory already exists, select a fresh directory with `--output-dir ...`; the example refuses overwrite.

## 3. Results in the development environment

For the 120-step validation (three checkpoints at steps 30, 60 and 90), all four methods preserved full state, diagnostics and time-series indexes on every step.

| Method | Steps | Restart points | Frames for each enabled stream | Peak $|F_{xy}|$ | Peak polar angle |
|:--|--:|:--|--:|--:|--:|
| MPM | 120 | 30 / 60 / 90 | 31 | 0.0338 | 1.94° |
| uGIMP | 120 | 30 / 60 / 90 | 31 | 0.0345 | 1.99° |
| cpGIMP | 120 | 30 / 60 / 90 | 31 | 0.0345 | 1.99° |
| CPDI1 (dynamic) | 120 | 30 / 60 / 90 | 31 | 0.2620 | 16.15° |

Both the continuous and restarted CPDI variants produce **three** output streams (particles/grid/domains); the other methods produce particles/grid.

Example 22 prints paths to the checkpoint `.json` and `.visit` animations. The script operates on saved solver state **without applying a material or load modification**; branch restart with altered physics is a separate future feature.

## 4. Failure scenarios audited

The automated tests include:

1. Reject a corrupted or truncated `.npz` through SHA-256 verification.
2. Reject missing `.npz`, missing `.json`, malformed JSON and unsupported schema versions.
3. Reject missing, extra, or inconsistent `.vtu`, `.visit`, `.pvd` before resuming animations.
4. Reject overwriting an already-published checkpoint at the same step.
5. Permit loading an older intact checkpoint if a newer checkpoint is corrupted — **the user must select the older manifest explicitly**.
6. Deny animation continuation from the older checkpoint if the visualization directory has progressed beyond it; copying the snapshot-era visualization folder to a new branch directory is supported.
7. Confirm independent Python-process restart and exact reconstruction of CPDI domain corners.
8. Confirm checkpoint writing/reading does not itself mutate input data.

## 5. Limitations to carry into Stage 10G-D

- **No automatic checkpoint recovery or retention policy**: the user chooses an intact checkpoint; the stage does not automatically discover the newest valid pair.
- **Visualization `.vtu` contents are not cryptographically authenticated by the checkpoint**. Existing B2 recovery checks the matching *set of filenames* and exact `.visit`/`.pvd` contents. A `.vtu` overwritten with different bytes under the same filename could remain undetected; add optional file hashes or archive manifest in a future stage.
- Checkpoint file publishing uses temporary data and a JSON commit marker; this stage does not simulate an actual power loss / storage-controller failure or guarantee cross-filesystem `fsync` durability.
- **Cross-hardware bitwise equivalence** is not guaranteed; tests were executed within one software/runtime environment.
- No automatic stage event, particle insertion/deletion, elastoplastic state, contact history, or multiphysics state.
- Long-run visual snapshots may consume substantial storage. Benchmark write/read latency, sizes, and retention policies in Stage 10G-D.
- A valid checkpoint captures a **specific state** and does not by itself authorize changing material parameters that affected the prior stress history.

## 6. Stage gate

Stage 10G-C passes when all four method chains have identical per-step state and indexes, large-deformation CPDI shows a nontrivial $\mathbf F$, and the failure-mode tests reject damage safely. Stage 10G-D should subsequently audit I/O throughput, file retention, manifest safety, tamper detection for VTU, and recovery policy before Stage 11 elastoplastic materials.
