# GeoPyMPM — Stage 10G-D: Reliability & Performance Audit (Initial Assessment)

**Date:** 9 November 2026  
**Source baseline:** Uploaded `geopympm-main-stage10G_C.zip` (Stage 10G-C).  
**Status:** **10G-D INITIAL AUDIT COMPLETE; STAGE 10G-D NOT YET CLOSED.**  
**Changes in this audit package:** Benchmark script, regression tests, documentation and measured CSV/JSON only. No modification to `src/geopympm/`, numerical methods, checkpoint schema or the existing Stage 10G-C tests.

## 1. Objective and gates

Audit reliability of checkpoint/restart, VTK time-series continuation and write safety, and measure warmed explicit solver and I/O overhead from 64 to 16,384 particles for MPM, uGIMP, cpGIMP and CPDI1. The goal is to identify required remediation before introducing Stage 11 elastoplastic material history. This is **not** constitutive validation, a durability guarantee or a finished performance optimization.

## 2. Test results

- Stage 10G A1/A2/A3, B1/B2 and C regression selection: **117 passed** (`PYTHONPATH=src python -m pytest -q tests/test_stage10g_a1_vtk_writer.py tests/test_stage10g_a2_mesh_vtk.py tests/test_stage10g_a3_time_series.py tests/test_stage10g_b1_checkpoint.py tests/test_stage10g_b2_restart.py tests/test_stage10g_c_extended_restart.py`).
- Stage 10G-D preliminary tests: **1 passed, 1 expected failure (xfail)**. The xfail records a demonstrated VTK snapshot-integrity gap, not a resolved bug.
- Attempted full regression suite exceeded the execution time limit after approximately 65% progress without a reported failure. **Do not claim full-suite PASS.**

## 3. Reliability matrix

| Area | Finding | Severity / next action |
|---|---|---|
| Checkpoint `.npz` integrity | SHA-256 checked against checkpoint JSON; corruption rejected in B/C tests | Existing baseline PASS for accidental corruption |
| Checkpoint JSON and schema | Schema/version/malformed metadata rejected in B/C tests | Existing baseline PASS; cryptographic authenticity of JSON is **not** provided |
| Per-step multi-restart | Stage 10G-C compares state and indexes for 4 interpolation schemes | Existing baseline PASS for tested 2D elastic configurations |
| Missing/extra VTU and modified `.visit`/`.pvd` | Output restore rejects mismatched file lists/index contents | Existing baseline PASS |
| **Modified VTU contents with unchanged filenames** | **REPRODUCED: restore accepts a rewritten `particles_000000.vtu`** | **OPEN HIGH priority**: bind hashes to checkpoint, verify before restore |
| Automatic newest-intact checkpoint selection | No discovery/fallback API; previous intact checkpoint must be explicitly chosen | OPEN; implement fail-closed selection and safe branching |
| File retention/quota protection | No retention policy; VTU snapshots can accumulate | OPEN; define explicit verified checkpoint retention + snapshot policy |
| Crash durability | NPZ and JSON temp-file fsync and final renames exist; directory fsync / simulated sudden power loss not established | OPEN; don't claim crash durability |
| Concurrent writers | Single-writer directory is assumed; existence checks plus `os.replace` are not a guaranteed no-clobber protocol under concurrency | OPEN; add directory locking or explicit single-writer enforcement |
| Material history | Schema 1 supports 2D LinearElastic only; Stage 11 state must not be silently dropped | OPEN future schema/design gate |
| Platform reproducibility | Same-environment bitwise verification; cross-hardware bitwise portability unproven | Documented limitation |

**Reproduction of VTU integrity gap:** Construct a Stage 10G-C MPM solver and OutputManager, emit initial VTU, write a checkpoint, overwrite the contents of `particles_000000.vtu` with `b"tampered"`, then invoke `restore_output_manager(manifest, load_checkpoint(manifest))`. It accepts the directory because only filenames and index bytes are checked. See `tests/test_stage10g_d_audit.py` for an xfail gate.

## 4. Benchmark setup

- Python 3.13.5, NumPy 2.3.5, Numba 0.65.1 in the audit environment. Figures depend on machine load, processor and I/O subsystem.
- A regular 2D block, **4 particles/cell**, structured grid with one-cell padding, `LinearElastic(E=1200 Pa, nu=0.25)`, density 1000 kg/m³, `dt=0.001 s`, small gravity, corotational stress, exponential F and affine-exponential advection.
- First solver step timed separately to reveal JIT/cache start-up cost. Each warmed time is median of *per-step* times over 3 batches of 8 steps for small cases and 2 batches of 5 steps for large cases.
- For small cases, output after initial state and 8 steps; both particle/grid VTK streams, plus CPDI domains for CPDI1, inclusive of `.visit` and `.pvd` updates. Large cases use `--skip-vtk` to avoid excessive artifact sizes.
- Checkpoint sizes include JSON+compressed NPZ. Near-static repetitive data compress well; do not extrapolate these byte sizes to complicated spatially heterogeneous plastic states.
- `tracemalloc` was disabled during timing to avoid interpreter-level instrumentation skew. Native peak RAM is **not** measured per individual case; process peak RSS is cumulative.

### 4.1 Warmed solver timing (median milliseconds per step)

| Method | 64 | 256 | 1,024 | 4,096 | 16,384 |
|---|---:|---:|---:|---:|---:|
| MPM | 0.97 | 1.13 | 2.27 | 4.71 | 20.78 |
| uGIMP | 1.88 | 3.04 | 7.19 | 24.81 | 126.48 |
| cpGIMP | 1.94 | 3.51 | 7.29 | 24.22 | 125.73 |
| CPDI1 | 2.02 | 2.82 | 6.17 | 16.18 | 109.90 |

These measurements are **indicative** (short runs, one local environment); no universal performance claim should be made. The relative ranking is a measurement for this specific setup, not a general claim that CPDI1 is faster than uGIMP.

### 4.2 I/O measurements

| Method (1,024 particles) | Checkpoint write | Checkpoint load | Checkpoint pair size | VTK frame-write + index median |
|---|---:|---:|---:|---:|
| MPM | 7.03 ms | 5.92 ms | 82.6 KiB | 63.4 ms |
| uGIMP | 8.55 ms | 5.76 ms | 91.2 KiB | 61.8 ms |
| cpGIMP | 7.26 ms | 6.10 ms | 91.2 KiB | 64.8 ms |
| CPDI1 | 8.12 ms | 6.46 ms | 92.7 KiB | 100.1 ms |

For 16,384 particles (without VTK), checkpoint write median ranged from 53.0 to 59.7 ms, load from 17.7 to 23.9 ms in these trials. With every step exported, output can dominate run time. Snapshot frequency must be based on accuracy/inspection needs rather than blindly saving every step.

Machine-readable outputs are stored at `validation/results/stage10g_d/{small,large}_{measurements.csv,summary.json}`.

## 5. Recommended completion sequence

1. **10G-D1 (this package):** Establish reliable baseline metrics, test inventory and explicit risk register. **Done.**
2. **10G-D2:** Add opt-in/checkpoint-versioned hashes for each required VTU snapshot and verify bytes before restoring OutputManager. Keep backwards compatibility explicit, not silent. Add corruption, missing-hash and intact-read tests.
3. **10G-D3:** Implement optional newest-valid checkpoint discovery and **safe fallback**, only with a compatible visualization snapshot set. Implement directory durability and collision tests as feasible; introduce a conservative retention policy that never deletes the only verified good checkpoint.
4. **10G-D4:** Profile P2G/G2P/interpolation vs Python orchestration and VTU/index writing on realistic particle states at 10k–100k+ scale. Apply non-regressive optimizations only after identifying a bottleneck; compare with baseline.
5. **Closure:** Rerun whole regression suite, extended multi-restart verification, failure injection and benchmark. Do not declare Stage 10G-D complete until the agreed reliability gates pass and a performance report records before/after or accepted limits.

## 6. Run instructions

From repository root (editable install or `PYTHONPATH=src`):

```bash
python -m pytest -q tests/test_stage10g_d_audit.py
python -m validation.stage10g_d.audit --output-dir results/10gd_small \
  --sizes 4 8 16 --steps 8 --repeats 3
python -m validation.stage10g_d.audit --output-dir results/10gd_large \
  --sizes 32 64 --steps 5 --repeats 2 --skip-vtk
```

Each directory must be NEW. Use a different output directory to rerun. Avoid comparing absolute performance numbers from different computers without controlling workload and compiler warmup.
