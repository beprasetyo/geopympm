# GeoPyMPM Stage 10G-D4 — Performance Profiling and Safe VTU Optimization

**Date:** 9 November 2026  
**Baseline:** Stage 10G-D3 (Stage 10G-C + incremental patches D1, D2, D3)  
**Patch type:** incremental files only; no repository repackaging  
**Scope:** *performance audit + one low-risk VTK ASCII writer optimization*

## 1. Summary

- Measured warmed explicit solver timings separately from Numba's first step.
- Profiled MPM, uGIMP, cpGIMP and CPDI1 using the same synthetic 2D LinearElastic case as Stage 10G-D1.
- Separately timed checkpoint serialization, NPZ validation/reload, standalone VTU, scheduled visualization, full VTU SHA-256 rehashing, and checkpoint write with VTU metadata.
- Optimized the shared Float64 VTU ASCII array serializer, used by particle, grid and CPDI mesh writers. When an array has 256 or more rows it now uses NumPy `savetxt` into an in-memory stream, retaining `%.17g` round-trip precision. Smaller arrays and all integer arrays retain the original serializer. **No binary VTU format change.**
- No changes to the explicit solver, interpolation basis/kernels, CFL estimate, rollback, constitutive update, checkpoint schema, restart, retention, or VTU hashing/integrity rules.

## 2. Motivating profiles and constraints

A short warm `cProfile` run (576 particles; 5 explicit steps) identified repeated consistency validation, including `validate_current_gimp`, as a substantial overhead for the finite-domain methods. Independent warm benchmarks also showed that ASCII VTU output can cost much more than one solver step.

**Important design decision:** repeated interpolation validation was **not disabled or bypassed** in this patch. Those checks reject stale/inconsistent interpolation at all standalone entry points and are bound to the error/rollback guarantees. Any future short circuit must preserve those guarantees, with adversarial and conservation tests. The P2G/G2P kernels remain serial (parallel deposition would require race-free accumulation), and all D2 VTU hashes continue to be recomputed from actual file bytes at checkpoint time.

## 3. Measured baseline versus D4

Benchmark environment: see `validation/stage10g_d4/results/benchmark_environment_09nov2026.json`.

The following columns are **median standalone default particle VTU write wall time** with no VTK Python dependency. Same synthetic case and requested settings on both implementations. The runs were sequential, not randomized; differences include noise and filesystem effects.

| Method | Particles | D3 baseline (ms) | D4 (ms) | Observed speedup |
|---|---:|---:|---:|---:|
| MPM | 1,024 | 48.0 | 39.5 | 1.21× |
| uGIMP | 1,024 | 44.1 | 40.6 | 1.09× |
| cpGIMP | 1,024 | 43.8 | 41.6 | 1.05× |
| CPDI1 | 1,024 | 47.4 | 41.2 | 1.15× |
| MPM | 4,096 | 174.5 | 171.0 | 1.02× |
| CPDI1 | 4,096 | 183.0 | 172.9 | 1.06× |

The exact, unrounded measurements and warm solver/checkpoint/hash timings are in `validation/stage10g_d4/results/benchmark_comparison_09nov2026.csv`. There is no guarantee of the same speedup on another machine or for other field selections. Variation in warm solver times cannot be attributed to this patch because **no solver code was changed**. In particular, it would be incorrect to claim that P2G, G2P, GIMP/CPDI or SHA-256 hashing have been accelerated.

## 4. Validation

- Targeted Stage 10G-A through D4 tests: **197 passed**.
- Additional solver and MPM/GIMP/CPDI regression tests: **112 passed**.
- D4-specific tests: **22 passed** (included in the 197); tests ensure `.17g` output is byte-identical with the previous serializer for representative random, nextafter, signed-zero, tiny and maximum Float64 values at 1/255/256/512 rows and 1/2/3/6/9 columns. Float64 parse/re-import is exact.
- Manual byte-for-byte check on *seven* representative 1,024-particle MPM, uGIMP and CPDI1 particle/grid/domain VTU files: **all seven identical** to D3 output.
- No full 740-test repository suite has been claimed.

Numerical and checkpoint verification is deliberately performed separately from performance measurements; no time-dependent throughput assertion is used as a pass/fail test.

## 5. How to run locally (Ubuntu, project root)

Install D4 **on top of D3**, activate your editable package environment, then run:

```bash
source .venv/bin/activate
python -m pytest -q tests/test_stage10g_d4_profile_vtk_performance.py
python -m pytest -q tests/test_stage10g_*.py
```

For a fresh profiling run (uses 4 particles per 2D cell and refuses to overwrite existing result directories):

```bash
python -m validation.stage10g_d4.profile \
    --methods mpm gimp cpgimp cpdi --sizes 8 16 32 \
    --steps 8 --repeats 3 --profile-steps 5 \
    --output-dir results/stage10g_d4_local
```

You will get `profile_summary.json`, `measurements.csv`, and per-case `solver_profile.txt` reports, plus the generated checkpoint/VTU files. For CPU profiling of much larger cases, consider a second run using `--skip-vtk`, then profile visualization separately to avoid excessive disk usage. **The profiler intentionally re-reads full VTU files to measure the SHA-256 hashing cost, and it cannot be used to infer results for real rainfall-induced slope failures.**

## 6. Stage 10G-D completion criteria and Stage 11 gate

The following Stage 10G-D work is implemented and has the stated regression evidence:

- D1 baseline audit, D2 VTU hash integrity, D3 checkpoint recovery/retention, D4 reproducible profiler and safe VTU optimization.
- Existing output metadata/checkpoint compatibility remains unchanged; no new state needs transferring to Stage 11.
- The broader test suite, independent physics validation and large-particle performance runs remain desirable before a performance claim or production deployment.

**Next development:** Stage 11 (elastoplastic models) should introduce material-specific history fields and must therefore explicitly design/version the checkpoint schema and restart contracts before claiming complete elastoplastic restarts. Avoid premature removal of input validation, rollback snapshots or integrity rehashing merely to improve benchmark timings.
