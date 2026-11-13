# Stage 13D_V1 — Live Progress Reporting (13 November 2026)

**Baseline:** GeoPyMPM Stage 13D_V1 cantilever validation. This is a **diagnostic-only incremental patch**, not a new numerical stage. Solver, constitutive models, numerical kernels, time stepping, VTU formatting and checkpoint schema are unchanged.

## Motivation

Long explicit MPM simulations (e.g. 60,000 steps / 6 s at `dt=1e-4 s`) previously appeared idle because the script wrote diagnostics and printed its status only after completing the run. Now the V1 runner prints a line after every chosen number of **committed** steps, with immediate `flush=True`. The output is visible on the Ubuntu terminal even while CSV/VTU results are still being generated.

## Usage

From the root repository, after applying the incremental patch:

```bash
python -m pytest -q tests/test_stage13d_v1_progress.py tests/test_stage13d_v1_cantilever.py
python examples/30_stage13d_v1_cantilever_beam.py \
  --gravity-scale 1.0 --interpolation gimp --dt 0.0001 --steps 60000 \
  --sample-every 200 --vtu-every 3000 --progress-every 500 --plot \
  --output-dir results/stage13d_v1_gimp_progress
```

Default progress interval: **500 steps**; change using `--progress-every 100`, `1000`, etc. To suppress terminal reporting, choose `--progress-every 0`.

Example format (numbers for illustration only):

```text
[Stage 13D_V1] START | target=60,000 steps | progress-every=500 | initial Numba compilation may take extra time
[Stage 13D_V1] step=500/60,000 (  0.8%) | t=0.05000 s | wall=00:00:04 | speed=warming up | ETA=warming up | tip_uy(m)=-0.0002 | vmax(m/s)=0.01 | Jmin=0.999
[Stage 13D_V1] step=1,000/60,000 (  1.7%) | t=0.10000 s | wall=00:00:05 | speed=500.0 steps/s | ETA=00:01:58 | tip_uy(m)=-0.0008 | vmax(m/s)=0.015 | Jmin=0.998
...
[Stage 13D_V1] COMPLETED | committed=60,000/60,000 steps | t=6.00000 s | wall=00:02:00
```

The `t` value is **physical simulation time**; `wall` and `ETA` are **real-world elapsed and estimated remaining times**. ETA is based on the last two progress reports (not a guarantee), so it can fluctuate as Numba compiles, VTU output is written, and the load changes.

On a numerical failure, the reporter prints **NUMERICAL FAILURE** with the last *committed* step and the exception message. The existing V1 behavior still writes `summary.json`, `diagnostics.csv`, `final_particles.csv` and available plots, then exits with status code 2.

## Implementation and limits

- New reusable utility: `src/geopympm/io/progress.py` (`ProgressReporter`). Exported from `geopympm.io`.
- Updated V1 loop: `validation/stage13d_v1/cantilever.py`, with `run_case(..., progress_every=500)` and the CLI argument.
- The monitor samples `near-tip displacement`, `maximum particle speed` and `minimum J` **only** at progress reporting steps. It does not modify particle/grid arrays or the integration schedule. It is independent of `sample_every` (CSV) and `vtu_every` (VTU).
- For many particles, very frequent reporting can add Python diagnostics cost; a few hundred or thousand steps is generally sufficient.
- This patch only wires live reporting into V1. Other examples may later import and use `ProgressReporter` without modifying the numerical solver.

## Regression requirements

Test terminal output frequency, disabled mode, committed-step behavior on failure, and bit-for-bit equality of V1 CSV state/diagnostics with progress on versus off. Existing Stage 13D_V1 tests must continue passing.
