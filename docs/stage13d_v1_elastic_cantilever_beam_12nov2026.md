# GeoPyMPM Stage 13D_V1 — Elastic Cantilever Beam under Self-Weight

**Status:** reproducible explicit-dynamics benchmark and **preliminary dynamic-relaxation** verification; **not** a completed full-gravity, nonlinear quasi-static validation.

**Baseline:** GeoPyMPM Stage 13D (Stage 13A–13C include MC; V1 does not alter MC or other `src/geopympm` code). Applied as an incremental patch. **Date:** 12 November 2026.

## 1. Published problem and exact provenance

Iaconeta, Larese, Rossi & Oñate, *Discrete-continuum hybrid modelling of flowing and static regimes* (2020), Section 4.5.1, "2D cantilever beam. Static analysis": https://www.scipedia.com/public/Iaconeta_et_al_2020a

The paper's cantilever is **8 m long, 1 m high, 1 m thick**, density **1000 kg/m³**, **E=90 MPa**, **nu=0**, plane strain, with self-weight. Its reported *small-deflection beam-theory* tip result is **−0.67806 m** and the *fine-mesh nonlinear/quasi-static numerical* reference is **−0.67433 m**. The published continuum material is **hyperelastic**, with a static formulation different from GeoPyMPM's present explicit implementation.

**This patch does not claim to reproduce the publication exactly.** It is a staged verification of the currently available explicit solver, using available `LinearElastic` or `CompressibleNeoHookean`. The two choices are **not guaranteed** to be the same hyperelastic energy law as the publication. The full-gravity case may reveal numerical instability, and this must be recorded.

## 2. Model setup

```
     x = 0                LENGTH L = 8 m
     fix both v components at x=0
     |==================================|  y=1 m
     |    rho=1000 kg/m3, E=90MPa       |   v=0 at start
     |==================================|  y=0 m
                 g ↓

     thickness = 1 m; plane strain
```

- Rectangular particle seeding with **2×2 particles per grid cell** by default; structured background grid (`dx=dy=0.5 m` by default), padded around the solid.
- Left support is implemented with **nodal prescribed zero velocity** in both components at the `x=0` grid line, for nodes covering the beam/root neighborhood. No boundary traction is imposed.
- Default `interpolation=mpm`, `scheme=usl`, `pic_fraction=0` (FLIP); `kinematics_update=exponential`.
- **Numerical relaxation:** each successful step scales velocity by `exp(-r*dt)`, `r=2 s^-1` by default. This is *algorithmic/artificial damping* applied between steps, **not** a material damping law or a change to the GeoPyMPM solver.
- The body acceleration is smoothly ramped with `s^2*(3-2s)`, `s=min(1,t/ramp_time)`, default ramp time `1.5 s`; no sudden self-weight impulse.
- **Default load is intentionally 10% of the publication:** `g=0.981 m/s²`, to create a controlled small-deformation verification. Full publication gravity `g=9.81 m/s²` requires `--gravity-scale 1.0`, and has not passed a full run on the baseline grid.
- The position monitor is the **rightmost initial particle column** (default `x0=7.875 m`), not the exact geometric tip at `x=8 m`. An analytical formula is evaluated at *both* positions to prevent a misleading error comparison.

## 3. Independent beam-theory calculation

Let `b=h=1 m`, `A=bh`, `I=bh^3/12`, `As=5A/6`, `G=E/[2(1+nu)]`, and line load `w=rho*g*A`.

The small-deflection cantilever deflection under uniform weight, including shear deformation, at arbitrary `0<=x<=L` is

\[
u_y(x)=-\frac{w x^2 (6L^2-4Lx+x^2)}{24EI}
        -\frac{w(Lx-x^2/2)}{G A_s}.
\]

At the geometric tip `x=L`,

\[
u_y(L)=-\frac{wL^4}{8EI}-\frac{wL^2}{2G A_s}.
\]

- Full g: **−0.6780672 m**, agreeing with the paper's rounded **−0.67806 m**.
- 10% g: **−0.06780672 m** at `x=8.0 m`.
- 10% g: **−0.06641131696 m** at the actual *monitor column* `x=7.875 m` on the default mesh.

These calculations are purely analytical and do **not** reuse the GeoPyMPM return mapping or MPM kernels.

## 4. Run instructions

Install the patch from the GeoPyMPM repository root:

```bash
unzip -o ~/Downloads/GeoPyMPM_Stage13D_V1_Elastic_Cantilever_Validation_patch_12Nov2026.zip -d .
source .venv/bin/activate
python -m pytest -q tests/test_stage13d_v1_cantilever.py
```

Smoke test (10% gravity, still dynamic):

```bash
python examples/30_stage13d_v1_cantilever_beam.py \
  --steps 500 --output-dir results/stage13d_v1_smoke
```

Longer test used for the supplied reference run (10% gravity, **30,000 steps / 6 s**, may take time):

```bash
python examples/30_stage13d_v1_cantilever_beam.py \
  --gravity-scale 0.1 --material linear \
  --cell-size 0.5 --ppc 2 --dt 0.0002 \
  --ramp-time 1.5 --damping-rate 2.0 --steps 30000 \
  --sample-every 200 --vtu-every 3000 --plot \
  --output-dir results/stage13d_v1_10pct
```

Full paper gravity (a **challenging exploratory run**, not yet validated):

```bash
python examples/30_stage13d_v1_cantilever_beam.py \
  --gravity-scale 1.0 --steps 10000 --plot \
  --output-dir results/stage13d_v1_full_gravity
```

A failing step exits nonzero but leaves useful files and a **`summary.json` with a numerical failure reason**. Each output directory must be unused. Avoid silently comparing an intermediate dynamic result with a static reference.

Alternative interpolation and constitutive-law screening:

```bash
python -m validation.stage13d_v1.cantilever --help
# Example:
python -m validation.stage13d_v1.cantilever \
  --material cnh --interpolation gimp --scheme musl \
  --steps 1000 --output-dir results/stage13d_v1_cnh_gimp_smoke
```

## 5. Files written

- `summary.json`: model inputs, physical time, convergence *screen*, measured deflection, independent reference(s), and failure reason if any.
- `diagnostics.csv`: `time_s`, near-tip displacement, maximum particle speed, kinetic energy, min J, stress, gravity fraction, etc.
- `final_particles.csv`: initial/current coordinates, displacements, selected stresses, J for all material points.
- With `--plot`: `tip_history.png`, `max_speed_history.png`, `beam_shape.png`.
- With `--vtu-every N`: `visualization/particles.pvd` (open in ParaView), `.visit` (VisIt), and particle snapshots with physical-time metadata.

## 6. Reproduced results and interpretation

**10% gravity, Linear Elastic, default grid, 30,000 steps, 6 seconds**:

| Measure | Observed |
|---|---:|
| Particles | 128 |
| Total mass | 8,000 kg |
| Near-tip monitor x0 | 7.875 m |
| Near-tip displacement at 6 s | **−0.06429878 m** |
| Analytical at same x | **−0.06641132 m** |
| Relative difference at same x | **~3.18%** |
| Analytical at exact tip x=8 m | −0.06780672 m |
| Relative difference to exact-tip formula (not same location) | ~5.17% |
| Maximum particle speed over final 1 s | 0.00095372 m/s |
| Near-tip drift over final 1 s | 0.00037723 m |
| Minimum J over sampled history | 0.998419 |

The case passes a **heuristic relaxation screen** (full scaled load maintained at least 1 s; max speed below 0.002 m/s; tip drift in last second below 0.004 m). This is *not* a proof of equilibrium. The pressure and stress redistribution, numerical damping, grid-sensitivity and continuum reference assumptions still need analysis.

**Full gravity, Linear Elastic, initial 0.5-m grid:** at **step 3183**, **t=0.6366 s**, the run stopped because the current CFL estimate fell below `dt=0.0002 s`. The last committed near-tip deflection was roughly **−0.1185733 m**. The run is flagged as **numerical failure**, not a meaningful static deflection or failed physical material.

## 7. What counts as validated later?

- Compare at identical applied gravity and *identical monitored location*.
- Refine grid spacing and particles per cell, reduce `dt` and vary damping rate; demonstrate convergence of measured deflection and stability.
- Establish static equilibrium by checking force/residual or more stringent kinetic and displacement stabilization, not simply that a run has finished.
- Compare linear and hyperelastic responses while acknowledging that the published hyperelastic constitutive law may differ from CNH.
- Investigate coarse-grid / large-deformation instabilities at full gravity. Do not decrease `dt` indefinitely without testing stress/objectivity and particle-domain effects.

**Stage 13D_V1** is therefore the first **boundary-value validation case in a chronological validation series**. It is a reproducible diagnostic benchmark with a stable reduced-load baseline and an explicitly documented full-load limitation—not a complete published-method reproduction.
