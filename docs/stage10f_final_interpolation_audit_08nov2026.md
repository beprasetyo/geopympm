# GeoPyMPM Stage 10F — Final Interpolation Verification and Audit

**Audit date:** 8 November 2026  
**Baseline:** `GeoPyMPM_Stage10E_B2_General2D_full_08Nov2026.zip` (Stage 1–10E-B2)  
**Result:** Numerical interpolation audit passed; source solver/kernels unchanged. **Not** a certification for general finite-strain geomechanics or landslide runout.

## 1. Scope and artifacts

This audit consolidates the implementation of standard Quad4 MPM, unchanged-domain GIMP (`gimp`), contiguous-particle GIMP (`cpgimp`), and first-order CPDI (`cpdi`) in the explicit 2D solver. New independent randomized tests inspect the production interpolation and transfer kernels. A reproducible warm-runtime comparison includes solver-level Python validation and state copies. Original Stage-10 benchmark JSONs are indexed into a provenance-aware CSV; those benchmarks are **not** silently rerun or reformulated.

Additions in this stage:

- `validation/stage10f/final_audit.py`: randomized particle/interpolation/P2G/force checks; optional warm timing.
- `validation/stage10f/historical_inventory.py`: provenance-preserving extraction from existing JSON results, with explicit missing-source reporting.
- `tests/test_stage10f_audit.py`: 19 new deterministic tests.
- `validation/results/stage10f/`: JSON, CSV, PNG results.
- This report and optional `docs/stage10f_overleaf_addendum.tex`.

**No changes** to `src/geopympm/` are included. Source-code checks are reviews of the existing implementation and do not imply every possible defect has been eliminated.

## 2. Source-code audit

| Component | Audited implementation and finding | Current limitation |
|---|---|---|
| Point MPM | `interpolation/mpm.py`, `kernels/interpolation_quad4.py`: Quad4 weights and gradients at particle center, four-node support. | Piecewise linear gradients jump across cell boundaries. |
| uGIMP | `interpolation/gimp.py`, `kernels/interpolation_gimp.py`: tensor-product, exact averages of 1D nodal hats and exact spatial derivatives; `domain_size` is full width. | Axis-aligned fixed rectangular domains; each side must remain <= one grid cell. |
| cpGIMP | `interpolation/cpgimp.py`, `kernels/domain_update.py`: `domain_size = domain_size0 * diag(F)` after the current step. | Requires positive diagonal of F and domain widths <= cell size; cannot represent arbitrary rotations/shear of particle-domain shape. |
| CPDI1 | `interpolation/cpdi.py`, `kernels/interpolation_cpdi.py`: four convected parallelogram corners; full F used; 4 corner evaluations x 4 Quad4 nodes = 16 contribution slots (duplicates allowed). | First-order parallelogram approximation; corner/domain geometry must remain on the grid and nondegenerate; gaps/overlaps possible under nonaffine motion. |
| Transfer kernels | `kernels/p2g.py`, `kernels/grid_update.py`, `kernels/g2p.py`, `kernels/kinematics*.py`: loops use actual `node_ids.shape[1]`, supporting 4 and 16 slots. | Serial particle-to-node assembly; extra contribution slots and Python validation affect runtime. |
| Solver sequencing | `solvers/explicit.py`: evaluate at `(x_n, domain_n, F_n)`, P2G, nodal forces, kinematics/stress, G2P, then update cpGIMP/CPDI domain. Failure restores particle and transient grid state. | Explicit updated-grid-velocity/USL-like scheme, **not** MUSL. Fixed grid with no contact or traction-integrating BCs. |
| Nodal update | `kernels/grid_update.py`: computes `a_i = f_i / m_i` for `m_i > 0`; includes velocity BCs. | Near-zero positive nodal mass can yield large raw acceleration diagnostic values; no ad hoc cutoff was introduced. |
| Consistency checks | `validate_current_gimp()` and `validate_current_cpdi()` check stale support/geometry; invoked across P2G, grid update, kinematics and G2P. | Deliberate repeated recomputation incurs overhead; do **not** remove without a validated data versioning/cache strategy. |

### Data conventions and interface assumptions

- `ParticleSet` uses Cauchy stress, tension positive; Voigt order `[xx, yy, zz, xy, yz, xz]`; 2D plane strain, normally one-meter out-of-plane thickness for the published benchmarks.
- `domain_size0` is the full reference side length; `domain_size` is the current full axis-aligned rectangle size in uGIMP/cpGIMP, but a **bounding-box envelope** in CPDI1. CPDI shape functions are built from `domain_size0` and F, **not** from the envelope.
- Current particle mass/volume and interpolation-domain area are distinct quantities; comparisons against integrated continuum force use specially tiled domains where appropriate.
- Grid spacing is fixed. Positivity and finite-state checks do not replace physical contact/free-surface or strong-distortion verification.
- `kinematics_update='exponential'`, `stress_update='corotational'`, `advection_update='affine_exponential'`, and `pic_fraction=0.0` must be explicitly selected when recreating the reported finite-deformation benchmark configuration; they are not all default values.

## 3. New independent randomized identities

For each method, 128 reproducible random interior particle positions were generated on a rectangular 2D grid; cpGIMP received random positive diagonal F, and CPDI received random stretches, rotations and shear. The **production** kernels were used. Independent reconstructions used the node coordinate array.

The verified identities are

\[
\sum_i S_{ip}=1,\quad \sum_i\nabla S_{ip}=\boldsymbol0,\quad
\sum_i S_{ip}\mathbf x_i=\mathbf x_p,\quad
\sum_i \mathbf x_i\otimes\nabla S_{ip}=\mathbf I.
\]

The reconstructed velocity `v(x)=b+A*x` and gradient A were additionally compared with direct analytic values. Mapping mass/momentum with production P2G kernels was checked against particle totals, and the sum of assembled internal forces was checked to be zero for uniform stress and complete grid support.

| Interpolation | Slots | Max `sum(S)-1` | Max `sum(x_i tensor gradS)-I` | Checks |
|---|---:|---:|---:|---|
| MPM | 4 | 1.11e-16 | 8.88e-16 | All 8 checks passed |
| uGIMP | 16 | 3.33e-16 | 1.13e-14 | All 8 checks passed |
| cpGIMP | 16 | 2.22e-16 | 1.87e-14 | All 8 checks passed |
| CPDI1 | 16 | 0 | 2.13e-14 | All 8 checks passed |

All 19 additional Stage-10F pytest tests passed, using several random seeds, affine fields, a multi-cell CPDI domain, a single-cell GIMP/point-MPM equivalence check, and output-file verification. **These identities do not validate arbitrary constitutive response.** JSON: `validation/results/stage10f/10F_audit.json`.

## 4. Consolidation of existing Stage-10 benchmarks

The new `10F_historical_validation.csv` contains 28 rows extracted from source JSON files. Each row records its original JSON, method, metric, units and applicability. Examples:

| Case | Relevant measured result | Interpretation / boundary |
|---|---|---|
| 10A moving block force | MPM RMS force error 4.2008 N; uGIMP 2.99e-12 N | Exact tiled-domain integration under prescribed translation; not a general stress test. |
| 10A moving-frame wave | 4 PPC, pure FLIP: time-RMS stress error MPM 24.72%, uGIMP 25.28%; boosted-versus-rest difference around 39% | No universal stress accuracy advantage or frame invariance demonstrated. |
| 10D prescribed large shear | CPDI1 F error 1.78e-15 at gamma 2; stress max error 0.00283 Pa | Validated affine kinematics and Jaumann-specific stress update against a manufactured solution; not force-driven. |
| 10E-A nonlinear longitudinal wave | Stress error at h=0.25 m: MPM 26.462%, uGIMP 1.546%, cpGIMP 1.345%, CPDI1 1.504% | Exact pre-shock 1D finite-stretch **reduction**, not general nonlinear 2D. |
| 10E-B constrained shear | Four methods ~5.485% shear-stress error at h=0.25 m | 1D shear characteristic benchmark embedded in 2D; vy constrained. |
| 10E-B2 oblique P/S waves | Four methods ~17–18% total stress error at h=0.25 m | Both grid velocity components free; **linearized** continuum reference valid as a small-amplitude comparator only. |

The original reports and their exact parameters remain authoritative. Results are not pooled into a single composite accuracy score, since reference physics, grid spacings, loading schemes, and metrics differ.

### Open physical/numerical findings

1. **Galilean sensitivity** (10A): pronounced translated-frame stress difference; uGIMP does not eliminate it under the tested settings.
2. **Near-massless nodal accelerations** (10E-A.1 and 10E-B2): stress/force contributions to G2P were negligible in the specifically diagnosed 10E-A.1 case, but a universal guarantee has not been established; do not blindly introduce mass clipping.
3. **CPDI1 nonaffine domain nonconformity** (10D, 10E-B2): shared edges may separate or overlap. At a strong-amplitude mixed-wave test, about 0.020 m maximum mismatch was measured. This is not a measured uncovered-area fraction.
4. **Limited general finite-strain constitutive validation**: objective Jaumann hypoelastic integration is not synonymous with an energy-consistent finite-strain hyperelastic law. Small-amplitude mixed P/S linear reference is not exact at large amplitude.
5. **Boundaries and loading**: all particle-domain corners must remain inside the fixed computational grid. There is no production contact, traction integration, moving-grid extension or free-surface remapping.
6. **Time-step stability**: the elastic-wave CFL estimate is heuristic, not a proof for highly distorted CPDI domains.

## 5. Repeatable warm-runtime comparison

The benchmark executes **identical particle counts, physical mass/volume, initial uniform velocity, h=0.25 m and an unstressed linear elastic material** for all four methods. The fixed background grid contains 4,225 nodes. It warms the compiled code before collecting medians of repeated timing batches, separately for (a) interpolation evaluation including Python validation and (b) full solver step including P2G, grid force/momentum, kinematics, objective stress integration, G2P, rollback snapshots and revalidation. It is a throughput probe, not a physical accuracy test.

Environment: Python 3.13.5, NumPy 2.3.5, Numba 0.65.1, single CPU process. Time values will vary on different machines.

| Particles | MPM full step | uGIMP full step | cpGIMP full step | CPDI1 full step |
|---:|---:|---:|---:|---:|
| 256 | 1.175 ms | 3.101 ms | 3.230 ms | 2.844 ms |
| 1,024 | 2.073 ms | 7.852 ms | 7.734 ms | 5.812 ms |

At 1,024 particles, the measured full-step factors relative to MPM are about **3.79x** (uGIMP), **3.73x** (cpGIMP), and **2.80x** (CPDI1). These results include Python correctness guards and repeated full interpolation re-evaluation, and **must not be interpreted as pure kernel throughput or extrapolated to HPC scaling**. Additional P2G scatter contention and memory behavior are not characterized here. Raw data: `10F_performance.csv`; chart: `10F_step_performance.png`.

## 6. Final validation-status matrix

| Capability | Status at Stage 10F | Qualification |
|---|---|---|
| Quad4, uGIMP, cpGIMP, CPDI1 interpolation | **Implemented** | 2D structured grid only |
| Partition of unity, affine reproduction, transfer conservation | **Verified** | Independent randomized and regression tests |
| CPDI affine shear/stretch/rotation | **Verified** | Analytic prescribed-affine cases |
| Domain-based cell-crossing continuity | **Verified for designed cases** | Particularly uGIMP tiled-domain force integration |
| Force-driven nonlinear axial wave | **Validated for special reference problem** | Exact pre-shock 1D finite-stretch reduction |
| Constrained nonlinear shear wave | **Validated for special reference problem** | vy=0; core only |
| General 2D force-driven motion | **Verified** | Mixed P/S small-amplitude linear comparator; not general finite-strain accuracy |
| Full Galilean invariance | **Open** | Measurable frame sensitivity |
| CPDI domain nonconformity in nonaffine flow | **Open** | Edge mismatch/overlap observed |
| Contact, elastoplasticity, FEM transfer | **Not implemented** | Do not claim validated runout |

## 7. Audit disposition and Stage 11 entry gate

**Disposition: Stage 10F audit complete, conditional approval to proceed with Stage 11 material-point constitutive development.** No confirmed critical interpolation defect was found in the scoped audit, and no changes to existing kernels/solver are recommended based on these tests alone. This is **not a declaration of a production-ready general large-deformation MPM solver**.

Preserve the existing default `interpolation='mpm'` for backward compatibility. Use `interpolation='cpdi'` **explicitly** for problems requiring convected parallelogram particle domains, subject to mesh-boundary, domain-degeneracy, geometry-closure and convergence checks. For matched axis-aligned nominal domains, uGIMP/cpGIMP remain valid comparison baselines. Do not select a universal winner from incomparable benchmark classes.

Recommended next development: validate standalone **Von Mises elastoplastic integration** under controlled material-point loading/unloading (including objective spin where appropriate), then Mohr–Coulomb under stress paths relevant to geotechnics. Couple the new material to MPM only after its local stress integration passes independent tests. Keep general 2D finite-strain accuracy, free-surface/contact handling, and CPDI domain nonconformity as explicit blockers for claims about predictive landslide runout.

## 8. How to reproduce

Activate `.venv` and run at the repository root:

```bash
python -m validation.stage10f.final_audit --performance
python -m validation.stage10f.historical_inventory
python -m pytest tests/test_stage10f_audit.py -q
python -m pytest -q
```

When the full suite exceeds available execution time, use separated groups:

```bash
python -m pytest -q -k 'not stage6f'
python -m pytest -q tests/test_stage6f_energy_convergence.py
```

**Observed in the audit environment:** `516 passed, 27 deselected` plus `27 passed` separately = **543 individually passing tests**. The original Stage 10E-B2 source previously had 524 passing tests in a single local run. The combined 543-test invocation did not complete within this audit environment's limit, so it must not be described as a successfully completed one-process regression.
