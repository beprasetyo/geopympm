# GeoPyMPM Stage 10B — Classical Contiguous-Particle GIMP (cpGIMP)

**Baseline:** `GeoPyMPM_Stage10A_Sensitivity_Full_28Oct2026.zip`  
**Date:** 28 October 2026  
**Scope:** Optional 2D axis-aligned domain evolution for existing explicit GIMP; no new constitutive model or CPDI.

## 1. Numerical definition

For each material particle, `domain_size0` stores the **full** initial side lengths
`(l_x0, l_y0)`. The existing Stage 8 deformation gradient is the cumulative
in-plane mapping `F` measured from the same MPM reference configuration.
The classical axis-aligned cpGIMP approximation is

```text
l_px(n+1) = l_px(0) * F_xx(n+1)
l_py(n+1) = l_py(0) * F_yy(n+1)
```

The particle interpolation domain remains a rectangle aligned to the grid.
Its size changes, but there is no separate rotation/shear of that domain.
GIMP shape weights/gradients are computed using the SAME exact integrated
bilinear grid hat as Stage 10A (`interpolation/gimp.py`, `kernels/interpolation_gimp.py`).

**Not the same as particle volume:** physical particle volume is still
`volume0 * det(F)` from Stage 8. The product `l_px*l_py` describes the area
of the interpolated rectangle. They coincide in their stretch *ratio* for
pure diagonal `F`, but are not in general equal to a particle's representative
physical area/volume. For shear/rotation even their stretch ratios differ.

## 2. Update order and consistency

The explicit scheme remains:

1. Evaluate GIMP interpolation with `(x_n, domain_size_n)`.
2. P2G mass/momentum, grid forces and nodal momentum update.
3. Kinematics and constitutive stress using the SAME old interpolation.
4. G2P particle velocity/position update using the SAME old interpolation.
5. **cpGIMP only:** calculate candidate `domain_size_(n+1)` using the updated
   cumulative `F_(n+1)`, then validate and commit.
6. Next timestep builds new interpolation from `(x_(n+1), domain_size_(n+1))`.

Updating `domain_size` *before* G2P would cause the existing strict
interpolation-staleness checks to reject the step and would mix old and new
shape functions in one time step. Therefore evolution happens AFTER G2P.

The solver's per-step atomic rollback now also snapshots `domain_size`. It
restores all particle/grid mutable arrays and leaves the clock unchanged if
an evolved domain is invalid.

## 3. API

```python
solver = ExplicitMPMSolver(
    particles, grid, LinearElastic(100.0, 0.25), dt=0.001,
    interpolation="cpgimp",        # Stage 10B, opt-in
    kinematics_update="exponential", # Stage 8, recommended
    stress_update="corotational",  # Stage 7, recommended for rotation
    advection_update="affine_exponential", # Stage 9, optional
    pic_fraction=0.0,
)
```

- `interpolation="mpm"`: unchanged standard point-MPM default.
- `interpolation="gimp"`: Stage 10A uGIMP, full domain side lengths fixed.
- `interpolation="cpgimp"`: Stage 10B classical axis-aligned domain lengths
  updated from `domain_size0 * diag(F)` at the end of each successful step.
- Standalone helpers: `candidate_cpgimp_domain(particles, grid, position=None)`
  (pure validation/calculation) and `update_cpgimp_domain(particles,grid)`
  (commit only if every particle passes validation).

### Initial state and FEM transfer

If `F=I` at MPM initialization, then `domain_size == domain_size0`, as normal.
If a future GeoPyFEM transfer explicitly imports `F != I`, the initialized
current `domain_size` **must equal** `domain_size0*diag(F)` and its meaning
must be compatible with the chosen MPM reference configuration. The cpGIMP
solver rejects inconsistent initial domains. An imported Cauchy stress alone
is not reason to set `F` away from identity.

## 4. Explicit limitations and safety behavior

- **Stencil:** current GIMP uses a fixed **4×4 candidate grid-node stencil**.
  Hence each *full* current side length must remain `0 < l_px <= dx` and
  `0 < l_py <= dy`. The routine **raises `ValueError`** if a stretched domain
  exceeds this bound. There is no silent clamp: using a truncated stencil
  would violate the partition-of-unity and gradient completeness invariants.
- **Fixed computational grid:** the entire current interpolation rectangle
  must remain inside the grid. No dynamic grid extension, periodic domain,
  domain clipping or special free-boundary treatment.
- **Axis-aligned approximation:** large shear and finite rotations are **not
  faithfully handled**. For simple shear `F=[[1,gamma],[0,1]]`, lengths remain
  unchanged. For 45-degree rigid rotation, both lengths shrink by `cos(45°)`
  even though `det(F)=1`: this classical cpGIMP formula is *not objective*.
  Large rotation can make an `F` diagonal non-positive; this is rejected.
  A corotational *stress* algorithm does not fix this domain geometry problem.
- **Not CPDI:** properly convecting domains as parallelograms and handling
  their full geometry is a separate method and future stage.
- **Numerical accuracy:** no claim that cpGIMP universally reduces stress
  error, Galilean-frame dependence or PIC/FLIP dissipation. These require
  their own benchmark studies, particularly under large deformation.
- **Physical model:** still plane-strain linear elasticity, no Mohr–Coulomb,
  contact, fluid coupling, or GeoPyFEM-to-MPM production transfer.

## 5. Verification and experiments

### 5.1 Domain kinematics and rollback tests

`tests/test_stage10b_cpgimp.py` verifies:

- Exact full-length updates under diagonal tension/compression.
- No accumulation of error through stepwise multiplying the previous width:
  lengths are always derived from *cumulative reference* `F`.
- Domain-average GIMP partition of unity, gradient sum zero, affine
  completeness after prescribed changes in lengths.
- Prescribed affine velocity gradient with exponential-map `F`, and
  **independent** `exp(a t)` / `exp(b t)` reference for domain and position.
- Exact rigid-body translation, no artificial deformation of domains.
- Explicitly demonstrated shear/rotation limitation.
- Invalid `F`, grid exit, excessive domain stretching, inconsistent initial
  `F`/domain, and complete solver rollback on invalid domain evolution.
- Unchanged default MPM/uGIMP behavior and rejection of unsupported modes.

### 5.2 Force-driven travelling-wave comparison

`validation/stage10b/validate_cpgimp.py` includes a force-driven elastic
wave from Stage 10A (linear d'Alembert stress reference in material labels).
Three interpolation choices (`mpm`, `gimp`, `cpgimp`) are exercised in both
rest and translated frames. Axial nodal velocities are **not prescribed**;
internal stress produces grid forces and accelerations. This dynamic check is
small strain, not a finite-strain exact solution.

For a smoke/short comparison with `ppc_x=2`, 120 steps, `dt=0.002 s` and
`drift=1 m/s`:

| Method | Final relative stress L2 | Cell crossings | Max x-domain change |
|---|---:|---:|---:|
| MPM | 0.111554 | 320 | 0 (no GIMP domains updated) |
| uGIMP | 0.129318 | 320 | 0 m |
| cpGIMP | 0.129286 | 320 | 0.000296 m |

These values indicate the cpGIMP pathway is active and the entire solver
remains finite and conservative for this small elastic wave. They do **not**
validate accuracy for large rotations, plastic runout, landslide impact, etc.
The slight change from uGIMP is expected since Gaussian wave strains are
small and cpGIMP changes the interpolation lengths only slightly.

With the **default benchmark** (`steps=160`, `ppc_x=2`, `dt=0.002 s`),
translated-frame (`drift=1 m/s`) values are:

| Method | Final relative stress L2 | Crossings | Max x-domain change |
|---|---:|---:|---:|
| MPM | 0.153162 | 480 | 0 |
| uGIMP | 0.172381 | 480 | 0 |
| cpGIMP | 0.172348 | 480 | 0.000387 m |

The exact diagonal-stretch benchmark achieves maximum full-length error
`1.67e-16 m` against the independent exponential solution. The small
positive/negative deviations in the dynamic stress errors should not be
interpreted as a robust method ranking.

### 5.3 Test status

- 17 new numerical cpGIMP tests passed (Stage 10B domain and solver behavior).
- 122 focused tests passed (Stage 10B, Stage 10A GIMP, explicit solver,
  Stage 8 exponential kinematics and Stage 9 advection).
- A repository-wide pytest attempt did **not finish** within the runtime
  limit; it is NOT being reported as fully passed. Run locally for closure.

## 6. Reproduction

From the root of an editable-installed GeoPyMPM repository:

```bash
python examples/12_cpgimp_deforming_domains.py
python -m validation.stage10b.validate_cpgimp
python -m pytest tests/test_stage10b_cpgimp.py tests/test_stage10b_validation.py -q
python -m pytest -q  # full regression, may take time
```

Stage 10B default outputs are in `validation/results/stage10b/`:

- `10B_affine_history.csv` and `10B_affine_domain.png`
- `10B_force_driven_history.csv` and `10B_force_driven_stress_error.png`
- `10B_summary.json`

## 7. Source changes

- **Added:** `src/geopympm/interpolation/cpgimp.py`
- **Added:** `src/geopympm/kernels/domain_update.py`
- **Changed:** `src/geopympm/solvers/explicit.py` (new opt-in mode, update
  AFTER G2P, rollback includes domain_size)
- **Changed:** `src/geopympm/interpolation/gimp.py` (optional candidate-domain
  validation parameter for safe precommit checks)
- **Changed:** `src/geopympm/interpolation/__init__.py` (exports)
- **Added:** `examples/12_cpgimp_deforming_domains.py`
- **Added:** `tests/test_stage10b_cpgimp.py`, `tests/test_stage10b_validation.py`
- **Added:** `validation/stage10b/validate_cpgimp.py` + validation outputs

## 8. Background references

- Bardenhagen, S. G. & Kober, E. M. (2004), *The Generalized Interpolation
  Material Point Method*, Computer Modeling in Engineering & Sciences, 5(6),
  477–496, DOI: https://doi.org/10.3970/cmes.2004.005.477 .
- Sołowski, W. T. et al. (2021), *Material Point Method: Overview and
  Advances*, GIMP method overview (section 2.4):
  https://publications.sci.utah.edu/publications/Sol2021b/CH0003%20pre-proof-without-videos.pdf
- iGIMP paper discussion of classical diagonal-`F` cpGIMP vs rotated/stretch
  domains: https://durham-repository.worktribe.com/OutputFile/1379506
