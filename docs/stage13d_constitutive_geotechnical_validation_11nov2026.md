# GeoPyMPM Stage 13D — Constitutive & Geotechnical Verification (11 November 2026)

## Baseline and scope

**Apply after Stage 13C.** This is a validation-only incremental patch: nothing under
`src/geopympm/` is modified. The Stage 13C Mohr–Coulomb multisurface
stress-integrator remains unchanged, including its documented handling of
inadmissible nonassociated apex returns. All stress quantities use Pa and
**tension-positive** Cauchy stress `[xx, yy, zz, xy, yz, xz]`. In-plane
strain uses **engineering shear** `gamma_xy`.

This stage tests the physical response, rather than merely checking that
yield residuals are small. It separates **analytical references** from
**comparisons against a separately integrated material point**. Both are
valuable, but they provide different levels of independence.

### Reproduction

At the repository root, in the project's virtual environment:

```bash
python -m pytest -q tests/test_stage13d_validation.py
python examples/29_mohr_coulomb_stage13d_validation.py
# Optional CSV + plots (requires optional matplotlib):
python -m validation.stage13d.run --output-dir results/stage13d --plot
```

Output files are:

- `pure_shear.csv`: 80-step constitutive response and exact monotonic solution
- `confined_compression.csv`: compression, `sigma_xx`, `sigma_yy`, `sigma_zz`, yield residual, accumulated plastic strain
- `triaxial_yield_envelope.csv`: independently calculated confinement–strength points and solver yield residuals
- `affine_shear_mpm.csv`: 12 scheme/interpolation MPM combinations compared with the exact pure-shear response
- `affine_confined_compression_mpm.csv`: 12 scheme/interpolation combinations compared with elastic first-yield onset and a separately integrated material point
- `summary.json`: pass/fail criteria, scalar errors, scope, and analytical values
- Three optional PNG plots under `--plot`

All runs are deterministic for the same source code/Numba/NumPy environment.

## Benchmark 1 — Exact pure shear, zero-friction limit

For *perfectly plastic Mohr–Coulomb* with `phi = psi = 0`, the model reduces
to the pressure-insensitive Tresca yield condition. In virgin monotonic
simple shear, no normal stresses are applied and

\[
G=\frac{E}{2(1+\nu)},\qquad
\tau(\gamma)=\operatorname{sgn}(\gamma)\min(G|\gamma|,c).
\]

For engineering plastic shear `gamma_p` and equivalent plastic strain
(using the deviatoric J2 strain invariant of the Stage 13C history definition):

\[
\gamma_y=\frac{c}{G},\qquad
\gamma_p=\gamma-\frac{\tau}{G},\qquad
\bar\varepsilon^p=\frac{|\gamma_p|}{\sqrt{3}}.
\]

The exact energy input in monotonic shear after yield is

\[
W=\int_0^\gamma \tau\,d\gamma = c\gamma-\frac{c^2}{2G},
\quad W_e=\frac{c^2}{2G},\quad W_p=c\left(\gamma-\frac{c}{G}\right).
\]

**Reference configuration:** `E=1000 Pa`, `nu=0.3`, `c=10 Pa`,
`phi=psi=0`, final `gamma=0.08`. Then `G=384.6153846 Pa`,
`gamma_y=0.026`, `tau_final=10 Pa`, `gamma_plastic=0.054`,
`eqp=0.031176914536`. We test one-step and split-increment responses,
positive/negative loading, elastic unloading/reloading and energy balance.

This is an analytical validation: expected stresses/history/work are not
computed by invoking GeoPyMPM's return-mapping routine.

## Benchmark 2 — Independent Mohr–Coulomb triaxial yield envelope

Under triaxial *compressive stress magnitudes*, with confining pressure
`σ3c >= 0`, the peak deviatoric stress at yield is

\[
q_f=\frac{2\big(c\cos\phi+\sigma_{3c}\sin\phi\big)}{1-\sin\phi}.
\]

The independently computed `q_f` is mapped to GeoPyMPM stress convention
as `sigma = [-sigma3c-q_f, -sigma3c, -sigma3c, 0,0,0]`. The solver's
three-dimensional principal-stress yield function should return zero at
`q=q_f`, negative before, and positive after. The test spans different
cohesion/friction and confinement combinations.

**Caveat:** This is a *stress-space strength-envelope check*. The current
GeoPyMPM solver is **2D plane strain** and cannot reproduce a complete
axisymmetric triaxial laboratory boundary-value test simply by assigning
these principal stresses. Do not label this as a simulated triaxial test.

## Benchmark 3 — Plane-strain laterally confined compression

For virgin isotropic zero stress, `epsilon_xx=-e`,
`epsilon_yy=epsilon_zz=0` before first yield:

\[
\sigma_{xx}=-(\lambda+2G)e,\quad
\sigma_{yy}=\sigma_{zz}=-\lambda e.
\]

Hence, if `G-(lambda+G) sin(phi) > 0`, the **analytical first-yield
compressive axial strain** is

\[
e_y=\frac{c\cos\phi}{G-(\lambda+G)\sin\phi}.
\]

Reference input: `E=1000 Pa`, `nu=0.2`, `c=10 Pa`,
`phi=10 degrees`, `psi=0` gives `e_y ≈ 0.0332618061`. Test checks no
plastic flow before this strain and plastic response after crossing. The
first detected plastic sample may be within **one prescribed increment**
after the analytical onset; this is the expected discretization tolerance.

A separate post-yield material-point trajectory is compared to the
MPM uniform-strain boundary-value results. That latter comparison tests
**consistency** of interpolation/solver/constitutive dispatch; the entire
post-yield curve is *not* an independent analytical solution.

## Benchmark 4 — Two affine MPM kinematic patch tests

The background grid has `10x10` cells with `h=0.1 m`; four particles are
placed inside the interior. **Velocities are imposed at every grid node**
(no gravity; no free surface; no complex contact):

- Shear: `v_x = 20*(y-0.5) m/s, v_y=0` for 40 steps of `dt=1e-4 s`,
  leading to engineering `gamma=0.08`. Stress and plastic-strain state
  are compared to the independent Tresca limit above.
- Confined compression: `v_x = -20*(x-0.5) m/s, v_y=0` for 25 steps,
  leading to axial compression `e=0.05`. The observed first-yield step
  is checked against the independent closed-form `e_y`, while the final
  post-yield stress/history is checked against standalone material-point
  integration under the same strain path.

Each test runs **3 explicit schemes** (`usl`, `usf`, `musl`) multiplied by
**4 interpolation types** (`mpm`, `gimp`, `cpgimp`, `cpdi`), yielding
12 combinations per kinematic problem and 24 MPM combinations in all.

These are **uniform affine deformation patch tests**. Fully prescribing
all grid node velocities suppresses physical wave propagation and is not
an independently loaded experiment. They do *not* demonstrate accurate
free-boundary soil compression, contact, landslide stability or runout.

## Checkpoint/restart under plastic loading

A split `20+20` step MPM/MUSL and CPDI1/MUSL shear run is compared
against the corresponding uninterrupted `40`-step run. Recovered stresses,
particle positions, deformation gradients, and constitutive histories must
match exactly for these deterministic configurations. This additionally
regresses Stage 13C checkpoint schema 6.

## Acceptance criteria and interpretation

Analytical checks have absolute stress tolerances usually below `1e-7 Pa`
on this low-stress benchmark and absolute plastic-strain tolerances below
`1e-8`. The accepted first-yield onset is one prescribed strain increment
of the analytical value. These tolerances are tailored to this setup, not
universal solver tolerances for high-stiffness soils. See `summary.json`.

**Test status should be reported by scope.** Passing these tests means:

1. The perfectly-plastic frictionless shear response matches closed form.
2. MC stress-space triaxial compression strength is correctly evaluated.
3. Plane-strain compression first-yield onset is consistent with elasticity
   and the MC surface.
4. Under prescribed affine nodal velocities, all four interpolations and
   three time schemes reproduce the same small-increment material response.
5. Plastic history survives checkpoint/restart for the tested cases.

It **does not yet validate**:

- A free boundary-value geotechnical loading experiment against measured
  data or independently solved GeoPyFEM results (no GeoPyFEM run included);
- General Lode-angle stress paths or *all* six-plane active-set degeneracies;
- Large-rotation objective plastic stress integration (Stage 13C remains a
  small-increment, **non-objective** formulation);
- Perfectly plastic strain localization mesh-dependence, loss of
  hyperbolicity, tensile cutoff, hardening/softening or pore-water effects;
- Mohr–Coulomb apex hydrostatic excess tension for `psi=0`, where the
  current nonassociated flow has no admissible return;
- General 3D, physical contact, gravity equilibrated slope stability,
  runout, or quantitative agreement with PLAXIS / Uintah / GeoPyFEM.

### Recommended next validation (not claimed completed)

An independently specified *dry plane-strain compression / biaxial* test
with measured or GeoPyFEM reference curves, followed by a shallow footing
or undrained (appropriate model) bearing-capacity benchmark with systematic
mesh, timestep, particle-density and interpolation convergence. Then add
objective plastic integration before large-rotation runout claims.
