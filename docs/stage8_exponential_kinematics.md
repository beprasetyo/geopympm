# GeoPyMPM Stage 8 — Exponential-Map Particle Kinematics (2D Plane Strain)

## Scope and motivation
Stage 7 implemented a corotational *Cauchy stress* update, but Stage 3D still
updated particle deformation gradient with forward Euler:

\[F_{n+1}=(I+\Delta t L_p)F_n.\]

For rigid spin \(L=W=\begin{bmatrix}0&-\omega\\\omega&0\end{bmatrix}\), the
exact volume change is zero. Yet the Euler update has incremental Jacobian
\(1+(\omega\Delta t)^2\), which accumulates spurious material volume.

Stage 8 adds a **switchable exponential-map update**:

\[\boxed{F_{n+1}=\exp(\Delta t L_p)F_n}\]

with the spatial velocity gradient \(L_p=\sum_i v_i\otimes\nabla N_{ip}\)
from the Stage-3B updated grid and Stage-2 Quad4 interpolation at \(x_p^n\).
This assumes L is **frozen during each substep**; it is exact for constant L,
not an exact general MPM trajectory integrator.

For 2-D plane strain, \(F\) is stored as (n,2,2), with out-of-plane stretch
fixed at one. Since \(\det\exp(B)=\exp(\mathrm{tr}(B))\), the exact constant-L
update satisfies

\[J_{n+1}=J_n\exp(\Delta t\mathrm{tr}(L_p)),\qquad V_{n+1}=V_{p0}J_{n+1}.\]

Under rigid spin, \(L_p=W\), \(\mathrm{tr}(W)=0\), so \(J\) remains one to
floating-point accuracy. Compression/dilation produces its expected positive
Jacobian. Non-finite output and non-positive J are checked before any state is
committed.

## Implementation details

- `src/geopympm/kernels/kinematics_exponential.py`: independent Numba
  numerical kernel, with stable closed-form real 2×2 exponential using
  traceless decomposition and a Taylor branch for near-zero discriminant.
  No SciPy runtime dependency is introduced.
- `src/geopympm/solvers/kinematics.py`: adds keyword-only
  `kinematics_update='euler' | 'exponential'`, shares the existing full Quad4
  interpolation/grid mass checks and atomic update.
- `src/geopympm/solvers/explicit.py`: adds keyword-only `kinematics_update`
  passed through Stage 3D. **Default stays `euler`** so Stage 6 baseline
  verification numbers and old user code do not change.
- `validation/stage7/objective_rotation.py`: exposes an optional
  `kinematics_update` argument (default Euler), enabling exact like-for-like
  comparison without duplicating the benchmark.
- `validation/stage8/finite_rotation.py`, `run_kinematics.py`: a rigid spin
  benchmark (16 particles, 0.3 rad total rotation) with Stage 7
  `stress_update='corotational'`, compares Euler against exponential-map F.
- `tests/test_stage8_exponential_kinematics.py`: unit tests for rigid spin,
  isotropic dilation, pure shear, general affine gradients vs independent
  Taylor-series matrix exponential, multi-particle consistency,
  errors/atomicity/solver rollback and regression defaults.

## Usage

```python
solver = ExplicitMPMSolver(
    particles, grid, LinearElastic(50_000.0, 0.25),
    dt=0.001,
    pic_fraction=0.0,
    stress_update='corotational',
    kinematics_update='exponential',
)
solver.run(100)
```

Standalone kinematics:

```python
result = update_particle_kinematics(
    particles, grid, interpolation, dt,
    kinematics_update='exponential',
)
```

The Stage-3D call must still be **after** Stage-3B grid momentum update and
**before** Stage-3C particle position update. Nothing in the code changes the
USL-like cycle, grid/mass transfer, or constitutive laws.

## Verification

Run:

```bash
python -m pytest -q tests/test_stage8_exponential_kinematics.py
python -m validation.stage8.run_kinematics --plots
```

For \(\omega=3\) rad/s, \(dt=0.001\) s, and 100 steps:

| Metric | Euler F | Exponential-map F |
| --- | ---: | ---: |
| Final rigid-spin Jacobian | 1.000900401 | ~1.000000000000000 |
| Maximum `|J-1|` | ~9.004e-4 | ~1.3e-15 |
| Objective Cauchy stress error | ~8e-13 Pa | ~8e-13 Pa |
| End particle-position error against continuum rotation | ~2.0e-4 m | **unchanged** |

The reference solution is the exact rotation matrix \(R(\omega t)\). The
benchmark imposes nodal velocity; it does **not** test a free rigid body
subject to contact and internal forces.

## Limitations: what this stage does *not* fix

1. **Particle position update remains Euler (Stage 3C).** For the prescribed
   rigid spin test, the particle trajectories drift from exact rotation, even
   though F and V are now correct. Improving this requires a separately
   verified particle-advection method.
2. `particles.strain` still accumulates `dt * sym(L)` engineering-Voigt
   increments. It is a small-increment *diagnostic*, not an objective
   finite-strain material history. Do not interpret large accumulated values
   as Green–Lagrange or logarithmic strain.
3. Stage-7 corotational **Jaumann hypoelasticity is not hyperelasticity**;
   its finite-strain response is model-dependent. Plasticity, contact,
   damping, pore pressure and GIMP are not included.
4. The exponential of frozen L does not in general solve time-varying
   gradients exactly or guarantee nonlinear/impact numerical stability.
5. The solver uses the existing wave-speed/advection CFL screening estimate;
   a positive Jacobian alone does not establish accurate integration.

## Next tasks

Validate an improved particle advection scheme for rotational kinematics;
subsequently add GIMP (including updated particle domains) and generalized
constitutive/contact interfaces for slope runout **and** pile penetration.
