# GeoPyMPM Stage 6D — Force-driven elastic standing wave

## Purpose and scope

Stage 6A–6C verified uniform translation, constant-acceleration free fall,
and stress/strain updates under *prescribed* affine nodal velocities. Stage 6D
is the first **force-driven elastic-motion** verification: initial elastic
stresses create internal nodal forces, which generate accelerations and particle
motion. It uses the unchanged Stage 5 ExplicitMPMSolver, standard Quad4 MPM,
plane-strain LinearElastic, and the Stage 5 USL-like update.

**This is analytical numerical verification, NOT experimental validation or
large-deformation runout validation.** No solver, particle/grid container, or
Numba kernel files are modified by the Stage 6D patch.

## Physical problem and analytical solution

A uniform 2D strip with unit out-of-plane thickness:

- Length **L = 4 m**, height **H = 0.5 m**, mass density **rho = 1200 kg/m³**.
- Isotropic linear elastic **E = 120,000 Pa**, **nu = 0.25**.
- Each cell has four particles; baseline grid has **nx = 32**, **ny = 2**.
- All nodal **vertical velocities** are fixed to zero (uniaxial *strain*,
  epsilon_yy = epsilon_zz = 0). Only the two **end faces** have fixed x velocity.
  **Interior x velocities are never prescribed**; they follow from internal
  elastic forces. There is no gravity or externally applied force.
- Start from the *initially deformed* configuration, with the first longitudinal
  standing wave mode and the consistent initial small strain and stress:

  `u_x(X,0) = A sin(pi X/L)`, with **A = 0.002 m**, and `v_x(X,0) = 0`.

The stiffness for this laterally constrained 2D plane-strain longitudinal wave is

```
G   = E/[2(1+nu)]
lambda = E*nu/[(1+nu)(1-2nu)]
M = lambda + 2G = 144000 Pa
c = sqrt(M/rho) = 10.95445115 m/s
k = pi/L
omega = k*c
T = 2*pi/omega = 2L/c = 0.7302967433 s
```

**Do not use `sqrt(E/rho)` here**: that is the one-dimensional uniaxial-*stress*
bar idealization, not this transverse-strain-constrained plane-strain test.

Small-strain analytical solutions (reference/material coordinate X):

```
u_x(X,t)      = A sin(kX) cos(omega t)
v_x(X,t)      = -A omega sin(kX) sin(omega t)
epsilon_xx    = A k cos(kX) cos(omega t)
sigma_xx     = M epsilon_xx
sigma_yy,zz  = lambda epsilon_xx
```

Stress is tension-positive. Shear is zero for this ideal mode.

## Correct initialization

We retain undeformed `X` separately to evaluate the analytical solution.
ParticleSet is initialized at `x(X,0) = X + u(X,0)`, and then initialized
consistently with

```
strain[:,0]       = A*k*cos(kX)
stress[:,0]       = M*strain[:,0]
stress[:,1:3]     = lambda*strain[:,0]
F[:,0,0]          = 1 + strain[:,0]
volume            = volume0 * (1 + strain[:,0])
mass              = rho * volume0
```

This is a small-amplitude approximation consistent with the incremental
Stage 3D/4 formulation. `ParticleSet.position0` is the initial *deformed*
coordinate and should not be used as the reference `X` for the analytical wave.

## Numerics and pass criteria

- Baseline: **FLIP** `pic_fraction=0`, `dt=0.0005 s`, `t_end=0.6 s` = 1200 steps.
- `solver.step()` recalculates Quad4 interpolation, P2G, internal grid forces,
  momentum, kinematics, elastic stress, and G2P every step.
- Error: L2 norm of particle displacement, x velocity, and sigma_xx minus the
  analytical continuous-wave field, each normalized by **the L2 norm of its
  fixed initial-mode scale**. In particular, the denominator does *not* go to
  zero at a quarter-cycle zero crossing.
- The numerical period is measured from two interpolated successive zero
  crossings of the displacement mode; this is meaningful only if at least
  two crossings occur. The comparison time of 0.6 s covers both crossings.
- Energy diagnostic: `E = Σ 1/2 m_p |v_p|² + Σ 1/2 M ε_xx,p² V_p,0`.
  This is a **small-strain diagnostic**, NOT an exact discrete MPM energy identity;
  independent quadrature errors and transfer/integration may alter it.

Baseline pass limits (engineering regression tolerances rather than exactness):

| Metric | Limit |
| --- | ---: |
| Maximum normalized displacement L2 error | 0.03 |
| Maximum normalized velocity L2 error | 0.04 |
| Maximum normalized sigma_xx L2 error | 0.08 |
| Relative measured period error | 0.015 |
| Maximum absolute relative energy drift | 0.12 |
| Relative total mass error | 1e-12 |
| Internal force magnitude | > 0 |
| Minimum Jacobian | > 0 |
| At least two modal zero crossings | required |

The **PIC** `pic_fraction=1` comparison uses the *same exact solution*, but is
expected to have large numerical damping and therefore **fail the tight
undamped-wave FLIP acceptance limits**. This is not an MPM implementation defect:
comparison PIC is **informational and excluded from the Stage 6D suite PASS**.

## Run and generated outputs

From the repo root, with editable `geopympm` installed:

```bash
python -m pytest tests/ -v
python -m validation.stage6.run_wave --plots --compare-pic --refinement
```

Default outputs go to `validation/results/stage6d/`:

- `6D_summary.json` — FLIP metrics and PASS/FAIL, optional PIC, optional refinement.
- `6D_FLIP_history.csv` and `6D_FLIP_profiles.csv` — wave-mode time series and
  selected particle-position displacement/stress profiles.
- Optional `6D_PIC_history.csv` and `6D_PIC_profiles.csv`.
- Optional `6D_joint_refinement.csv` (nx=16,32,64 and dt=0.001,0.0005,0.00025).
- PNG plots (when `--plots`): modal displacement, modal velocity, energy,
  stress error, spatial displacement; optional joint refinement.

The refinement changes **both h and dt** (keeping their ratio fixed), so it
supports a joint grid/time resolution improvement claim, **not a pure spatial
convergence order**. The number of particles per cell remains four.

## Baseline results on reference implementation

With FLIP, `nx=32`, `dt=0.0005`, `t_end=0.6`:

| Observable | Result |
| --- | ---: |
| Analytical period | 0.73029674 s |
| Measured period | 0.73093746 s |
| Relative period error | 0.0008773 |
| Max normalized displacement L2 error | 0.0040898 |
| Max normalized velocity L2 error | 0.0131569 |
| Max normalized sigma_xx L2 error | 0.0490879 |
| Max relative energy drift | 0.0021861 |
| Final/initial energy ratio | 1.0016193 |
| Peak internal x force | 56.63 N |

For PIC at the same resolution and t_end, the final/initial energy ratio is
about **0.109** and the displacement L2 error reaches ~**0.495**. This
quantifies substantial algorithmic numerical dissipation with PIC, rather than
asserting either method universally superior (PIC may be more robust in other
problems). Joint refinement reduces maximum stress L2 error approximately
0.0982 -> 0.0491 -> 0.0245 for nx=16 -> 32 -> 64.

## Current limitations and next work

- No automatic surface traction or contact friction: the simple fixed-end and
  transversely constrained velocity BCs use the existing grid velocity mask.
- Small-strain incremental stress law is not objective under large rotations.
- Only one longitudinal first mode with uniform isotropic elasticity is tested;
  numerical dispersion and complex 2D wave problems are not fully covered.
- Nodal grid is reset every step; higher-order transfer/FLIP vs PIC affects
  numerical damping and energy balance.
- This benchmark is NOT physical landslide runout validation, a full elastic
  stress-equilibrium patch test, or proof of stability for arbitrary dt.

Recommended next: 6E nonzero-stress cell crossing and 6F energy/convergence,
then objective stress update and plasticity.
