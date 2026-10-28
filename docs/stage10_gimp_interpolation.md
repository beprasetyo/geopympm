# Stage 10 — 2D uGIMP Interpolation and Solver Integration

## Overview

Stage 10A introduces **unchanged-domain generalized interpolation MPM (uGIMP)**
for 2D structured background grids, while keeping the Stage 1–9 **standard
Quad4 point MPM** default. The first implementation prioritizes numerical
correctness and backward compatibility. It does **not** implement CPDI,
deforming-domain GIMP, contact, soil plasticity, or 3D GIMP.

## Numerical definition

The full particle-domain side lengths are stored in `ParticleSet.domain_size[p]`
(`lp_x, lp_y`, not half-widths). These describe the rectangular interpolation
domain; **they are not particle volume** and need not obey `lp_x*lp_y = Vp`.

For the conventional 1D grid hat `Ni(x) = max(0, 1-|x-xi|/h)`, unchanged-domain
GIMP averages this hat over the particle's finite domain:

```
Sp,i(xp)    = (1/lp) * integral[xp-lp/2, xp+lp/2] Ni(x) dx
∂Sp,i/∂xp   = (Ni(xp+lp/2) - Ni(xp-lp/2))/lp
```

For a 2D axis-aligned rectangular domain, tensor products provide

```
Sp,ij       = Sx,i * Sy,j
∂x Sp,ij    = (∂x Sx,i) * Sy,j
∂y Sp,ij    = Sx,i * (∂y Sy,j)
```

The 1D integrals use exact piecewise-linear integration, split at nodal hat
breaks, rather than numerical quadrature. Numba compiles the 2D per-particle
loop. This smooths the gradient discontinuity at ordinary grid cell crossings
as long as the domains remain valid and have nonzero width.

## Node support and restrictions

For Stage 10, each full side length must be **strictly positive and numerically resolvable**, and
`lp_x <= dx`, `lp_y <= dy`. In this restricted range, the active support
has up to **3 nodes per axis** (up to 9 nodes in 2D), represented by a
**fixed 4×4 candidate stencil** (`node_ids.shape=(Np,16)`) for simple
Numba kernels and future extensions. Unused slots have **zero weight and
zero gradient**, with valid clamped node IDs; they do not participate.

The **whole particle domain** must remain within the fixed grid bounds.
A boundary violation raises `ValueError` and restores state if called from
the integrated solver. No extrapolation, domain clipping, periodic boundaries,
free-boundary special treatment, or grid extension is implemented.

`domain_size` remains unchanged under motion/deformation: this is explicitly
**uGIMP**. `deformation_gradient`, `volume`, and diagnostic strain still update
through Stage 8/3D and stress through Stage 7/4; they are distinct from GIMP
interpolation domain evolution. Deforming-domain GIMP or CPDI should be
introduced as a separate, independently tested follow-up.

## New API

```python
from geopympm.core import ParticleSet, BackgroundGrid
from geopympm.interpolation import evaluate_gimp
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver

particles = ParticleSet(
    position=[[1.2, 1.4]], mass=[10.0], volume=[0.25],
    domain_size=[[0.5, 0.5]],
)
grid = BackgroundGrid((0.0, 0.0), (1.0, 1.0), (4, 4))
interpolation = evaluate_gimp(particles, grid)
assert interpolation.node_ids.shape == (1, 16)

solver = ExplicitMPMSolver(
    particles, grid, LinearElastic(100.0, 0.25),
    dt=0.001, interpolation="gimp",
    pic_fraction=0.0, stress_update="corotational",
    kinematics_update="exponential", advection_update="affine_exponential",
)
solver.step()
```

Default `interpolation="mpm"` gives the original four-node Quad4 pathway.
Existing public substep functions (`particle_to_grid`, `update_grid_momentum`,
`update_particle_kinematics`, `grid_to_particle`) accept either interpolation
object. Performance kernels now loop over `node_ids.shape[1]`; the old
Quad4 function names are kept for backward compatibility.

## Verification and regression

```bash
python -m pytest tests/test_stage10_gimp.py -v
python examples/11_gimp_interpolation.py
python -m pytest -q
```

Stage 10 checks:

1. 2D partition of unity and zero sum of shape gradients.
2. Affine/linear reproduction of nodal coordinates and gradient completeness.
3. Exact GIMP gradients compared with finite differences.
4. Vanishing-domain limit approaching the standard Quad4 values.
5. Reduced force jump around a grid line for a single stressed particle.
6. Integrated rigid translation across PIC/FLIP, Euler/exponential F updates,
   and Euler/affine-exponential advection.
7. Detection of missing, oversized, and out-of-grid domains, plus stale
   `domain_size` and safe whole-step rollback.
8. Original Stage 1–9 regression tests retain default point-MPM behavior.

A smoother single-particle nodal-force trace **is not** sufficient to validate
energy conservation, long-duration dynamics, free-surface treatment, or
geotechnical large-deformation runout. A comprehensive Stage 10 benchmark
suite, including convergence and cost, remains a separate milestone.
