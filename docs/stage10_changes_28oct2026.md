# GeoPyMPM — Stage 10A implementation notes (28 October 2026)

**Source baseline:** `geopympm-main-update-oct27.zip` (uploaded 27 Oct 2026).

## Implemented

- New `interpolation/gimp.py` defines `GIMPInterpolation`, `evaluate_gimp`,
  full-domain validation, and stale/interpolation-integrity checks.
- New `kernels/interpolation_gimp.py` computes exact 1D hat averages and
  derivatives, tensor-product 2D shape functions, and padded 16-slot support.
- All existing P2G, nodal force, G2P, kinematics and affine-exponential
  advection numerical loops now honor dynamic `node_ids.shape[1]` support.
- High-level P2G / nodal update / G2P / kinematics accept either Quad4 or
  GIMP interpolation, while retaining old Quad4 validation/regression.
- `ExplicitMPMSolver(..., interpolation='gimp')` opts in to uGIMP.
  `interpolation='mpm'` preserves the original default.
- New example, 2D crossing diagnostic, Stage 10 docs and tests.

## Numerical acceptance

- `python -m pytest -q` -> **402 passed** on the development container.
- Cell-crossing diagnostic (single 0.2 m³ particle, 100 Pa sigma_xx,
  grid h=1 m; fixed y=1.5 m; x=0.75..1.25 m, 101 samples):
  maximum adjacent nodal-force change **20 N** (MPM) versus **0.2 N** (uGIMP).
- This is **not** proof of long-term energy conservation, numerical stability,
  3D support or landslide/pile validation.

## Deliberate limitations / Stage 10B candidates

- uGIMP keeps `domain_size` fixed during deformation; its interpolation
  rectangle does not rotate/shear or follow the full deformation gradient.
- Restrict full side lengths to `(0,h]` per dimension; supports are padded
  to 16 nodes but only up to 9 carry positive weight.
- Every particle domain, including half widths, must stay inside the grid.
- Strict interpolation re-evaluation occurs at each substep for integrity;
  profile and optimize after physical correctness is established.
- Future milestone: domain evolution or CPDI, multi-cell domain support,
  conservation and convergence studies, free-surface handling, geomechanics.

## Run locally

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python examples/11_gimp_interpolation.py
python -m validation.stage10.run_crossing
```

No Git commit or push was executed in the development container.
