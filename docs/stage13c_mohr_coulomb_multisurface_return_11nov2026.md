# GeoPyMPM Stage 13C — Mohr–Coulomb multisurface face, edge & apex return

**Baseline:** Stage 13B, after applying Stages 10G-D1–D4, 11, 12, 13A, 13B.  
**Deliverable:** incremental patch; only modified/new files.  
**Date:** 11 November 2026.

## Scope and limitations

This stage replaces the Stage-13B **one-face** Mohr–Coulomb stress update
with a full six-plane multisurface *active-set search in principal-stress space*.
The plane-strain small-increment model retains the exact conventions and
constitutive histories established in Stage 13B:

- Stress (Pa), **tension positive**, Voigt `[xx,yy,zz,xy,yz,xz]`.
- Strain increments and accumulated plastic strain use **engineering shear**.
- Plane strain means total `Δε_zz = Δγ_yz = Δγ_xz = 0`, but
  `σ_zz` and `ε^p_zz` **need not be zero**.
- Perfect plasticity and non-associated flow allowed with `0 ≤ ψ ≤ φ`.
- Material history unchanged: `plastic_strain (n,6)` and
  `equivalent_plastic_strain (n,)`, so existing state buffers and rollback
  still work.
- Explicit solver supports USL-like, USF, MUSL with MPM, uGIMP, cpGIMP, CPDI1.

**Still excluded:** finite-strain objective Mohr–Coulomb integration;
large-rotation stress objectivity; tensile cutoff; hardening/softening;
pore-pressure/effective-stress coupling; regularized strain localization;
initial geostatic equilibrium; unsaturated physics; 3D MPM; all frictional
contact/soil–structure models. Stage 13C does not validate geotechnical runout.

## Mathematical formulation

For principal stresses `s = (σ_0, σ_1, σ_2)` in any (consistent) order,
the **six** Mohr–Coulomb planes are indexed by ordered pairs `(i,j)`
where `i ≠ j`:

\[
f_{ij}(s) = (1+\sin\phi) s_i - (1-\sin\phi) s_j - 2c\cos\phi \leq 0.
\]

The corresponding non-associated potential branch is

\[
g_{ij}(s) = (1+\sin\psi) s_i - (1-\sin\psi) s_j.
\]

Thus, `n_f=∂f/∂s` and `n_g=∂g/∂s` are constant for each plane. Elastic
trial stresses use the isotropic elastic stiffness
`C = λ 1⊗1 + 2G I` in principal stress space. For an active set `A`:

\[
\begin{aligned}
\Delta\gamma_A &:\quad
\sum_{b\in A}(n_{f_a}^\mathrm{T}C n_{g_b})\Delta\gamma_b
= f_a(s^{trial}),\quad a\in A,\\
s^{n+1} &= s^{trial}-C\sum_{b\in A}\Delta\gamma_b n_{g_b},\\
\Delta\varepsilon^p &= \sum_{b\in A}\Delta\gamma_b n_{g_b}.
\end{aligned}
\]

The implementation enumerates **independent active sets of size one, two,
and three**, solves each small linear system using pivoted Gaussian elimination,
and accepts only candidates satisfying all six final inequalities,
active-plane equality, and nonnegative plastic multipliers (within numerical
tolerances). If the system is singular, infeasible, or outside tolerance,
it tries another set, and ultimately raises without committing any
particle state. One, two, and three independent constraints represent
smooth-face, edge, and apex returns. Multiple physically active planes
may be redundant at the apex.

The stress return assumes coaxial principal directions with the elastic trial
stress. In plane strain, the in-plane 2×2 eigenvalues/eigenvectors are
computed analytically, so **the Numba MC kernel no longer calls
`np.linalg.eigh` and does not require SciPy/LAPACK**.
The out-of-plane principal stress is included when forming the six planes.
Engineering shear plastic strains and the scalar equivalent plastic strain
are reconstructed consistently in the fixed coordinates.

**Important non-associated apex qualification.** For hydrostatic stress
above `c cot(φ)` and `ψ=0`, each plastic potential gradient is trace-free.
Plastic correction cannot change mean stress, so no admissible
multisurface correction exists. Rather than silently substitute a fictitious
dilatant return, GeoPyMPM raises `ValueError` and retains the last committed
state. A proper tensile cutoff, viscoplastic regularization, or physically
justified alternative plastic potential is a **separate constitutive choice**.
At `φ=0` the hydrostatic apex formula is inapplicable (Tresca-type limit).

## Checkpoint and restart — schema 6

Stage 13C retains the same numerical state inventory as Stage 13B but changes
the constitutive return algorithm. New Mohr–Coulomb checkpoints therefore
use **schema 6** and record:

```
"constitutive_algorithm": "mohr_coulomb_multisurface_active_set_v1"
```

The loader validates that exact ID. Invalid or absent metadata is rejected.
Stage 13B schema-5 checkpoints remain readable and restore all numerical
state, but **continuing a schema-5 run with Stage 13C can change results** if
a later increment enters a formerly unsupported corner/apex. Preserve the
original executable source and input metadata for exact historical
reproduction. Writing a checkpoint after loading legacy schema 5 upgrades
the format to schema 6 without modifying old checkpoint files.

Schemas 1–4 remain accepted. Existing Linear Elastic and CNH checkpoints
are not changed by Stage 13C.

## Validation and regression

- Pure elastic stress increment and original Stage-13B smooth-face response.
- Deterministic face, edge, apex return with yield residual ≪ 1 Pa at 1000 Pa E.
- Hydrostatic apex with nonzero dilation and incompatibility with `ψ=0`.
- In-plane rotated loading path, principal-stress consistency, and history.
- Atomic failure on one particle in a multi-particle batch; timestep rollback.
- Solver-driven edge/apex in **2 branch types × 3 schemes × 4 interpolation
  methods = 24 full explicit-solver tests**, using prescribed affine nodal BC.
- Checkpoint schema-6 material/history round-trip and repeated continuation;
  corrupt return-algorithm metadata rejected; legacy schema-5 read.
- All retained Stage-13B tests must pass (one test expects schema 6 instead
  of schema 5; old `ψ=0` inadmissible apex test remains valid).

**Numerical validation still required at Stage 13D:** independent
triaxial and plane-strain stress paths, mesh and timestep refinement,
non-associated flow stability, boundary-value tests, benchmark against
GeoPyFEM or an independent MC return integrator. A tiny yield residual is a
necessary local consistency check, **not** proof of general accuracy.

## Installation

From root of a repository with Stage 13B applied:

```bash
unzip -o ~/Downloads/GeoPyMPM_Stage13C_Mohr_Coulomb_Multisurface_patch_11Nov2026.zip -d .
source .venv/bin/activate
python -m pytest -q tests/test_stage13b_mohr_coulomb.py tests/test_stage13c_mc_multisurface.py
python examples/28_mohr_coulomb_multisurface_material_point.py
```

For broader regressions, also run your project's full `python -m pytest -q`
and keep the exact environment, hardware, and test output in the project log.
The Stage-13B SciPy dependency patch may remain installed; the Stage-13C MC
kernel itself no longer requires SciPy.

## Files

- `src/geopympm/materials/mohr_coulomb.py` — multisurface stress integrator
- `src/geopympm/io/checkpoint.py` — new schema-6 algorithm metadata
- `src/geopympm/io/restart.py` — schema-6 loader compatibility
- `tests/test_stage13b_mohr_coulomb.py` — schema expectation update
- `tests/test_stage13c_mc_multisurface.py` — new local/integration tests
- `examples/28_mohr_coulomb_multisurface_material_point.py` — runnable demo
- `docs/stage13c_mohr_coulomb_multisurface_return_11nov2026.md` — this document
