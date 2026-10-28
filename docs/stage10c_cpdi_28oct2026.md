# GeoPyMPM Stage 10C — 2D CPDI1 (28 October 2026)

## Status

**Implemented and verified for affine 2D motion**; force-driven small-strain test is a **diagnostic** rather than a large-deformation validation. The previous Stage 10A/10B modes are unchanged by default. `interpolation="cpdi"` is **opt-in**.

## Purpose and references

This implements the **first-order convected particle domain interpolation (CPDI1)** of Sadeghirad, Brannon & Burghardt (2011), *A convected particle domain interpolation technique to extend applicability of the material point method for problems involving massive deformations*, IJNME 86, 1435–1456, doi:10.1002/nme.3110. This is different from cpGIMP: CPDI1 retains the **full deformation-gradient tensor** and tracks parallelogram particle domains, including in-plane spin and simple shear. This is not CPDI2, higher-order CPDI, a contact formulation, or a final large-runout validation.

## State conventions

- `ParticleSet.position`: current particle **center**, shape `(n,2)`.
- `ParticleSet.domain_size0`: **initial full side lengths** `(l_x0,l_y0)`; required positive.
- `ParticleSet.deformation_gradient`: **current full 2×2 F** from Stage 8, relative to the current MPM reference.
- `ParticleSet.domain_size` in **CPDI mode only** is the **axis-aligned full bounding-box width** around the convected parallelogram; used for diagnostics and state consistency, **not** for CPDI shape function integration. uGIMP/cpGIMP retain their original distinct meanings of this field.
- Physical `ParticleSet.volume` remains the solver's updated continuum volume (`V0 det F`); the CPDI shape integrates over a *geometric* particle domain. Geometric domain area and physical `volume` are not implicitly forced equal; initial sampling and transfer must make them consistent where the formulation needs physical domain coverage.

## Formulation

Define particle half-edge vectors

\[
\mathbf a_p=\mathbf F_p\begin{bmatrix}l_{x0}/2\\0\end{bmatrix},\qquad
\mathbf b_p=\mathbf F_p\begin{bmatrix}0\\l_{y0}/2\end{bmatrix}.
\]

With center `x_p`, corners are ordered `(-a-b), (+a-b), (+a+b), (-a+b)` plus `x_p`. Their oriented geometric area is `A_p = 4 (a_x b_y - a_y b_x)`; it must be positive and nondegenerate.

For each background-grid node `i`, evaluate the *standard Quad4 grid* shape `N_i` at each corner `x_{p,c}`. The CPDI1 shape **weight** is

\[
S_{ip}^{CPDI1} = \frac14 \sum_{c=1}^{4} N_i(\mathbf x_{p,c}).
\]

Let `(\xi,\eta)` denote local particle domain coordinates. Mapping: `x = x_p + a xi + b eta` on `[-1,1]^2`. For each corner `c` with signs `(s_x,s_y)` equal to `(-,-),(+,-),(+,+),(-,+)`, the **domain-averaged gradient of that corner's Q4 shape** is

\[
\overline{\nabla Q_c}=\tfrac14(s_x\nabla\xi+s_y\nabla\eta),\quad
\nabla\xi=\frac{1}{\mathbf a\times\mathbf b}(b_y,-b_x),\quad
\nabla\eta=\frac{1}{\mathbf a\times\mathbf b}(-a_y,a_x).
\]

Then

\[
\nabla S_{ip}^{CPDI1}= \sum_{c=1}^{4} N_i(\mathbf x_{p,c})\,\overline{\nabla Q_c}.
\]

**Important:** this gradient is *not* an average of grid-node shape gradients at corners. For affine nodal fields it satisfies `sum S=1`, `sum grad S=0`, `sum_i S_i x_i = x_p`, and `sum_i x_i \otimes grad S_i = I` (up to roundoff). Four grid nodes from each of the four particle corners produce **16 fixed contribution slots** even if nodes repeat; repeated contributions sum naturally in the existing Numba P2G/G2P kernels. CPDI1 has **no `domain_size <= cell_size` cap**, although the 4 corners must remain within the **fixed grid bounds**.

## Implementation

| Source | Role |
|---|---|
| `src/geopympm/kernels/interpolation_cpdi.py` | Numba kernel for corner grid weights and averaged particle-Q4 gradients |
| `src/geopympm/interpolation/cpdi.py` | Validate geometric support, evaluate weights/gradients, update bounding-box field, stale-state checks |
| `src/geopympm/interpolation/gimp.py` | Common 16-slot stale-state dispatcher for CPDI, keeping uGIMP behavior |
| `src/geopympm/solvers/g2p.py` | After Stage 8 changed F, validate against **old** F interpolation snapshot; check next domain corners at new position |
| `src/geopympm/solvers/explicit.py` | Add opt-in `interpolation='cpdi'`, update CPDI bounding box **after** G2P; step rollback on invalid state |
| `examples/13_cpdi_convected_domains.py` | Standalone shear demonstration |
| `tests/test_stage10c_cpdi.py` | Unit tests including spin, shear, >cell domain, partition, affine reproduction, rollback |
| `validation/stage10c/validate_cpdi.py` | Prescribed affine verification and force-driven wave comparison |

For CPDI the state sequence is:

1. At time `n`, evaluate `CPDI1` weights from `(x_n, F_n, domain_size0)`.
2. Map particle state to grid; calculate forces, grid update; update particle kinematics to `F_{n+1}` and stress.
3. **Still use the old `CPDI1` interpolation** to perform G2P and update `x_{n+1}`. The solver explicitly opts into the current-step `F` change in the old-interpolation validator. The same validator would reject a changed `F` outside that documented orchestration.
4. Validate new domain corners from `(x_{n+1}, F_{n+1})` and store their bounding-box widths for the next time step.
5. Invalid next-step corners or orientation cause full per-step rollback of particle and transient nodal state.

**Limitations:** fixed 2D structured background grid; all four corners must stay on it; no domain fragmentation, arbitrary quadrilateral CPDI2, remeshing, volume/mass partition correction or automatic mesh expansion; no FEM transfer, contact, plasticity or free-boundary validation. Large aspect-ratio/domain rotations need additional robustness/convergence testing. The generic solver's CFL formula is heuristic and not a CPDI-specific stability proof.

## Numerical verification and expected results

- `tests/test_stage10c_cpdi.py`: affine reproduction under identity, stretch, shear, ~53° and 90° rotations, and 180° rotation; invalid/degenerated/inverted F rejection; domain can span multiple cells; stale snapshot; solver shear/rotation; rollback; opt-in API.
- `python -m validation.stage10c.validate_cpdi` runs affine shear/rotation/extension (comparison with exact `exp(tA)`) and **force-driven** small-strain elastic-wave test comparing `mpm`, `gimp`, `cpgimp`, `cpdi` with drift velocity 1 m/s.
- The d'Alembert reference is **small-strain**. Its stress error is not an independent finite-deformation, rotating-domain solution and **must not** be used to assert CPDI superiority for general large-deformation landslides.
- Run full regression locally: `python -m pytest -q`; the original baseline is not silently changed to CPDI.

## Suggested next validation after Stage 10C

A rotating/shearing **force-driven continuum** with an independent finite-deformation reference, followed by mesh and particle sensitivity for CPDI and comparison to uGIMP/cpGIMP. Also investigate Galilean invariance and domain overlap/gap effects. Only after those validations should this implementation be used as a candidate for landslide runout research.
