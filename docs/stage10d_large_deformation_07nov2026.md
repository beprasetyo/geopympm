# GeoPyMPM Stage 10D — Large-Deformation CPDI1 Validation

**Date:** 7 November 2026  
**Baseline:** GeoPyMPM Stage 10C CPDI1 (28 October 2026)  
**Implementation impact:** validation code and tests only; **no changes to solver kernels**.

## 1. Purpose and claim boundary

Stage 10C established CPDI1 corner interpolation and small-strain wave testing. Stage 10D checks whether the **existing CPDI1 solver** correctly evolves full deformation gradients, particle centers, parallelogram domains, volume and **Jaumann hypoelastic** stress under large prescribed affine motions.

All active grid-node velocities are **prescribed**. Although nodal internal forces are assembled, they do **not** determine the motion of constrained degrees of freedom. This is a **kinematic/material-point verification**, **not** a validation of force-driven nonlinear dynamics, traction conditions, elastoplastic soils, or landslide runout. The reference stress is *not* a finite-strain hyperelastic model; it is the analytical solution of the **same objective Jaumann rate** used by the implemented Stage 7 stress updater.

We keep `interpolation="mpm"`, `"gimp"`, `"cpgimp"` and `"cpdi"` unchanged. Stage 10D does not claim that CPDI1 globally outperforms the other schemes. Results from Stage 10A–10C about force-driven wave stress errors remain open limitations.

## 2. Reproduction

From repo root with the editable package installed:

```bash
python -m validation.stage10d.validate_large_deformation --refinement
python -m pytest tests/test_stage10d_large_deformation.py -q
python -m pytest -q
```

Optional parameters: `--dt 0.005`, `--cell-size 0.25`, `--n-side 4`, `--no-plot`, `--output OUTPUT_DIR`. `--refinement` compares stress time steps (`0.02, 0.01, 0.005 s`) and background grid sizes (`1.0, 0.5, 0.25 m`). No `matplotlib` is needed for tests or CSV/JSON outputs; if it is installed, the default run also writes PNGs.

Results are written to `validation/results/stage10d/`:

- `10D_summary.json`, `10D_affine_history.csv`, `10D_end_summary.csv`;
- `10D_refinement.csv`, `10D_refinement_summary.json` when requested;
- `10D_domain_comparison.png`, `10D_F_error.png`, `10D_stress_error.png` (matplotlib optional).

## 3. Benchmark geometry and conditions

A regular **4 x 4 patch** of material points represents a **contiguous** planar square region. Each initial particle domain is `0.4 x 0.4 m`, so its representative volume with 1-m unit thickness is `0.16 m^3`. Density is `1200 kg/m^3`; isotropic elastic parameters are `E=1200 Pa` and `nu=0.25`. Grid size `h=0.5 m` on a `12 x 12 m` computational domain. Default fixed time step is `0.01 s`; all active grid node velocities satisfy

\[
\mathbf v(\mathbf x)=\mathbf A(\mathbf x-\mathbf c)+\mathbf U,\qquad \mathbf c=(5,5)\ \mathrm m.
\]

The **same** affine gradient is used for each particle. The grid velocity DOFs are prescribed to eliminate errors due to force feedback and isolate the interpolation/kinematic/stress machinery.

| Mode | \(\mathbf A\) (s^-1) | Final time (s) | Characteristic response |
| --- | --- | --- | --- |
| Large simple shear | `[[0, 1], [0, 0]]` | 2.0 | \(\gamma=2\), with drift \(\mathbf U=(0.7,0)\,\mathrm{m/s}\) |
| Rigid-body rotation | `[[0, -pi/2], [pi/2, 0]]` | 2.0 | Rotation of 180 degrees; initial nonzero stress |
| Rotation/stretching | `[[0.35, -1.20], [0.80, -0.12]]` | 1.0 | Coupled spin, stretching and nonzero \(J\) |

## 4. Independent references

For constant velocity gradient \(\mathbf A\):

\[
\mathbf F(t)=\exp(t\mathbf A),\qquad J=\det\mathbf F,
\]

and for the particle centers:

\[
\mathbf x(t)=\mathbf c+\exp(t\mathbf A)(\mathbf X-\mathbf c)
+\int_0^t\exp(s\mathbf A)\mathbf U\,ds.
\]

The reference matrix exponential uses a direct 2x2 **Taylor power series**, independent of the Stage 8 Numba closed-form trace/traceless exponential implementation. This explicitly handles the **defective simple-shear Jordan matrix**, for which naive eigendecomposition is numerically unreliable.

CPDI1 reference corners for half-reference-domain edges \(\mathbf a_0=(l_{x0}/2,0)\) and \(\mathbf b_0=(0,l_{y0}/2)\) are

\[
\mathbf x_p^{(\pm,\pm)}(t)=\mathbf x_p(t)\pm\mathbf F_p(t)\mathbf a_0\pm\mathbf F_p(t)\mathbf b_0.
\]

For the same constant-\(\mathbf A\) plane-strain **Jaumann hypoelastic** constitutive law as the code, \(\mathbf D=\tfrac12(\mathbf A+\mathbf A^T)\), \(\mathbf W=\tfrac12(\mathbf A-\mathbf A^T)\), and \(\mathbf B=2\mu\mathbf D+\lambda\operatorname{tr}(\mathbf D)\mathbf I\). The stress reference solves

\[
\dot{\boldsymbol\sigma}=\mathbf W\boldsymbol\sigma-\boldsymbol\sigma\mathbf W+\mathbf B,
\]

so that, for \(\mathbf R(s)=\exp(s\mathbf W)\),

\[
\boldsymbol\sigma(t)=\mathbf R(t)\boldsymbol\sigma(0)\mathbf R(t)^T
+\int_0^t\mathbf R(s)\mathbf B\mathbf R(s)^T\,ds.
\]

The integral is evaluated with independent 32-point Gauss-Legendre quadrature. The out-of-plane component follows \(\sigma_{zz}=\sigma_{zz}^0+\lambda\operatorname{tr}(\mathbf D)t\). For initially stress-free simple shear \(\gamma=t\) at shear rate 1/s:

\[
\sigma_{xx}=\mu(1-\cos\gamma),\quad
\sigma_{yy}=-\mu(1-\cos\gamma),\quad
\sigma_{xy}=\mu\sin\gamma,
\]

which is characteristic of the **Jaumann hypoelastic** model and should not be confused with finite-strain hyperelastic stress.

## 5. Primary results (dt=0.01 s, 4 x 4 domains)

| Case | Max \(|F-F_{ref}|\) | Max center error (m) | Max stress component error (Pa) | Max adjacent-edge mismatch (m) |
| --- | ---: | ---: | ---: | ---: |
| Shear \(\gamma=2\) | 1.78e-15 | 6.84e-14 | 0.0028323 | 8.88e-14 |
| Rotation 180 degrees | 1.09e-15 | 7.64e-15 | 7.03e-13 | 1.11e-14 |
| Combined rotation/stretch | 1.30e-14 | 5.69e-15 | 0.0039754 | 1.51e-14 |

Final Jacobians from the analytic map: `1.0` (shear), `1.0` (pure spin), `1.2586000099` (combined). CPDI1 polygon areas agree with the particle volume under the test's **initial domain area = representative particle volume** convention. Mass remains constant. Particle cell indices change during the tests; this is not a standalone force-driven cell-crossing accuracy test.

### 5.1 Time-step refinement

| dt (s) | Maximum shear stress error (Pa) |
| ---: | ---: |
| 0.020 | 0.0113293 |
| 0.010 | 0.0028323 |
| 0.005 | 0.0007081 |

The error approximately quarters on each halving of dt, consistent with the expected **second-order midpoint stress integration** for constant \(\mathbf A\). Refinement of the *background grid* `h=1.0, 0.5, 0.25 m` at fixed dt produces the same manufactured-affine stress error (`~0.0028323 Pa`) and essentially machine-precision \(F\). This is a **consistency/invariance check for an imposed affine field**—**not** spatial convergence of a free dynamic solution.

### 5.2 Particle-domain geometry under shear

For the same analytical centers and final full \(\mathbf F=[[1,2],[0,1]]\):

| Representation | Max adjacent-domain edge mismatch (m) | Pairwise polygon overlap in this *affine* test (m^2) |
| --- | ---: | ---: |
| uGIMP fixed rectangles | 0.8 | 0 |
| cpGIMP axis-aligned rectangles | 0.8 | 0 |
| CPDI1 convected parallelograms | ~2e-15 | 0 |

The uGIMP/cpGIMP shapes in this geometry-only comparison were constructed at **identical analytically convected centers**; this is not a full dynamic solve for those interpolation methods. A zero pairwise-overlap measure is **not evidence of no gaps**; the nonzero edge residual reveals loss of conformity for these rectangles.

### 5.3 Non-affine deformation (limitation demonstration)

A separate geometric diagnostic samples the smooth map

\[
x'=x+k(x-c_x)(y-c_y)^2,\qquad y'=y,\qquad k=1\,\mathrm{m}^{-2},
\]

at particle centers and uses each **local** deformation gradient in the CPDI1 parallelogram construction. The exact continuum map is continuous, but its local linear approximation is not globally conforming between independently updated particle domains:

- Max shared-edge endpoint mismatch: `0.096 m`;
- Sum of pairwise polygon overlaps: `0.0768 m^2`.

The overlap sum is **not** an exact union-overlap measure when three or more polygons overlap. Edge mismatch indicates the potential for gaps or overlap; this test directly detects overlap but does **not** quantify the total uncovered area. These are **diagnostics of CPDI1's local affine approximation**, not evidence that the integrated solver created those exact values in a force-driven nonuniform simulation.

## 6. Interpretation and development status

- **Verified in Stage 10D:** large imposed affine kinematics; CPDI1 domain rotation/stretch/shear; consistency of Jacobian, physical volume and polygon area; stress-objectivity under 180-degree rotation; Jaumann integration convergence; partition of the affine particle patch.
- **Not verified by Stage 10D:** large-deformation force-driven accuracy, fully free-boundary problems, physical energy conservation for a hyperelastic potential, plasticity, contacts, FEM-to-MPM transfer, general non-affine conformity, 3D or runout.
- The prior Stage 10A wave/Galilean sensitivity observations **remain open**, not invalidated by manufactured-affine results.
- CPDI1's parallelogram assumption is appropriate for a local affine map, but non-affine spatial variation can create gaps and overlaps. Extended/second-order or enriched CPDI variants, domain remapping and robust particle-to-grid transfer may be needed for strongly nonuniform finite deformation.

### 6.1 Source locations

- `validation/stage10d/validate_large_deformation.py` — benchmark, independent references and diagnostics.
- `tests/test_stage10d_large_deformation.py` — reproducible checks.
- The **existing** solver code in `src/geopympm/` is unchanged by this patch.

### 6.2 References

- Sadeghirad, A., Brannon, R. M. & Burghardt, J. (2011). *A convected particle domain interpolation technique to extend applicability of the material point method for problems involving massive deformations*. International Journal for Numerical Methods in Engineering, 86(12), 1435–1456. DOI: [10.1002/nme.3110](https://doi.org/10.1002/nme.3110).
- Sadeghirad, A., Brannon, R. M. & Guilkey, J. (2011). *Enriched convected particle domain interpolation (CPDI) method for analyzing weak discontinuities*. Particles 2011 conference contribution, discussing gap/overlap effects in particle domains.
