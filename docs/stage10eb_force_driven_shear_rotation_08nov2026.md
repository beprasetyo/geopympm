# GeoPyMPM Stage 10E-B — Force-Driven Nonlinear Shear and Local Rotation

**Date:** 8 November 2026  
**Baseline:** Stage 10E-A.1 near-massless diagnostics (8 November 2026)  
**New files:** validation benchmark, nonintrusive boundary-node diagnostic, tests, example, plots and CSV/JSON.  
**Solver and constitutive kernels:** unchanged.

## 1. Scientific question and validity scope

Does the existing GeoPyMPM solver reproduce a **finite-amplitude shear wave** whose shear stress, incremental deformation gradient, and local polar rotation can be obtained from an independent analytical reference, when all horizontal motion is computed by the internal-force time integrator?

This benchmark uses the **Jaumann hypoelastic rate law** actually implemented in GeoPyMPM, not an interchangeable hyperelastic law. Although it is implemented in a two-dimensional MPM solver and exhibits shear plus local rotation, its exact reference is a **one-dimensional spatial reduction of plane-strain kinematics**. It constrains all nodal velocities in the y direction to zero. No horizontal grid velocity is imposed, gravity is zero, and no external nodal forces are used. A finite-width strip has free lateral boundaries; errors are evaluated only in the middle x band before boundary disturbances can reach it.

It therefore does **not** validate arbitrary two-dimensional shear loading, bending, a rotating disk, or a 180-degree force-driven rotation. Those remain to be tested before claiming general 2D large-deformation validity.

## 2. Derivation of independent continuum reference

In the constrained simple-shear reduction,

\[
\mathbf v=(v(y,t),0),\quad
\mathbf L=\begin{bmatrix}0&v_{,y}\\0&0\end{bmatrix},\quad
\dot\gamma=v_{,y},\quad \det\mathbf F=1.
\]

Integrating the **Jaumann rate** of isotropic hypoelasticity along simple shear with shear modulus \(\mu\) gives

\[
\sigma_{xy}=\mu\sin\gamma,\qquad
\sigma_{xx}=\mu(1-\cos\gamma),\qquad
\sigma_{yy}=-\mu(1-\cos\gamma).
\]

The y-velocity constraint provides the reaction to the nonzero normal stress gradient. The x-momentum balance and kinematics become

\[
\rho_0\,v_{,t}=\partial_y\sigma_{xy},\qquad
\gamma_{,t}=v_{,y},\qquad
c(\gamma)=\sqrt{\frac{\mu\cos\gamma}{\rho_0}}.
\]

Hyperbolicity requires \(|\gamma|<\pi/2\). A right-going simple wave satisfies

\[
 v(\gamma)=-\int_0^\gamma c(q)\,dq,\qquad
 \gamma_{,t}+c(\gamma)\gamma_{,y}=0.
\]

For \(\gamma_0(y)=A\exp[-(y-y_c)^2/(2w^2)]\), the pre-shock solution is determined independently by scalar characteristic inversion,

\[
 y=\xi+t c(\gamma_0(\xi)),\qquad
 \gamma(y,t)=\gamma_0(\xi).
\]

The code uses bisection and a 48-point Gauss–Legendre integration of \(v(\gamma)\), not the MPM interpolation/constitutive kernels, to construct the reference. The minimum characteristic-map Jacobian at \(t=0.9\,\mathrm s\) is **0.75145**, so the reference remains pre-shock.

**Initialization/reference convention:** The initial MPM particle configuration is taken as the current, pre-stressed reference. \(\mathbf F_{\mathrm{MPM}}(0)=\mathbf I\), even though initial stresses encode a virtual initial total shear \(\gamma_0\). Hence for the shear-wave kinematics the expected **incremental** deformation gradient is

\[
\mathbf F_{\mathrm{MPM}}=\begin{bmatrix}1&\gamma(y,t)-\gamma_0(y)\\0&1\end{bmatrix}.
\]

The associated local polar rotation angle is \(\theta=-\arctan[(\gamma-\gamma_0)/2]\). This is not an absolute total rotation of an initially unstressed block; the reference state begins with prescribed stresses and velocities.

## 3. Discretization and computational details

| Parameter | Value |
| --- | --- |
| Young's modulus \(E\) | 1200 Pa |
| Poisson's ratio \(\nu\) | 0.25 |
| Shear modulus \(\mu\) | 480 Pa |
| Density \(\rho_0\) | 1200 kg/m³ |
| Shear-wave speed at \(\gamma=0\) | 0.6324555 m/s |
| Peak initial shear \(A\) | 1.1 (engineering shear) |
| Gaussian width \(w\) | 0.65 m |
| Particle strip (x,y) | [4,10] × [2,8] m |
| Comparison window | \(6\le x_0\le8\) m |
| Grid size | \(h=0.25\) m (refinement 0.5, 0.125) |
| Particles per cell per direction | 2 |
| Particle count, h=0.25 | 2304 |
| Time step | 0.003 s |
| Final time | 0.9 s, 300 steps |
| PIC fraction | 0 (FLIP) |
| Kinematics, advection, stresses | Exponential F, affine-exponential advection, corotational Jaumann |
| Vertical boundary condition | Every grid node has \(v_y=0\) |
| Horizontal velocity BC | None |
| External nodal forces | None |

Horizontal particle velocities are initialized from the analytic *right-going* simple-wave invariant, but are **not prescribed during time stepping**. Horizontal internal forces produce nonzero grid acceleration and feedback into particle motion.

## 4. Main benchmark results

Errors are computed per particle only in the initial **central x band**; the full-domain free-edge response is not assumed to satisfy the one-dimensional reference.

| Interpolation | Final relative \(L_2\) error in \(\sigma_{xy}\) | RMS absolute \(F_{xy}\) error | RMS polar-angle error |
| --- | ---: | ---: | ---: |
| Standard MPM | 5.484616% | 0.03044 | 0.855° |
| uGIMP | 5.484612% | 0.03044 | 0.855° |
| cpGIMP | 5.484614% | 0.03044 | 0.855° |
| CPDI1 | 5.484502% | 0.03044 | 0.855° |

All four interpolation schemes give very similar mid-strip shear errors, **not** a demonstrated accuracy advantage for CPDI1. The chosen particle spacing and loading may not distinguish the schemes well; this should be a follow-on sensitivity question rather than being hidden.

For CPDI1, the *reference* maximum incremental \(|F_{xy}|\) is 0.5319 and the numerical maximum is 0.5019 on h=0.25 m. The reference maximum local polar rotation is 14.89° and the computed maximum is 14.09°. The motion is finite shear plus **local rotation**, not a prescribed 180° rigid-body rotation.

### Grid sensitivity (CPDI1)

| h (m) | Shear stress relative L2 error | RMS absolute Fxy error |
| ---: | ---: | ---: |
| 0.500 | 11.8935% | 0.06340 |
| 0.250 | 5.4845% | 0.03044 |
| 0.125 | 2.6069% | 0.01470 |

The error decreases monotonically with grid refinement. All 10 benchmark checks in `10EB_summary.json` are true. **The checks establish reproducibility and numerical consistency for the benchmark, not general-purpose solver validation.**

### Conservation and reduced-model energy

The four runs at h=0.25 m conserve total particle mass exactly within the reported floating-point precision. The maximum total x-momentum drift is on the order of \(10^{-11}\,\mathrm{kg\,m/s}\); the vertical velocity constraint is satisfied. The **center-band reduced shear-energy proxy** is defined as

\[
 E_{\mathrm{core}}=\sum_{p\in\mathrm{core}}m_p\bigg(
 \frac12 v_{x,p}^2 + \frac{\mu}{\rho_0}[1-\cos(\gamma_0(y_{p,0})+F_{xy,p})]\bigg).
\]

Its maximum relative drift is approximately \(2.7\times10^{-5}\)–\(3.0\times10^{-5}\) in the h=0.25 m runs. This quantity is **valid only as a diagnostic for the constrained one-dimensional shear reduction**; the general 2D Jaumann hypoelastic stress law does not automatically define this finite-strain stored-energy potential.

## 5. Side-boundary acceleration and low-mass-node warning

The benchmark also exposes significant peaks in **nodal** acceleration at free x-boundary regions. The peak is not at the central reference window. Values for h=0.25 m:

| Method | Raw maximum nodal | Mass at peak | Mass / characteristic positive-node mass |
| --- | ---: | ---: | ---: |
| MPM | 1868.3 m/s² | 2.238e-4 kg | 3.27e-6 |
| uGIMP | 5975.9 m/s² | 1.499e-7 kg | 2.26e-9 |
| cpGIMP | 13085.1 m/s² | 2.831e-8 kg | 4.28e-10 |
| CPDI1 | 6.089 m/s² | 1.044e-2 kg | 1.57e-4 |

Using **only diagnostic**, not solver, masking of nodes with mass less than \(10^{-8}\) times the characteristic mass, the maximum nodal acceleration for uGIMP falls to 116.5 m/s² and cpGIMP to 11.48 m/s². At these peak steps, the absolute upper bound of near-node FLIP \(|\Delta v|\) is \(1.43\times10^{-7}\,\mathrm{m/s}\) (uGIMP) and \(5.93\times10^{-8}\,\mathrm{m/s}\) (cpGIMP). MPM's peak node is **above** that diagnostic threshold and cannot be dismissed as simply massless.

All the corresponding interpolation supports are captured **before** `solver.step()` so that diagnostic node masses and accelerations refer to precisely the same interpolation, as in Stage 10E-A.1. No mass filtering is used in the solver. Peak boundary accelerations do not by themselves prove unstable particle dynamics, but make boundary and transfer sensitivity a **remaining research item**.

## 6. Reproduction

From project root with an editable package install (`pip install -e .`) and activated `.venv`:

```bash
python examples/15_force_driven_2d_shear.py
python -m validation.stage10eb.nonlinear_shear --refinement
python -m validation.stage10eb.edge_node_diagnostics
python -m pytest tests/test_stage10eb_nonlinear_shear.py -q
python -m pytest -q
```

Outputs go to `validation/results/stage10eb/`: cases CSV, history CSV, analytical comparison PNGs, refinement PNG, summary JSON, and edge peak diagnostic JSON.

Regression in the preparation environment: **480 tests passed excluding `test_stage6f_energy_convergence.py`; that file separately passed 27 tests**. Running the entire suite in one process exceeded its execution cap and is not claimed complete. The user's own local run should repeat the full suite.

## 7. Stage-status decision and remaining work

**Completed:** reproducible benchmark of finite-amplitude force-driven x-motion in a constrained plane-strain shear reduction, analytical stress/deformation/rotation references, refinement, local energy and conservation diagnostics, nonintrusive boundary low-mass diagnosis.

**Still open:** arbitrary spatially two-dimensional force-driven shear/bending/torque, large rigid-body rotations created by forces, continuum-consistent moving surface tractions, support and free-surface sensitivity, non-affine CPDI particle-domain overlap/gaps. The source at this stage is not adequate evidence for fully general 2D landslide runout.

**No source modifications:** `src/geopympm` stays byte-for-byte identical to Stage 10E-A.1. Continue to Stage 10E-B2 or Stage 10F only with this limitation explicitly recorded.
