# GeoPyMPM Validation Example: Sulsky Colliding Elastic Disks

**Date:** 9 November 2026  
**Baseline:** GeoPyMPM Stage 10G-D4  
**Status:** reproducible 2D *linear-elastic surrogate*, **not** a validated quantitative reproduction of Uintah's finite-strain Neo-Hookean response.  
**Patch:** adds one example, one test file, and this document; **no `src/geopympm/` modifications**.

## 1. Source specification and provenance

- Uintah User Guide (Guilkey et al., 2009), §7.6, "Colliding Disks", Figure 7.5, pp. 152–153: <https://publications.sci.utah.edu/publications/SCITechReports/UUSCI-2009-007.pdf>.
- Uploaded `Uintah-master.zip`, file `Uintah-master/src/StandAlone/inputs/UintahRelease/MPM/disks_sulsky.ups`: source-of-truth for geometry, constitutive constants, boundary conditions, time limits, and output interval.
- Sulsky, Chen & Schreyer (1994), "A particle method for history-dependent materials", *Computer Methods in Applied Mechanics and Engineering* 118, 179–196, §7.3, Figures 4–5: <https://users.cs.utah.edu/~mhall/cs6963s09/lectures/Sulsky_1994.pdf>.

### Input parameters

| Parameter | Uintah `disks_sulsky.ups` | This example |
|---|---|---|
| Domain | (0,0,0) – (1,1,0.05) m | (0,0) – (1,1) m; plane-strain thickness 0.05 m |
| Grid | 20×20×1 cells | 20×20 2D cells |
| Cell spacing | 0.05×0.05×0.05 m | 0.05×0.05 m |
| Disk axis | (0.25,0.25,z), (0.75,0.75,z) | Center A (0.25,0.25), B (0.75,0.75) |
| Radius | 0.20 m | 0.20 m |
| Mass density | 1000 kg/m³ | 1000 kg/m³ |
| Velocity A / B | (+0.1,+0.1,0) / (−0.1,−0.1,0) m/s | same in-plane |
| E and ν | Comments in UPS: 1000 Pa and 0.30 | 1000 Pa and 0.30 |
| K and G | 833.3 Pa and 384.6 Pa (UPS) | 833.333… Pa and 384.615… Pa (from E,ν) |
| Constitutive model | `comp_neo_hook` (compressible Neo-Hookean) | **small-increment linear elastic**, corotational stress update |
| Interpolation | `linear` (standard MPM) | `mpm` default; optional `gimp` sensitivity |
| Particle resolution | `res=[2,2,2]` (per Eulerian cell) | 2×2 subcell centers, through-thickness layers collapsed to 2D mass/volume |
| Contact | `null`, material index [0], single shared material | one nodal velocity field; two **diagnostic** material IDs with identical elastic constants |
| Boundary conditions | x± and y± fixed velocity, z± symmetry | grid velocity fixed at x/y domain boundary, plane strain in z |
| Gravity | zero | zero |
| Time control | dt_min=0.0001, dt_max=0.01, multiplier=0.2, maxTime=3 s | dt=0.001 s fixed (Sulsky 1994), 3000 steps to 3 s |
| Output interval | 0.025 s | `--output-every 25`, i.e. 0.025 s for dt=0.001 |

**Discretization caveat:** The center-inclusion seeding produces 208 particles per disk (=416 total) on this lattice, each with mass 0.03125 kg. The discrete disk mass is **6.5 kg**, whereas the analytical circular cylinder mass is **6.283185… kg** (3.45% larger). These discrete counts follow this documented 2D seeding choice; exact Uintah particle counts and initial mass are **not asserted**. The UPS specifies a 3D particle resolution of 2×2×2 in one thickness cell. We collapse the thickness layers to one 2D particle per (x,y) position, retaining a 0.05 m tributary thickness. This is not a bit-for-bit translation of Uintah's particle creator.

**Contact caveat:** The UPS includes both disks as objects of one material and chooses null contact, whereas our two material IDs are just visualization/group labels mapped to identical properties. GeoPyMPM has one nodal velocity field, so it can transmit momentum when interpolation supports overlap. This is NOT a validated multimaterial contact scheme or an independently implemented contact law.

## 2. Running the example

From the root of the repository (after installing editable with `pip install -e .`):

```bash
source .venv/bin/activate
python -m pytest -q tests/test_sulsky_colliding_disks.py

# Quick smoke run: 100 steps, creates particle VTU/PVD and CSV/JSON
python examples/23_sulsky_colliding_disks.py \
  --steps 100 --output-dir results/sulsky_smoke

# Full reference time, 3000 steps (dt=0.001 s)
python examples/23_sulsky_colliding_disks.py \
  --output-dir results/sulsky_full --plot

# Faster, no VTU snapshots:
python examples/23_sulsky_colliding_disks.py \
  --output-dir results/sulsky_no_vtu --no-vtu --plot

# Optional sensitivity run, not an exact Uintah comparison:
python examples/23_sulsky_colliding_disks.py \
  --interpolation gimp --output-dir results/sulsky_gimp --plot
```

The program REFUSES to write to a nonempty output directory to avoid overwriting old validation results. Use a fresh `--output-dir` for each experiment. `--plot` requires Matplotlib (`python -m pip install matplotlib` if necessary), but plotting is optional.

VTU animation:
- ParaView: `results/sulsky_full/visualization/particles.pvd`.
- VisIt: `results/sulsky_full/visualization/particles.visit`.
- For coloring: `material_id` (0 vs 1), `velocity`, `von_mises`.

Output data:
- `diagnostics.csv`: time, per-disk center-of-mass position and momentum, total momentum, kinetic energy, **approximate** linear elastic energy, peak equivalent stress, and distance between centers.
- `summary.json`: simulation configuration, initial/final energy, momentum drift, disk reversal flags, and particle counts.
- `energy.png` and `momentum.png`: optional graphs when `--plot` is supplied.

## 3. Diagnostic definitions

Per-disk linear momentum:

\[\mathbf P_d=\sum_{p\in d} m_p\mathbf v_p.\]

Global kinetic energy:

\[K=\frac12\sum_p m_p\lVert\mathbf v_p\rVert^2.\]

Our diagnostic elastic energy is **not** the `StrainEnergy.dat` of Uintah's compressible Neo-Hookean model. It is constructed using the stress of the currently active linear elastic model:

\[U_\mathrm{lin}\approx\sum_p V_p\left[\frac{\sigma_{xx}^2+\sigma_{yy}^2+\sigma_{zz}^2-2\nu(\sigma_{xx}\sigma_{yy}+\sigma_{yy}\sigma_{zz}+\sigma_{zz}\sigma_{xx})}{2E}+\frac{\sigma_{xy}^2}{2G}\right].\]

Because this uses current Cauchy stress and volume under finite particle motion, the sum `K+U_lin` is **only an approximate diagnostic**. It must not be used as proof of exact energy conservation or equated with a finite-strain Neo-Hookean strain-energy function.

## 4. Observed baseline result (GeoPyMPM 10G-D4)

Locally verified with `--steps 3000 --no-vtu --plot`, default FLIP (`pic_fraction=0`), 416 particles, standard MPM, 0.001-second fixed step:

| Time (s) | Disk A px (kg m/s) | Kinetic energy (J) | Approx. U_lin (J) |
|---:|---:|---:|---:|
| 0.00 | +0.6500 | 0.1300 | 0.0000 |
| 0.50 | +0.6500 | 0.1300 | ~0.0000 |
| 1.00 | +0.5879 | 0.1111 | 0.0177 |
| 1.25 | +0.3050 | 0.0383 | 0.0844 |
| 1.50 | −0.1100 | 0.0063 | 0.1133 |
| 1.75 | −0.4494 | 0.0760 | 0.0482 |
| 2.00 | −0.6066 | 0.1188 | 0.0083 |
| 3.00 | −0.6147 | 0.1194 | 0.0067 |

- **Both disk momenta reverse signs** by about 1.5 s; rebound is observed.
- Maximum global momentum drift through 3000 steps: **1.67×10⁻¹⁵ kg m/s** in this symmetric initial condition.
- The computed approximate `K+U_lin` is 0.12614 J at 3 s, vs 0.13000 J initially (−2.97%). The minimum over the run reaches about 0.11929 J (−8.24%). This is not sufficient to certify total energy conservation.
- These results show qualitative rebound and symmetric momentum behavior; they **do not** establish quantitative agreement with Uintah's energy-time graph or with the exact contact response.

## 5. Validation boundary / proposed next steps

**Current assessment:** structural/geometric setup verified, smoke integration and two-disk rebound observed; **quantitative cross-code validation remains pending**.

To turn this into a quantitatively comparable validation case:

1. Implement a **compressible Neo-Hookean** stress law with consistent finite-deformation energy, validated independently by stress/deformation and energy tests.
2. Match particle seeding and time-control algorithms more exactly (including `res=[2,2,2]` and dynamic step-size policy).
3. Generate a reference run in Uintah using `disks_sulsky.ups` and export `KineticEnergy.dat`, `StrainEnergy.dat`, and per-disk momentum; compare at aligned physical times. (Original UPS supplies global momentum, not explicit per-disk momentum outputs; extra extraction may be required.)
4. Perform grid/particle refinement and PIC/FLIP/interpolation sensitivity; inspect symmetry, momentum balance, energy trend, deformation profile, and the onset of rebound.
5. Do not mistake single-field shared-node interaction for a validated separated-material contact implementation.

The benchmark is a useful **pre-Stage 11 dynamic integration check**, not a proof of full material or contact validation.
