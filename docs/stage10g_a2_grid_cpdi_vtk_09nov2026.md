# GeoPyMPM Stage 10G-A2 — Background Grid and CPDI1 Domain VTK Writer

**Date:** 9 November 2026  
**Baseline:** Stage 10G-A1 (Particle VTK Writer)  
**Scope:** Visualization output only. No modification to solver, interpolation kernels or particle/grid state.

## Scope and rationale

This stage exports two independent `.vtu` datasets. `write_grid_vtu` writes the fixed structured 2D `BackgroundGrid` as **VTK_QUAD (cell type 9)** mesh with nodal fields. `write_cpdi_domains_vtu` writes the actual 2D CPDI1 particle parallelograms as **VTK_QUAD** cells with per-particle data in `CellData`. These supplement the Stage 10G-A1 particle-vertex writer. All writers can be invoked after a committed solver step or on a state snapshot; automated output scheduling remains Stage 10G-A3.

## New modules

- `src/geopympm/io/_vtu_mesh.py` — private common XML mesh serializer and validation; ASCII Float64, 17 significant digits; optional `TimeValue`, `Step`; no `vtk` dependency.
- `src/geopympm/io/grid_vtk_writer.py` — `write_grid_vtu(grid, filename, *, time=None, step=None, fields=None, cell_fields=None, overwrite=False)`.
- `src/geopympm/io/cpdi_vtk_writer.py` — `write_cpdi_domains_vtu(particles, grid, filename, *, time=None, step=None, fields=None, point_fields=None, overwrite=False)`.
- `src/geopympm/io/__init__.py` — exports both public writers and field-name constants.
- `examples/18_grid_cpdi_vtu_export.py` — four example VTU files, with **prescribed/manufactured affine** shearing/rotation/stretching for visible domain evolution, not a force-driven solution.
- `tests/test_stage10g_a2_mesh_vtk.py` — XML, topology, geometry, field, error and independent VTK reader regression tests.
- `validation/stage10g_a2/validate_mesh_vtu.py` — validation and JSON report.

## Grid representation

2D nodes use `node_id(ix,iy) = iy*(nx+1)+ix`. Each grid cell uses the same CCW ordering as `BackgroundGrid.cell_node_ids`: **BL, BR, TR, TL**. Grid x/y positions are embedded as `(x,y,0)`. The export includes every grid cell, including cells with no active material. `PointData`: `node_id`, `mass`, `momentum`, `velocity`, `acceleration`, `force_internal`, `force_external`, `velocity_bc_mask`, `velocity_bc_value`. 2-component vectors/BC masks are extended with a zero z component. `CellData`: `cell_id`, `cell_ix`, `cell_iy`. The transient nodal fields reflect their actual snapshot; when called before P2G or after a nodal reset, their values can legitimately be zero.

## CPDI1 domain representation

For particle position **x**, initial **full** side lengths `(l0x,l0y)`, and full 2×2 deformation gradient **F**, the half-edge vectors are

```
a = F @ [l0x/2, 0]
b = F @ [0, l0y/2]
corners = [x-a-b, x+a-b, x+a+b, x-a+b]
```

The function calls `candidate_cpdi_geometry` from the existing CPDI module; this shares its finite-state, positive area and grid-boundary checks. **Do not use `domain_size` as the polygon dimensions**: for CPDI this is an axis-aligned bounding envelope only. Each domain has four independent VTK points, in CCW ordering, with `corner_id` as PointData and per-particle fields as CellData. `CellData`: `particle_id`, `material_id`, `domain_area`, `reference_domain_area`, `area_ratio`, `jacobian`, `pressure`, `von_mises`, `stress`, `deformation_gradient`. The full stress tensor retains the plane-strain `sigma_zz` and tension-positive internal convention; `pressure=-trace(sigma)/3` is compression-positive. The finite deformation satisfies `domain_area/reference_domain_area = det(F)` within floating-point tolerance.

**Limitations:** This writer is strictly for 2D CPDI1 parallelograms; cpGIMP/uGIMP domains are not represented by these quads. The writer rejects inverted/collapsed domains and corners outside the background grid, consistent with the existing CPDI interpolation contract. It does not perform geometric repair, clipping, time-series management, checkpointing or restart.

## Use and VisIt workflow

```bash
source .venv/bin/activate
python examples/18_grid_cpdi_vtu_export.py
python -m validation.stage10g_a2.validate_mesh_vtu
python -m pytest tests/test_stage10g_a2_mesh_vtk.py -q -rs
python -m pytest -q
```

In VisIt, open `results/stage10g_a2/grid_000.vtu` using a **Mesh** plot to show the background grid. Open one of `domains_000.vtu`, `domains_001.vtu`, `domains_002.vtu` as a **Mesh** plot or **Pseudocolor** of `area_ratio`, `domain_area`, or `von_mises` (**cell-centered**). These are independent snapshot files; Stage 10G-A3 will supply `.visit`/`.pvd` time series and automatic scheduling. Overlaying the grid and CPDI plots shows cell crossing and rotations.

The VTK Python reader test is optional (`pytest.importorskip('vtk')`). Installation with `python -m pip install vtk` enables independent parser checks. The VisIt GUI must be checked on the target machine separately.

## Acceptance criteria

1. Correct `VTK_QUAD` and x-fast BL/BR/TR/TL ordering for the grid.
2. Current nodal fields and velocity boundary conditions are exported without modifying grid arrays.
3. CPDI parallelogram vertices match the solver's analytic full-**F** domain geometry, including shear and rotation.
4. Area and orientation are correct, and degenerate/out-of-grid domains fail safely.
5. `.vtu` files can be parsed independently by VTK (when available); no solver/kernel modifications.
6. Old A1 and wider regression tests continue to pass.
