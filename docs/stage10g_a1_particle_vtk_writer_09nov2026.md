# Stage 10G-A1 — Particle VTK XML Writer

**Baseline:** GeoPyMPM Stage 10F (8 Nov 2026). **Status:** implemented and initially verified.

## Scope and choice

The writer creates a standard VTK XML **UnstructuredGrid** (`.vtu`), with one
`VTK_VERTEX` (cell type 1) for each particle. The particle's position is the
VTK point; its quantities are stored as `PointData`. This follows the output
principle of `MPM3D-F90/src/DataOut.f90` without copying its Fortran structure.
The writer does **not** save a restart state and does **not** schedule outputs;
those belong to Stage 10G-B and Stage 10G-A3 respectively.

**Implementation:** `src/geopympm/io/vtk_writer.py`, exported through
`geopympm.io.write_particles_vtu`. Python stdlib XML + NumPy; no VTK installation
is needed to *write* files. Format: ASCII, Float64 round-trip safe (`.17g`),
Int32 material IDs, Int64 connectivity, UInt8 VTK_VERTEX types. ASCII is
appropriate for initial validation, but is not an optimized large-run format.

## Usage

```python
from geopympm.io import write_particles_vtu

write_particles_vtu(particles, "results/particles_000100.vtu", time=0.1, step=100)
```

`particles` is a `ParticleSet` from a completed solver step. To reduce output,
choose `fields=("material_id", "velocity", "stress", "von_mises")`. By default
all universal fields are included. Existing files are not overwritten unless
`overwrite=True`.

**Demonstration:**

```bash
source .venv/bin/activate
python examples/17_particle_vtu_export.py
python -m validation.stage10g_a1.validate_particle_vtu
python -m pytest tests/test_stage10g_a1_vtk_writer.py -q
```

Open a `.vtu` output directly in VisIt or ParaView. The demonstration produces
six independently readable snapshots; proper `.visit` / `.pvd` time collection
and automatic output interval are planned for Stage 10G-A3. The VTK reference
reader test is optional (`vtk` Python package); if unavailable, structural XML
tests still run. The VisIt GUI itself was not exercised in this environment.

## Field schema and conventions

VTK `Points`: particle current coordinates (in metres), vectors padded to
3 components for 2D. Each `PointData` field has one tuple per particle.

| Field | Components | Interpretation |
|---|---:|---|
| `particle_id`, `material_id` | 1 | Index in current ParticleSet; nonnegative material label |
| `position0`, `displacement`, `velocity` | 3 | Vectors, z=0 for 2D |
| `mass`, `volume`, `volume0`, `density` | 1 | SI units kg, m³, m³, kg/m³ (2D with consistent thickness) |
| `stress_voigt`, `strain_voigt` | 6 | `[xx, yy, zz, xy, yz, xz]` (strain shear *engineering*) |
| `stress`, `strain` | 9 | Full 3×3 tensors (VTK row-major); strain off-diagonal = engineering gamma/2 |
| `deformation_gradient` | 9 | Full 3×3 F, 2D plane strain embedded with Fzz=1 |
| `jacobian` | 1 | det(F) |
| `domain_size`, `domain_size0` | 3 | Full axis-aligned domain sizes padded with z=0; CPDI parallelogram corners **not** represented here |
| `mean_stress`, `pressure` | 1 | mean = tr(stress)/3, pressure = −mean (compression positive) |
| `von_mises` | 1 | 3D equivalent Cauchy stress from the six Voigt components |

Core Cauchy stress is **tension positive**, and values are in **Pa** (not kPa).
For 2D plane strain, sigma_zz can be nonzero and is preserved. Stored strain
is the existing solver field, not a newly established general finite-strain
measure; VTK conversion does not change its physical interpretation.

Global VTK `FieldData` contains `TimeValue` in seconds and `Step` as an integer
when supplied. For VTK XML reader compatibility it is placed under
`UnstructuredGrid` (not `Piece`) and has `NumberOfTuples=1`.

## Verification and limitations

- Tests inspect actual XML, vertex-cell connectivity, selected fields, data
  types, 2D/3D tensor indices, engineering shear, time metadata, and float64
  round trip (and verify that writing does not mutate ParticleSet).
- Independent `vtkXMLUnstructuredGridReader` reads the produced files,
  retrieves 20 arrays and exposes the time metadata.
- No file-series manifests (`.visit`, `.pvd`), background-grid writer, CPDI
  domain mesh output, restart/checkpoint, compressing, or writer integration
  into the simulation loop yet. All are future Stage 10G items.
- Initial ASCII writer assembles the XML document in memory; benchmark I/O
  memory and speed before using millions of particles. It is also not an
  atomic/checksum-protected persistent archive, so do not use it for recovery.

## Change log

Stage 10G-A1 adds only I/O, example, tests, validation, and docs. Existing
particle/grid state, interpolation, constitutive models, and explicit solver
are unchanged.
