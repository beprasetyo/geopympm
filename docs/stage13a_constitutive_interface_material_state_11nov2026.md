# GeoPyMPM Stage 13A — Constitutive Interface & Material State Management

**Date:** 11 November 2026  
**Required baseline:** Stage 12 (Stage 11 CNH + USF, USL-like, MUSL patch, plus Stage 10G-D4).  
**Delivery:** incremental ZIP with only added/modified files.

## 1. Objective and boundaries

Stage 13A introduces an explicit **constitutive-dispatch contract** and a
**per-particle model history store**. This is groundwork for Stage 13B (Mohr–Coulomb)
and later finite-strain elasto-plastic constitutive models, not an implementation
of plasticity itself. The existing numerical formulation, P2G/G2P, and time
integration schemes are unchanged. Existing `LinearElastic` and
`CompressibleNeoHookean` are still the only supported constitutive families.

**The current material laws do not update user-declared history fields.**
For example, declaring `plastic_strain` does not activate a plastic flow rule.

## 2. Design: state ownership

- `ParticleSet`: universal kinematics and stress (position, velocity, stress,
  strain, `F`, volume, mass, material ID, interpolation domains).
- `MaterialState`: **model-specific arrays** stored outside `ParticleSet`.
  Data are `float64`, C-contiguous, owned copies, indexed by particle first.
  Scalars have shape `(n,)`; Voigt tensors `(n, 6)`; tensor histories `(n,2,2)`.
- `StateFieldSpec`: immutable declaration with validated lowercase name and
  component shape.
- `ConstitutiveAdapter`: model-level contract with supported stress modes,
  declared required history, wave moduli, and whole-array stress update.
  `resolve_constitutive_adapter` provides adapters for the two built-in families.
- `ExplicitMPMSolver`: owns a `material_state` attribute; optional in constructor.
  Default is an empty `MaterialState`, preserving old behavior. It calls the
  adapter for CFL wave moduli and stress update and includes history in rollback.

Adapters operate at the **whole-particle array** level (not one Python material
object per particle); hot loops remain in the pre-existing numerical kernels.
Stage 13A intentionally does not add third-party runtime model registration,
heterogeneous LinearElastic/CNH families, or dynamic allocation per step.

### Example

```python
from geopympm.materials import MaterialState, StateFieldSpec

state = MaterialState(n_particles, [
    StateFieldSpec("plastic_strain", (6,)),
    StateFieldSpec("hardening_parameter"),
])
# ... existing ParticleSet, BackgroundGrid, LinearElastic ...
solver = ExplicitMPMSolver(particles, grid, materials, dt=0.0005,
                           material_state=state)
```

History names/array shapes are fixed for the lifetime of the solver; values
are mutable and may be updated in-place by a future material kernel.
Users must keep all values finite and shapes unchanged; the solver validates
history before and after stepping. On a failed substep it restores the original
history values **without replacing ndarray identities**.

## 3. Checkpoint/restart compatibility

Stage 13A introduces **checkpoint schema 4 only when history fields exist**.
Schema 4 adds:

- `material_state`: array of `{name, component_shape, dtype}` declarations;
- NPZ arrays named `history.<name>` with exact `float64` shape `(n,*shape)`;
- mandatory `config.scheme` even if scheme is `usl`;
- full inclusion of history arrays in NPZ SHA-256 and array metadata validation.

All legacy histories-free checkpoints continue to use the same schema:

| Baseline configuration | Schema |
|---|---:|
| LinearElastic, USL-like, no history | 1 |
| CNH, USL-like, no history | 2 |
| Either, USF/MUSL, no history | 3 |
| Either family, any scheme, with history | **4** |

`inspect_checkpoint` and `load_checkpoint` reject invalid or incomplete history
metadata, unexpected fields, incorrect shapes/dtypes and corrupted NPZ contents.
`recover_latest_checkpoint` works through these readers and can fallback from
an invalid newest checkpoint, subject to the earlier VTU output constraints.

Schema 4 is a **state-storage** upgrade. Constitutive model parameters remain
serialized by the pre-existing explicit LinearElastic/CNH serializers. A future
plastic model will need a registered model-parameter serializer and a matching
restart constructor; unknown models must continue to fail closed.

## 4. Safety and compatibility decisions

- No plastic return mapping, yield surface, hardening integration or stress softening.
- No mixing LinearElastic and CNH within the same solver; multiple IDs with
  different parameter sets inside a single homogeneous family still work.
- No constitutive VTK field output yet: this will need explicit output selection
  and field metadata in a later stage.
- No change in plane-strain tensor, tension-positive stress sign, SI units,
  GIMP/CPDI interpolation, or MUSL momentum remapping.
- No change to historical checkpoint schema 1–3 or Stage 10G-D2 VTU integrity.
- Material history is independent of `ParticleSet`, avoiding universal storage
  of every model's internal variables.

## 5. Verification

Run after applying patch to the Stage 12 repository root:

```bash
source .venv/bin/activate
python -m pytest -q tests/test_stage13a_material_interface.py
python -m pytest -q tests/test_stage11_cnh.py tests/test_stage12_integration_schemes.py
python examples/26_material_state_checkpoint.py
```

Tests include invalid history declarations, deep-copy ownership, restoration of
array identity, rollback after both raised exceptions and nonfinite updates,
all three schemes with both current families, exact chained restart, schema
1/3 compatibility, and manifest/content corruption detection.

**Audit snapshot:** 31 Stage 13A tests passed; 272 checkpoint/restart and
constitutive regression tests passed in the controlled sandbox. Independent
Stage 12 vs Stage 13A comparison of 24 combinations (2 materials × 3 schemes
× 4 interpolation methods, 4 time steps each) produced identical SHA-256
checksums of relevant numerical arrays in all 24 cases. These numbers
describe only tests actually executed, not the entire GeoPyMPM repository.

## 6. Stage 13B prerequisites

1. Formally define the constitutive stress and strain measures, tension-positive
   convention, plane strain and the treatment of out-of-plane stress.
2. Implement Mohr–Coulomb plane-strain trial stress and return mapping with
   explicit yield/flow potentials and apex/edge handling.
3. Declare required material history (`plastic_strain`, equivalent plastic
   strain, optional hardening variables); do not infer state from total strain.
4. Extend `resolve_constitutive_adapter`, material validation, wave-speed
   screening, and checkpoint material serialization to new material family.
5. Add single-material-point stress path tests *before* full MPM runout.
6. Test all USF/USL/MUSL and MPM/uGIMP/cpGIMP/CPDI combinations as applicable.

### Relationship to the five project references

Uintah informs geomechanical constitutive and contact architecture; MPM2D's
DrySand is a useful example of storing `F^e`, `F^p` and hardening values (not
an exact Mohr–Coulomb source); MPM3D-F90 and MPM-Py inform explicit MPM
stepping and interpolation; GeoPyFEM will eventually provide compatible
initial state transfer. These are architectural comparisons, **not claims of
feature parity or code translation**.
