"""Stage 13A: constitutive adapter, independent history, transactional restart."""
from __future__ import annotations

import json
import numpy as np
import pytest

from geopympm.core import ParticleSet, BackgroundGrid
from geopympm.materials import (
    LinearElastic, CompressibleNeoHookean, MaterialState, StateFieldSpec,
    resolve_constitutive_adapter,
)
from geopympm.io import write_checkpoint, load_checkpoint, inspect_checkpoint
from geopympm.solvers import ExplicitMPMSolver


def make_solver(*, family="linear", scheme="usl", state=None):
    p = ParticleSet([[0.35, 0.42], [0.66, 0.57]], [0.1, 0.1], [0.002, 0.002],
                    velocity=[[0.02, -0.01], [-0.02, 0.01]])
    g = BackgroundGrid([0, 0], [0.1, 0.1], (10, 10))
    mat = (LinearElastic(1000, 0.3) if family == "linear" else
           CompressibleNeoHookean.from_young_poisson(1000, 0.3))
    return ExplicitMPMSolver(
        p, g, mat, dt=0.0005, pic_fraction=0.0,
        stress_update="incremental" if family == "linear" else "hyperelastic",
        kinematics_update="exponential", scheme=scheme,
        material_state=state,
    )


def sample_state():
    return MaterialState(2, [
        StateFieldSpec("equivalent_plastic_strain"),
        StateFieldSpec("plastic_strain", (6,)),
        StateFieldSpec("plastic_deformation_gradient", (2, 2)),
    ])


def test_empty_material_state_keeps_old_checkpoint_schema(tmp_path):
    s = make_solver()
    assert s.material_state.names == ()
    r = write_checkpoint(s, tmp_path)
    meta = inspect_checkpoint(r.manifest_path)
    assert meta["schema_version"] == 1
    assert "material_state" not in meta
    assert not any(k.startswith("history.") for k in meta["arrays"])
    assert load_checkpoint(r.manifest_path).material_state.names == ()


@pytest.mark.parametrize("name,shape", [
    ("x-y", ()), ("Upper", ()), ("foo.bar", ()), ("1foo", ()),
    ("name", (-1,)), ("name", (2.0,)), ("name", (True,)),
    ("name", [2]), ("name", (1, 1, 1, 1)),
])
def test_invalid_field_spec(name, shape):
    with pytest.raises(ValueError):
        StateFieldSpec(name, shape)


def test_invalid_duplicates_and_particle_count():
    with pytest.raises(ValueError, match="Duplicate"):
        MaterialState(2, [StateFieldSpec("a"), StateFieldSpec("a")])
    with pytest.raises(ValueError, match="n_particles"):
        MaterialState(0)
    with pytest.raises(ValueError, match="particle count"):
        make_solver(state=MaterialState(3, [StateFieldSpec("a")]))


def test_no_aliasing_and_zero_initialization():
    initial = np.array([[1., 2.], [3., 4.]])
    state = MaterialState(2, [StateFieldSpec("alpha", (2,))], values={"alpha": initial})
    initial[0, 0] = 99.
    assert state["alpha"][0, 0] == 1.
    assert state["alpha"].flags.c_contiguous
    s = sample_state()
    for field in s.fields:
        assert np.all(s[field.name] == 0.)


def test_requires_exact_finite_float64_fields():
    f = StateFieldSpec("q", (2,))
    with pytest.raises(ValueError, match="float64"):
        MaterialState(2, [f], values={"q": np.ones((2, 2), dtype=np.float32)})
    with pytest.raises(ValueError, match="finite"):
        MaterialState(2, [f], values={"q": np.full((2, 2), np.inf)})
    s = MaterialState(2, [f])
    s["q"][1, 0] = np.nan
    with pytest.raises(ValueError, match="Invalid material state"):
        s.validate()


def test_snapshot_restore_keeps_ndarray_identity():
    s = sample_state()
    s["plastic_strain"][:] = 2.3
    alias = s["plastic_strain"]
    before = s.snapshot()
    s["plastic_strain"][:] = -5.
    s.restore(before)
    assert s["plastic_strain"] is alias
    np.testing.assert_array_equal(s["plastic_strain"], 2.3)


@pytest.mark.parametrize("family", ["linear", "cnh"])
def test_built_in_constitutive_adapter_and_passive_history(family):
    state = sample_state()
    state["equivalent_plastic_strain"][:] = [0.1, 0.2]
    solver = make_solver(family=family, state=state)
    a = resolve_constitutive_adapter(solver.particles, solver.materials)
    assert a.family == family
    assert a.history_schema == ()  # Both current models have no native history.
    assert solver.stress_update in a.modes
    before = state.snapshot()
    solver.run(3)
    for key, old in before.items():
        np.testing.assert_array_equal(state[key], old)


@pytest.mark.parametrize("family", ["linear", "cnh"])
@pytest.mark.parametrize("scheme", ["usl", "usf", "musl"])
def test_schema4_exact_chained_restart_with_history(tmp_path, family, scheme):
    direct_state = sample_state()
    direct_state["equivalent_plastic_strain"][:] = [0.1, 0.2]
    direct_state["plastic_deformation_gradient"][:] = np.eye(2)
    direct = make_solver(family=family, scheme=scheme, state=direct_state)
    direct.run(7)

    chain_state = sample_state()
    chain_state["equivalent_plastic_strain"][:] = [0.1, 0.2]
    chain_state["plastic_deformation_gradient"][:] = np.eye(2)
    chain = make_solver(family=family, scheme=scheme, state=chain_state)
    chain.run(3)
    r = write_checkpoint(chain, tmp_path / f"{family}_{scheme}")
    meta = inspect_checkpoint(r.manifest_path)
    assert meta["schema_version"] == 4
    assert meta["config"]["scheme"] == scheme
    assert len(meta["material_state"]) == 3
    assert "history.plastic_strain" in meta["arrays"]
    continued = load_checkpoint(r.manifest_path)
    assert continued.material_state.names == chain_state.names
    for name in continued.material_state.names:
        np.testing.assert_array_equal(continued.material_state[name], chain_state[name])
    continued.run(4)
    for field in ("position", "velocity", "stress", "strain", "volume", "deformation_gradient"):
        np.testing.assert_array_equal(getattr(continued.particles, field), getattr(direct.particles, field))
    for name in continued.material_state.names:
        np.testing.assert_array_equal(continued.material_state[name], direct_state[name])


def test_state_mutation_rollback_is_transactional(monkeypatch):
    state = sample_state()
    state["equivalent_plastic_strain"][:] = 0.123
    s = make_solver(state=state)
    held = state["equivalent_plastic_strain"]
    old_p = s.particles.position.copy()
    old_stress = s.particles.stress.copy()
    old_grid = s.grid.momentum.copy()
    def faulty(*args):
        state["equivalent_plastic_strain"][:] = 0.999
        s.particles.stress[:] = 999.0
        raise RuntimeError("constitutive failure")
    monkeypatch.setattr(s, "_update_material_stress", faulty)
    with pytest.raises(RuntimeError, match="constitutive failure"):
        s.step()
    assert state["equivalent_plastic_strain"] is held
    np.testing.assert_array_equal(held, 0.123)
    np.testing.assert_array_equal(s.particles.position, old_p)
    np.testing.assert_array_equal(s.particles.stress, old_stress)
    np.testing.assert_array_equal(s.grid.momentum, old_grid)
    assert s.step_count == 0 and s.time == 0


def test_state_nonfinite_after_update_triggers_rollback(monkeypatch):
    state = sample_state()
    s = make_solver(state=state)
    original = s._update_material_stress
    def broken(k):
        original(k)
        state["plastic_strain"][0, 0] = float("nan")
    monkeypatch.setattr(s, "_update_material_stress", broken)
    with pytest.raises(ValueError, match="material state"):
        s.step()
    assert np.all(state["plastic_strain"] == 0)
    assert s.step_count == 0


def test_checkpoint_rejects_nonfinite_history(tmp_path):
    state = sample_state()
    s = make_solver(state=state)
    state["plastic_strain"][0, 0] = float("inf")
    with pytest.raises(ValueError, match="material state"):
        write_checkpoint(s, tmp_path)
    assert not list(tmp_path.glob("*"))


def test_checkpoint_detects_corrupt_history_archive(tmp_path):
    s = make_solver(state=sample_state())
    record = write_checkpoint(s, tmp_path)
    with record.data_path.open("ab") as out:
        out.write(b"corruption")
    with pytest.raises(ValueError, match="SHA-256"):
        load_checkpoint(record.manifest_path)


def test_reject_history_metadata_tamper(tmp_path):
    s = make_solver(state=sample_state())
    record = write_checkpoint(s, tmp_path)
    meta = json.loads(record.manifest_path.read_text())
    meta["material_state"].pop()
    record.manifest_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="required state|unknown array"):
        load_checkpoint(record.manifest_path)


def test_legacy_schema3_is_unchanged(tmp_path):
    solver = make_solver(scheme="musl")
    rec = write_checkpoint(solver, tmp_path)
    assert inspect_checkpoint(rec.manifest_path)["schema_version"] == 3
    loaded = load_checkpoint(rec.manifest_path)
    assert loaded.scheme == "musl"
    assert loaded.material_state.names == ()


def test_schema4_recovery_and_safe_retention(tmp_path):
    from geopympm.io import recover_latest_checkpoint, scan_checkpoints, prune_checkpoints
    s = make_solver(state=sample_state())
    s.step()
    first = write_checkpoint(s, tmp_path)
    s.step()
    newest = write_checkpoint(s, tmp_path)
    assert scan_checkpoints(tmp_path).valid[0] == newest.manifest_path
    newest.data_path.write_bytes(b"incomplete/corrupt")
    result = recover_latest_checkpoint(tmp_path)
    assert result.step == 1
    assert result.manifest_path == first.manifest_path
    assert result.solver.material_state.names == s.material_state.names
    # Retention must not delete a corrupt checkpoint automatically.
    report = prune_checkpoints(tmp_path, keep_last=1, dry_run=False)
    assert newest.data_path.exists()
    assert first.data_path.exists()


def test_schema4_rejects_missing_scheme_metadata(tmp_path):
    s = make_solver(state=sample_state(), scheme="usl")
    rec = write_checkpoint(s, tmp_path)
    meta = json.loads(rec.manifest_path.read_text())
    assert meta["schema_version"] == 4
    del meta["config"]["scheme"]
    rec.manifest_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="scheme"):
        load_checkpoint(rec.manifest_path)


def test_adapter_required_history_schema_validation():
    from geopympm.materials import ConstitutiveAdapter
    a = ConstitutiveAdapter(
        family="future", modes=("incremental",),
        history_schema=(StateFieldSpec("plastic_strain", (6,)),),
        stress_update=lambda *args: None,
        wave_moduli=lambda *args: (None, None),
    )
    with pytest.raises(ValueError, match="plastic_strain"):
        a.validate_state(MaterialState(2))
    a.validate_state(MaterialState(2, [StateFieldSpec("plastic_strain", (6,))]))
