"""Stage 10G-B1: full-state, versioned, non-pickle checkpoint writer tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.io import OutputManager, inspect_checkpoint, write_checkpoint
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver


def make_solver(interpolation="cpdi", mapped=False):
    coords = np.array([[0.625, 0.625], [0.875, 0.625],
                       [0.625, 0.875], [0.875, 0.875]], dtype=np.float64)
    ids = np.array([0, 1, 1, 0], dtype=np.int32) if mapped else None
    p = ParticleSet(coords, np.ones(4), np.full(4, .0625),
                    material_id=ids, domain_size=np.full((4, 2), .2),
                    velocity=np.tile((.03, -.01), (4, 1)))
    p.stress[:, 0] = -15.
    g = BackgroundGrid((0., 0.), (.25, .25), (8, 8))
    g.velocity_bc_mask[0, 0] = True
    g.velocity_bc_value[0, 0] = .25
    m = ({0: LinearElastic(1000., .25), 1: LinearElastic(1500., .28)}
         if mapped else LinearElastic(1000., .25))
    ext = np.zeros((g.n_nodes, 2))
    ext[g.node_id(3, 3), 0] = .01
    s = ExplicitMPMSolver(p, g, m, dt=.001, interpolation=interpolation,
                          pic_fraction=0., gravity=(0., -.02),
                          nodal_external_force=ext,
                          stress_update="corotational",
                          kinematics_update="exponential")
    return s


def assert_arrays_equal(s, path):
    with np.load(path, allow_pickle=False) as data:
        for obj, prefix in ((s.particles, "particle"), (s.grid, "grid")):
            for name, val in vars(obj).items():
                if isinstance(val, np.ndarray):
                    np.testing.assert_array_equal(data[f"{prefix}.{name}"], val)
        np.testing.assert_array_equal(data["solver.gravity"], s.gravity)
        np.testing.assert_array_equal(data["solver.nodal_external_force"], s.nodal_external_force)
        assert "particle.position0" in data
        assert "particle.volume0" in data
        assert "particle.domain_size0" in data
        assert "grid.velocity_bc_mask" in data
        assert "grid.force_internal" in data
        assert all(data[k].dtype.kind != "O" for k in data.files)


@pytest.mark.parametrize("interpolation", ["mpm", "gimp", "cpgimp", "cpdi"])
def test_four_interpolations_full_state_round_trip(tmp_path, interpolation):
    s = make_solver(interpolation)
    s.step()
    old_particle = s.particles.position.copy()
    old_grid = s.grid.momentum.copy()
    record = write_checkpoint(s, tmp_path)
    assert record.array_count == 25  # 12 particle + 11 grid + 2 solver
    assert record.step == 1
    assert record.data_path.name == "state_00000001.npz"
    assert record.manifest_path.name == "state_00000001.json"
    info = inspect_checkpoint(record.manifest_path)
    assert info["format"] == "geopympm-checkpoint"
    assert info["schema_version"] == 1
    assert info["step"] == 1
    assert info["time"] == pytest.approx(.001)
    assert info["config"]["interpolation"] == interpolation
    assert info["config"]["has_nodal_external_force"] is True
    assert info["grid"]["n_cells"] == [8, 8]
    assert_arrays_equal(s, record.data_path)
    np.testing.assert_array_equal(s.particles.position, old_particle)
    np.testing.assert_array_equal(s.grid.momentum, old_grid)
    assert record.bytes_written > 0


def test_initial_checkpoint_and_uniform_material(tmp_path):
    s = make_solver()
    s.nodal_external_force = None
    record = write_checkpoint(s, tmp_path)
    meta = inspect_checkpoint(record.manifest_path)
    assert meta["step"] == 0
    assert meta["time"] == 0.
    assert meta["materials"]["kind"] == "single"
    assert not meta["config"]["has_nodal_external_force"]
    with np.load(record.data_path, allow_pickle=False) as archive:
        assert "solver.nodal_external_force" not in archive.files


def test_multi_material_mapping_preserved(tmp_path):
    s = make_solver(mapped=True)
    record = write_checkpoint(s, tmp_path)
    meta = inspect_checkpoint(record.manifest_path)
    assert meta["materials"]["kind"] == "mapping"
    assert [v["id"] for v in meta["materials"]["items"]] == [0, 1]
    assert meta["materials"]["items"][1]["parameters"]["young_modulus"] == 1500.
    assert_arrays_equal(s, record.data_path)


def test_no_overwrite_or_mutation(tmp_path):
    s = make_solver()
    first = write_checkpoint(s, tmp_path)
    old = first.data_path.read_bytes()
    with pytest.raises(FileExistsError):
        write_checkpoint(s, tmp_path)
    assert first.data_path.read_bytes() == old
    assert inspect_checkpoint(first.manifest_path)["step"] == 0


def test_unique_prefix_and_multiple_steps(tmp_path):
    s = make_solver()
    a = write_checkpoint(s, tmp_path, prefix="snapshot")
    s.step()
    b = write_checkpoint(s, tmp_path, prefix="snapshot")
    assert a.manifest_path != b.manifest_path
    assert inspect_checkpoint(a.manifest_path)["step"] == 0
    assert inspect_checkpoint(b.manifest_path)["step"] == 1
    with pytest.raises(ValueError, match="prefix"):
        write_checkpoint(s, tmp_path, prefix="../../evil")


def test_corrupt_npz_detected(tmp_path):
    s = make_solver()
    record = write_checkpoint(s, tmp_path)
    with record.data_path.open("ab") as f:
        f.write(b"INVALID")
    with pytest.raises(ValueError, match="SHA-256"):
        inspect_checkpoint(record.manifest_path)
    assert inspect_checkpoint(record.manifest_path, verify=False)["step"] == 0


def test_missing_npz_detected(tmp_path):
    record = write_checkpoint(make_solver(), tmp_path)
    record.data_path.unlink()
    with pytest.raises(FileNotFoundError):
        inspect_checkpoint(record.manifest_path)


def test_invalid_manifest_rejected(tmp_path):
    record = write_checkpoint(make_solver(), tmp_path)
    meta = inspect_checkpoint(record.manifest_path)
    meta["schema_version"] = 999
    record.manifest_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="version"):
        inspect_checkpoint(record.manifest_path)


def test_manifest_traversal_rejected(tmp_path):
    record = write_checkpoint(make_solver(), tmp_path)
    meta = inspect_checkpoint(record.manifest_path)
    meta["data_file"] = "../somewhere.npz"
    record.manifest_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="unsafe"):
        inspect_checkpoint(record.manifest_path)


def test_unregistered_particle_state_fails_closed(tmp_path):
    s = make_solver()
    s.particles.plastic_strain = np.zeros(4)
    with pytest.raises(ValueError, match="Unregistered ParticleSet"):
        write_checkpoint(s, tmp_path)
    assert not list(tmp_path.iterdir())


def test_unregistered_solver_state_fails_closed(tmp_path):
    s = make_solver()
    s.history = np.zeros(4)
    with pytest.raises(ValueError, match="Unregistered ExplicitMPMSolver"):
        write_checkpoint(s, tmp_path)
    assert not list(tmp_path.iterdir())


def test_nan_rejected_before_writing(tmp_path):
    s = make_solver()
    s.particles.stress[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        write_checkpoint(s, tmp_path)
    assert not list(tmp_path.iterdir())


def test_output_manager_scheduler_saved(tmp_path):
    s = make_solver("cpdi")
    out = OutputManager(tmp_path / "viz", every_n_steps=2, every_dt=.0015,
                        particles=True, grid=False, cpdi_domains=True)
    out.write_initial(s)
    for _ in range(3):
        d = s.step()
        out.on_step_completed(s, d)
    record = write_checkpoint(s, tmp_path / "checkpoint", output_manager=out)
    meta = inspect_checkpoint(record.manifest_path)
    state = meta["output_manager"]
    assert state["last_seen_step"] == 3
    assert state["last_seen_time"] == s.time
    assert state["enabled"] == ["particles", "domains"]
    assert state["every_dt"] == .0015
    assert len(state["entries"]["particles"]) >= 2
    assert state["entries"]["domains"][-1]["step"] == 3


def test_stale_output_manager_rejected(tmp_path):
    s = make_solver()
    out = OutputManager(tmp_path / "viz", every_n_steps=1)
    out.write_initial(s)
    s.step()
    with pytest.raises(ValueError, match="observed"):
        write_checkpoint(s, tmp_path / "checkpoint", output_manager=out)
    assert not (tmp_path / "checkpoint").exists()


def test_json_publish_failure_cleans_npz(tmp_path, monkeypatch):
    import geopympm.io.checkpoint as checkpoint_module
    s = make_solver()
    real_replace = checkpoint_module.os.replace

    def fail_json(src, dst):
        if str(dst).endswith(".json"):
            raise OSError("disk failure")
        return real_replace(src, dst)

    monkeypatch.setattr(checkpoint_module.os, "replace", fail_json)
    with pytest.raises(OSError, match="disk failure"):
        write_checkpoint(s, tmp_path)
    assert not list(tmp_path.iterdir())


def test_invalid_material_rejected_without_side_effect(tmp_path):
    s = make_solver()
    s.materials = {0: object()}
    with pytest.raises(TypeError, match="LinearElastic"):
        write_checkpoint(s, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("interpolation", ["mpm", "gimp", "cpgimp", "cpdi"])
def test_checkpoint_does_not_change_future_trajectory(tmp_path, interpolation):
    a = make_solver(interpolation)
    b = make_solver(interpolation)
    for _ in range(2):
        a.step()
        b.step()
    write_checkpoint(a, tmp_path)
    for _ in range(3):
        a.step()
        b.step()
    assert a.time == b.time
    assert a.step_count == b.step_count
    for name in vars(a.particles):
        np.testing.assert_array_equal(getattr(a.particles, name),
                                      getattr(b.particles, name))
    for name, values in vars(a.grid).items():
        if isinstance(values, np.ndarray):
            np.testing.assert_array_equal(values, getattr(b.grid, name))


def test_unregistered_grid_state_fails_closed(tmp_path):
    s = make_solver()
    s.grid.extra_grid_state = np.zeros(s.grid.n_nodes)
    with pytest.raises(ValueError, match="Unregistered BackgroundGrid"):
        write_checkpoint(s, tmp_path)
    assert not list(tmp_path.iterdir())


def test_bad_particle_array_shape_fails_closed(tmp_path):
    s = make_solver()
    s.particles.stress = s.particles.stress[:, :3].copy()
    with pytest.raises(ValueError, match="stress must have shape"):
        write_checkpoint(s, tmp_path)
    assert not list(tmp_path.iterdir())


def test_integrity_check_rejects_missing_array_even_with_valid_hash(tmp_path):
    import hashlib
    record = write_checkpoint(make_solver(), tmp_path)
    meta = inspect_checkpoint(record.manifest_path)
    with np.load(record.data_path, allow_pickle=False) as src:
        subset = {key: src[key] for key in src.files if key != "particle.position0"}
    np.savez_compressed(record.data_path, **subset)
    meta["data_sha256"] = hashlib.sha256(record.data_path.read_bytes()).hexdigest()
    record.manifest_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="array list"):
        inspect_checkpoint(record.manifest_path)


def test_integrity_check_rejects_mismatched_array_metadata(tmp_path):
    record = write_checkpoint(make_solver(), tmp_path)
    meta = inspect_checkpoint(record.manifest_path)
    meta["arrays"]["particle.position"]["shape"] = [999, 2]
    record.manifest_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="dtype/shape"):
        inspect_checkpoint(record.manifest_path)
