"""Stage 10G-B2 restart invariance and strict output-resumption tests."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.io import (
    OutputManager, inspect_checkpoint, load_checkpoint, restore_output_manager,
    write_checkpoint,
)
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver


def make_solver(method="cpdi", *, mapped=False, ext=True, domains=True,
                advection="euler"):
    coords = np.array([[.625, .625], [.875, .625],
                       [.625, .875], [.875, .875]])
    p = ParticleSet(coords, np.ones(4), np.full(4, .0625),
                    velocity=np.tile((.03, -.01), (4, 1)),
                    material_id=np.array([0, 1, 1, 0], dtype=np.int32) if mapped else None,
                    domain_size=np.full((4, 2), .2) if domains else None)
    p.stress[:, 0] = -15.
    g = BackgroundGrid((0., 0.), (.25, .25), (8, 8))
    g.velocity_bc_mask[0, 0] = True
    g.velocity_bc_value[0, 0] = .25
    materials = ({0: LinearElastic(1000., .25), 1: LinearElastic(1500., .28)}
                 if mapped else LinearElastic(1000., .25))
    force = np.zeros((g.n_nodes, 2))
    force[g.node_id(3, 3), 0] = .01
    return ExplicitMPMSolver(
        p, g, materials, dt=.001, interpolation=method,
        pic_fraction=0., gravity=(0., -.02),
        nodal_external_force=force if ext else None,
        stress_update="corotational", kinematics_update="exponential",
        advection_update=advection,
    )


def assert_identical(a, b):
    assert a.step_count == b.step_count
    assert a.time == b.time
    assert a.materials == b.materials
    assert a.interpolation == b.interpolation
    assert a.stress_update == b.stress_update
    assert a.kinematics_update == b.kinematics_update
    assert a.advection_update == b.advection_update
    assert a.dt == b.dt
    assert a.pic_fraction == b.pic_fraction
    assert a.cfl == b.cfl
    for k, v in vars(a.particles).items():
        np.testing.assert_array_equal(v, getattr(b.particles, k), err_msg=f"Particle state {k}")
        assert not np.shares_memory(v, getattr(b.particles, k))
    for k, v in vars(a.grid).items():
        other = getattr(b.grid, k)
        if isinstance(v, np.ndarray):
            np.testing.assert_array_equal(v, other, err_msg=f"Grid state {k}")
            assert not np.shares_memory(v, other)
        else:
            assert v == other
    np.testing.assert_array_equal(a.gravity, b.gravity)
    if a.nodal_external_force is None:
        assert b.nodal_external_force is None
    else:
        np.testing.assert_array_equal(a.nodal_external_force, b.nodal_external_force)
        assert not np.shares_memory(a.nodal_external_force, b.nodal_external_force)


@pytest.mark.parametrize("method", ["mpm", "gimp", "cpgimp", "cpdi"])
def test_continuous_vs_checkpoint_restart_exact(tmp_path, method):
    original = make_solver(method, mapped=True, advection="affine_exponential")
    for _ in range(4):
        original.step()
    record = write_checkpoint(original, tmp_path / "checkpoints")
    loaded = load_checkpoint(record.manifest_path)
    assert_identical(original, loaded)
    for _ in range(5):
        a, b = original.step(), loaded.step()
        assert a == b  # all diagnostics, not just final state
        assert_identical(original, loaded)


@pytest.mark.parametrize("method", ["mpm", "gimp", "cpgimp", "cpdi"])
def test_initial_and_no_force_round_trip(tmp_path, method):
    s = make_solver(method, ext=False)
    record = write_checkpoint(s, tmp_path)
    loaded = load_checkpoint(record.manifest_path)
    assert_identical(s, loaded)
    for _ in range(2):
        assert s.step() == loaded.step()


def test_point_mpm_without_particle_domains_round_trip(tmp_path):
    s = make_solver("mpm", domains=False)
    s.step()
    record = write_checkpoint(s, tmp_path)
    restored = load_checkpoint(record.manifest_path)
    assert not restored.particles.has_gimp_domains
    assert_identical(s, restored)
    assert s.step() == restored.step()


def test_reloaded_checkpoint_can_be_checkpointed_again(tmp_path):
    s = make_solver("cpdi")
    s.run(4)
    a = write_checkpoint(s, tmp_path / "first")
    bsolver = load_checkpoint(a.manifest_path)
    b = write_checkpoint(bsolver, tmp_path / "second")
    left, right = inspect_checkpoint(a.manifest_path), inspect_checkpoint(b.manifest_path)
    for key in ("step", "time", "config", "materials", "grid", "n_particles", "ndim", "arrays"):
        assert left[key] == right[key]
    with np.load(a.data_path, allow_pickle=False) as data_a, np.load(b.data_path, allow_pickle=False) as data_b:
        assert set(data_a.files) == set(data_b.files)
        for key in data_a.files:
            np.testing.assert_array_equal(data_a[key], data_b[key])


def test_original_b1_checkpoint_compatibility(tmp_path):
    s = make_solver("cpdi")
    s.run(3)
    record = write_checkpoint(s, tmp_path / "ckpt")
    assert inspect_checkpoint(record.manifest_path)["schema_version"] == 1
    reloaded = load_checkpoint(record.manifest_path)
    assert_identical(s, reloaded)


def _output(solver, folder):
    manager = OutputManager(folder, every_n_steps=2, every_dt=.0015,
                            particles=True, grid=True,
                            cpdi_domains=(solver.interpolation == "cpdi"))
    manager.write_initial(solver)
    return manager


@pytest.mark.parametrize("method", ["mpm", "gimp", "cpgimp", "cpdi"])
def test_output_manager_restart_exact_series(tmp_path, method):
    direct = make_solver(method)
    resumed_base = make_solver(method)
    direct_out = _output(direct, tmp_path / "continuous")
    resumed_out = _output(resumed_base, tmp_path / "split")
    for _ in range(3):
        a, b = direct.step(), resumed_base.step()
        direct_out.on_step_completed(direct, a)
        resumed_out.on_step_completed(resumed_base, b)
    record = write_checkpoint(resumed_base, tmp_path / "checkpoints", output_manager=resumed_out)
    restored = load_checkpoint(record.manifest_path)
    resumed_out = restore_output_manager(record.manifest_path, restored)
    assert direct_out.frames == resumed_out.frames
    assert resumed_out._next_time == direct_out._next_time
    for _ in range(6):
        d1, d2 = direct.step(), restored.step()
        assert d1 == d2
        direct_out.on_step_completed(direct, d1)
        resumed_out.on_step_completed(restored, d2)
        assert direct_out.frames == resumed_out.frames
    assert_identical(direct, restored)
    for kind in direct_out.enabled:
        for ext in ("visit", "pvd"):
            assert (direct_out.directory / f"{kind}.{ext}").read_bytes() == (resumed_out.directory / f"{kind}.{ext}").read_bytes()


def test_output_directory_copy_override(tmp_path):
    s = make_solver("cpdi")
    o = _output(s, tmp_path / "original")
    for _ in range(3):
        d = s.step()
        o.on_step_completed(s, d)
    chk = write_checkpoint(s, tmp_path / "ckpt", output_manager=o)
    shutil.copytree(o.directory, tmp_path / "branch")
    fresh = load_checkpoint(chk.manifest_path)
    resumed = restore_output_manager(chk.manifest_path, fresh, directory=tmp_path / "branch")
    assert resumed.directory == tmp_path / "branch"
    assert resumed.frames == o.frames
    resumed.on_step_completed(fresh, fresh.step())
    assert (tmp_path / "branch" / "particles_000004.vtu").exists()
    assert not (tmp_path / "original" / "particles_000004.vtu").exists()


def test_missing_output_manager_data_refused(tmp_path):
    s = make_solver()
    c = write_checkpoint(s, tmp_path)
    with pytest.raises(ValueError, match="does not contain"):
        restore_output_manager(c.manifest_path, load_checkpoint(c.manifest_path))


def test_fork_old_checkpoint_refuses_newer_visualization(tmp_path):
    s = make_solver()
    o = _output(s, tmp_path / "viz")
    for _ in range(3):
        o.on_step_completed(s, s.step())
    c = write_checkpoint(s, tmp_path / "ckpt", output_manager=o)
    o.on_step_completed(s, s.step())  # later output written after checkpoint
    with pytest.raises(ValueError, match="extra/missing|does not match"):
        restore_output_manager(c.manifest_path, load_checkpoint(c.manifest_path))


def test_output_index_tampering_is_rejected(tmp_path):
    s = make_solver()
    o = _output(s, tmp_path / "viz")
    o.on_step_completed(s, s.step())
    c = write_checkpoint(s, tmp_path / "ckpt", output_manager=o)
    (o.directory / "particles.visit").write_text("!NBLOCKS 1\n")
    with pytest.raises(ValueError, match="does not match"):
        restore_output_manager(c.manifest_path, load_checkpoint(c.manifest_path))


def test_missing_vtu_is_rejected(tmp_path):
    s = make_solver()
    o = _output(s, tmp_path / "viz")
    o.on_step_completed(s, s.step())
    c = write_checkpoint(s, tmp_path / "ckpt", output_manager=o)
    (o.directory / "particles_000000.vtu").unlink()
    with pytest.raises(ValueError, match="extra/missing"):
        restore_output_manager(c.manifest_path, load_checkpoint(c.manifest_path))


def test_loader_refuses_corrupt_archive(tmp_path):
    s = make_solver()
    c = write_checkpoint(s, tmp_path)
    with c.data_path.open("ab") as f:
        f.write(b"INVALID")
    with pytest.raises(ValueError, match="SHA-256"):
        load_checkpoint(c.manifest_path)


def test_loader_refuses_unsupported_schema(tmp_path):
    c = write_checkpoint(make_solver(), tmp_path)
    info = inspect_checkpoint(c.manifest_path)
    info["schema_version"] = 2
    c.manifest_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="version"):
        load_checkpoint(c.manifest_path)


def test_loader_refuses_unregistered_constitutive_model(tmp_path):
    c = write_checkpoint(make_solver(), tmp_path)
    info = inspect_checkpoint(c.manifest_path)
    info["materials"]["model"] = "MohrCoulomb"
    c.manifest_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="LinearElastic"):
        load_checkpoint(c.manifest_path)


def test_loader_refuses_invalid_grid_metadata(tmp_path):
    c = write_checkpoint(make_solver(), tmp_path)
    info = inspect_checkpoint(c.manifest_path)
    info["grid"]["n_cells"] = [3, 3]
    c.manifest_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="node counts"):
        load_checkpoint(c.manifest_path)


def test_loader_refuses_invalid_numeric_state_dtype(tmp_path):
    # A manifest may describe even a hash-consistent archive with an
    # unsupported dtype. The loader must not silently cast it.
    import hashlib
    c = write_checkpoint(make_solver(), tmp_path)
    meta = inspect_checkpoint(c.manifest_path)
    with np.load(c.data_path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays["particle.velocity"] = arrays["particle.velocity"].astype("float32")
    with c.data_path.open("wb") as f:
        np.savez_compressed(f, **arrays)
    meta["data_sha256"] = hashlib.sha256(c.data_path.read_bytes()).hexdigest()
    meta["arrays"]["particle.velocity"]["dtype"] = arrays["particle.velocity"].dtype.str
    c.manifest_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="expected.*float64"):
        load_checkpoint(c.manifest_path)


def test_no_mutation_or_write_on_restore(tmp_path):
    s = make_solver()
    o = _output(s, tmp_path / "viz")
    for _ in range(3):
        o.on_step_completed(s, s.step())
    c = write_checkpoint(s, tmp_path / "ckpt", output_manager=o)
    files = [x for x in tmp_path.rglob("*") if x.is_file()]
    before = {x: x.read_bytes() for x in files}
    recovered = load_checkpoint(c.manifest_path)
    restore_output_manager(c.manifest_path, recovered)
    assert {x: x.read_bytes() for x in files} == before
    assert_identical(s, recovered)


def test_output_manager_before_first_output_can_resume(tmp_path):
    s = make_solver("mpm", domains=False)
    # OutputManager was configured but has never written a snapshot.
    pending = tmp_path / "pending_visualization"
    out = OutputManager(pending, every_n_steps=10, particles=True, grid=False)
    for _ in range(2):
        out.on_step_completed(s, s.step())
    assert not pending.exists()
    c = write_checkpoint(s, tmp_path / "checkpoints", output_manager=out)
    restored = load_checkpoint(c.manifest_path)
    resumed = restore_output_manager(c.manifest_path, restored)
    assert resumed.frames["particles"] == ()
    assert not pending.exists()
    for _ in range(8):
        resumed.on_step_completed(restored, restored.step())
    assert (pending / "particles_000010.vtu").exists()


def test_solver_clock_mismatch_refused_for_output_resume(tmp_path):
    s = make_solver()
    o = _output(s, tmp_path / "viz")
    o.on_step_completed(s, s.step())
    c = write_checkpoint(s, tmp_path / "ckpt", output_manager=o)
    restored = load_checkpoint(c.manifest_path)
    restored.step()
    with pytest.raises(ValueError, match="checkpoint step and time"):
        restore_output_manager(c.manifest_path, restored)


def test_output_manager_wrong_interpolation_refused(tmp_path):
    s = make_solver("cpdi")
    o = _output(s, tmp_path / "viz")
    c = write_checkpoint(s, tmp_path / "ckpt", output_manager=o)
    loaded = load_checkpoint(c.manifest_path)
    loaded.interpolation = "mpm"
    with pytest.raises(ValueError, match="CPDI domain output"):
        restore_output_manager(c.manifest_path, loaded)
