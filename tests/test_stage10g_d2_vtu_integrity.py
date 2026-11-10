"""Stage 10G-D2: per-VTU content integrity bound to checkpoint metadata.

The numerical checkpoint schema stays at version 1.  A separate versioned
OutputManager integrity extension covers every output snapshot.  Tests use
minimal real solver states and real VTK writers, not mocks of hashing.
"""
from __future__ import annotations

import hashlib
import json
import shutil

import pytest

from geopympm.io import (inspect_checkpoint, load_checkpoint,
                         restore_output_manager, write_checkpoint)
from validation.stage10g_c.extended_restart import make_case, _make_output


def _checkpoint(tmp_path, method="mpm", steps=9):
    solver = make_case(method)
    manager = _make_output(solver, tmp_path / "viz")
    for _ in range(steps):
        manager.on_step_completed(solver, solver.step())
    checkpoint = write_checkpoint(solver, tmp_path / "checkpoint", output_manager=manager)
    return solver, manager, checkpoint


def _rewrite_manifest(checkpoint, fn):
    meta = inspect_checkpoint(checkpoint.manifest_path)
    fn(meta)
    checkpoint.manifest_path.write_text(json.dumps(meta), encoding="utf-8")


@pytest.mark.parametrize("method", ["mpm", "gimp", "cpgimp", "cpdi"])
def test_new_checkpoint_hashes_every_vtu_frame(tmp_path, method):
    solver, manager, ck = _checkpoint(tmp_path, method)
    state = inspect_checkpoint(ck.manifest_path)["output_manager"]
    integrity = state["vtu_integrity"]
    assert (integrity["version"], integrity["algorithm"]) == (1, "sha256")
    assert set(integrity["files"]) == set(manager.enabled)
    assert set(integrity["files"]) == set(state["entries"])
    for kind in manager.enabled:
        hashes = integrity["files"][kind]
        expected = {frame.filename for frame in manager.frames[kind]}
        assert set(hashes) == expected
        for filename, digest in hashes.items():
            assert digest == hashlib.sha256((manager.directory / filename).read_bytes()).hexdigest()
    restored = restore_output_manager(ck.manifest_path, load_checkpoint(ck.manifest_path))
    assert restored.frames == manager.frames
    assert restored.directory == manager.directory
    assert solver.step_count == restored._last_seen_step


@pytest.mark.parametrize("kind,method", [("particles", "mpm"), ("grid", "gimp"),
                                          ("domains", "cpdi")])
def test_vtu_modified_content_cannot_resume(tmp_path, kind, method):
    _, manager, ck = _checkpoint(tmp_path, method)
    path = manager.directory / manager.frames[kind][0].filename
    path.write_bytes(b"not a valid VTK file; tampered contents")
    before = path.read_bytes()
    with pytest.raises(ValueError, match=r"VTU SHA-256 mismatch"):
        restore_output_manager(ck.manifest_path, load_checkpoint(ck.manifest_path))
    assert path.read_bytes() == before  # no healing, deletion or hidden rewrite


def test_hash_verification_checks_old_and_new_frames(tmp_path):
    _, manager, ck = _checkpoint(tmp_path, steps=17)
    assert len(manager.frames["particles"]) >= 3
    latest = manager.frames["particles"][-1].filename
    (manager.directory / latest).write_bytes(b"damaged tail")
    with pytest.raises(ValueError, match="VTU SHA-256 mismatch"):
        restore_output_manager(ck.manifest_path, load_checkpoint(ck.manifest_path))


@pytest.mark.parametrize("change", [
    lambda integ: integ["files"]["particles"].pop(next(iter(integ["files"]["particles"]))),
    lambda integ: integ["files"]["particles"].update({"stray_000000.vtu": "a" * 64}),
    lambda integ: integ.update({"version": 2}),
    lambda integ: integ.update({"version": True}),
    lambda integ: integ.update({"algorithm": "sha1"}),
    lambda integ: integ.update({"files": {}}),
    lambda integ: integ["files"]["particles"].update({"particles_000000.vtu": "not-a-hash"}),
])
def test_corrupt_or_incomplete_integrity_manifest_rejected(tmp_path, change):
    _, _, ck = _checkpoint(tmp_path)
    def mutate(meta):
        change(meta["output_manager"]["vtu_integrity"])
    _rewrite_manifest(ck, mutate)
    with pytest.raises(ValueError, match="VTU|integrity"):
        restore_output_manager(ck.manifest_path, load_checkpoint(ck.manifest_path))


def test_directory_copy_has_identical_vtu_hashes(tmp_path):
    _, manager, ck = _checkpoint(tmp_path, "cpdi")
    branch = tmp_path / "branch"
    shutil.copytree(manager.directory, branch)
    restored = load_checkpoint(ck.manifest_path)
    continued = restore_output_manager(ck.manifest_path, restored, directory=branch)
    assert continued.directory == branch
    assert continued.frames == manager.frames
    first_written = {}
    for _ in range(12):
        first_written = continued.on_step_completed(restored, restored.step())
        if first_written:
            break
    # The copied branch can be appended, but the original cannot change.
    assert first_written["particles"].is_file()
    assert not (manager.directory / first_written["particles"].name).exists()


def test_legacy_manifest_requires_explicit_override(tmp_path):
    _, manager, ck = _checkpoint(tmp_path)
    _rewrite_manifest(ck, lambda meta: meta["output_manager"].pop("vtu_integrity"))
    fresh = load_checkpoint(ck.manifest_path)
    with pytest.raises(ValueError, match="Legacy checkpoint lacks VTU SHA-256"):
        restore_output_manager(ck.manifest_path, fresh)
    resumed = restore_output_manager(
        ck.manifest_path, fresh, allow_unverified_legacy_vtu=True)
    assert resumed.frames == manager.frames

    # The legacy override is intentionally NOT an integrity guarantee.
    (manager.directory / "particles_000000.vtu").write_bytes(b"legacy changed")
    unchecked = restore_output_manager(
        ck.manifest_path, fresh, allow_unverified_legacy_vtu=True)
    assert unchecked.frames == manager.frames


def test_legacy_nonempty_manifest_optin_is_boolean_only(tmp_path):
    _, _, ck = _checkpoint(tmp_path)
    _rewrite_manifest(ck, lambda meta: meta["output_manager"].pop("vtu_integrity"))
    with pytest.raises(TypeError, match="must be a bool"):
        restore_output_manager(ck.manifest_path, load_checkpoint(ck.manifest_path),
                               allow_unverified_legacy_vtu=1)


def test_legacy_empty_output_series_can_resume_without_hashes(tmp_path):
    from geopympm.io import OutputManager
    solver = make_case("mpm")
    manager = OutputManager(tmp_path / "viz", every_n_steps=50,
                            particles=True, grid=False)
    manager.on_step_completed(solver, solver.step())
    ck = write_checkpoint(solver, tmp_path / "ckpt", output_manager=manager)
    _rewrite_manifest(ck, lambda meta: meta["output_manager"].pop("vtu_integrity"))
    resumed = restore_output_manager(ck.manifest_path, load_checkpoint(ck.manifest_path))
    assert resumed.frames["particles"] == ()
    assert not manager.directory.exists()


def test_writer_refuses_missing_vtu_before_publishing_checkpoint(tmp_path):
    solver = make_case("mpm")
    manager = _make_output(solver, tmp_path / "viz")
    (manager.directory / "particles_000000.vtu").unlink()
    with pytest.raises(FileNotFoundError, match="missing VTU"):
        write_checkpoint(solver, tmp_path / "ckpt", output_manager=manager)
    assert not (tmp_path / "ckpt").exists()


def test_writer_refuses_modified_output_index(tmp_path):
    solver = make_case("mpm")
    manager = _make_output(solver, tmp_path / "viz")
    (manager.directory / "particles.visit").write_text("bad index", encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent particles.visit"):
        write_checkpoint(solver, tmp_path / "ckpt", output_manager=manager)
    assert not (tmp_path / "ckpt").exists()


def test_writer_refuses_unregistered_vtu_frames(tmp_path):
    solver = make_case("mpm")
    manager = _make_output(solver, tmp_path / "viz")
    (manager.directory / "particles_987654.vtu").write_bytes(b"orphan")
    with pytest.raises(ValueError, match="extra/missing"):
        write_checkpoint(solver, tmp_path / "ckpt", output_manager=manager)
    assert not (tmp_path / "ckpt").exists()


def test_new_checkpoint_does_not_skip_hash_verification_with_legacy_optin(tmp_path):
    _, manager, ck = _checkpoint(tmp_path)
    (manager.directory / "particles_000000.vtu").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="VTU SHA-256 mismatch"):
        restore_output_manager(ck.manifest_path, load_checkpoint(ck.manifest_path),
                               allow_unverified_legacy_vtu=True)
