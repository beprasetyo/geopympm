"""Stage 10G-C: multi-checkpoint, dynamic CPDI and failure-mode audit."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

from geopympm.io import (OutputManager, inspect_checkpoint, load_checkpoint,
                         restore_output_manager, write_checkpoint)
from validation.stage10g_c.extended_restart import (
    METHODS, _check_output, _compare_states, _make_output, make_case, run_chain,
)


@pytest.mark.parametrize("method", METHODS)
def test_multi_checkpoint_long_restart_is_bitwise_equal(tmp_path, method):
    result = run_chain(method, tmp_path / method, steps=64)
    assert result["continuous_equals_restarted"]
    assert result["diagnostics_match_every_step"]
    assert result["time_series_match_every_step"]
    assert [r["step"] for r in result["checkpoints"]] == [16, 32, 48]
    assert len(result["frame_counts"]) == (3 if method == "cpdi" else 2)
    assert len(set(r["step"] for r in result["checkpoints"])) == 3


def test_deforming_cpdi_long_restart_preserves_rotation_and_domain(tmp_path):
    result = run_chain("cpdi", tmp_path / "dynamic", steps=120, dynamic=True)
    assert result["max_abs_Fxy"] > .15, result
    assert result["max_abs_Fyx"] > .05, result
    assert result["max_polar_angle_deg"] > 5., result
    assert result["min_det_F"] > 0., result
    assert result["final_max_abs_F_minus_I"] > .1, result
    assert [r["step"] for r in result["checkpoints"]] == [30, 60, 90]


def _setup(tmp_path, *, method="cpdi", nsteps=6):
    s = make_case(method)
    out = _make_output(s, tmp_path / "viz")
    for _ in range(nsteps):
        out.on_step_completed(s, s.step())
    ck = write_checkpoint(s, tmp_path / "checkpoint", output_manager=out)
    return s, out, ck


def test_unmodified_checkpoint_files_not_changed_by_loading(tmp_path):
    _, _, c = _setup(tmp_path)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (c.data_path, c.manifest_path)}
    for _ in range(2):
        fresh = load_checkpoint(c.manifest_path)
        restore_output_manager(c.manifest_path, fresh)
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in before} == before


def test_corrupt_latest_fails_but_earlier_intact_checkpoint_remains_recoverable(tmp_path):
    s, out, earlier = _setup(tmp_path, nsteps=6)
    for _ in range(4):
        out.on_step_completed(s, s.step())
    latest = write_checkpoint(s, tmp_path / "checkpoint", output_manager=out)
    with latest.data_path.open("ab") as stream:
        stream.write(b"post-write-corruption")
    with pytest.raises(ValueError, match="SHA-256"):
        load_checkpoint(latest.manifest_path)
    earlier_state = load_checkpoint(earlier.manifest_path)
    assert earlier_state.step_count == 6
    # Filesystem output has progressed to t=10; old checkpoint *cannot* append there.
    with pytest.raises(ValueError, match="extra/missing|does not match"):
        restore_output_manager(earlier.manifest_path, earlier_state)
    assert inspect_checkpoint(earlier.manifest_path)["step"] == 6


def test_missing_checkpoint_npz_is_rejected(tmp_path):
    _, _, c = _setup(tmp_path)
    c.data_path.unlink()
    with pytest.raises(FileNotFoundError):
        load_checkpoint(c.manifest_path)


def test_missing_checkpoint_manifest_is_rejected(tmp_path):
    _, _, c = _setup(tmp_path)
    c.manifest_path.unlink()
    with pytest.raises(FileNotFoundError):
        load_checkpoint(c.manifest_path)


def test_truncated_checkpoint_archive_is_rejected(tmp_path):
    _, _, c = _setup(tmp_path)
    c.data_path.write_bytes(c.data_path.read_bytes()[:20])
    with pytest.raises(ValueError, match="SHA-256"):
        load_checkpoint(c.manifest_path)


def test_corrupt_json_is_rejected(tmp_path):
    _, _, c = _setup(tmp_path)
    c.manifest_path.write_text("{broken JSON", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_checkpoint(c.manifest_path)


def test_unsupported_schema_version_is_rejected(tmp_path):
    _, _, c = _setup(tmp_path)
    meta = inspect_checkpoint(c.manifest_path)
    meta["schema_version"] = 999
    c.manifest_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        load_checkpoint(c.manifest_path)


@pytest.mark.parametrize("stream_file,change", [
    ("particles.visit", "corrupt"),
    ("particles.pvd", "delete"),
    ("domains.visit", "corrupt"),
    ("grid.pvd", "corrupt"),
    ("particles_000000.vtu", "delete"),
])
def test_corrupt_or_missing_visualization_refuses_resume_without_overwrite(tmp_path, stream_file, change):
    _, out, c = _setup(tmp_path)
    item = out.directory / stream_file
    if change == "delete":
        item.unlink()
    else:
        item.write_bytes(b"BROKEN")
    before = {p.name: p.read_bytes() for p in out.directory.iterdir() if p.is_file()}
    with pytest.raises((ValueError, FileNotFoundError)):
        restore_output_manager(c.manifest_path, load_checkpoint(c.manifest_path))
    after = {p.name: p.read_bytes() for p in out.directory.iterdir() if p.is_file()}
    assert before == after  # loader must not repair/overwrite failure silently


def test_extra_vtu_file_protects_against_old_checkpoint_replay(tmp_path):
    _, out, c = _setup(tmp_path)
    (out.directory / "particles_999999.vtu").write_bytes(b"extra")
    with pytest.raises(ValueError, match="extra/missing"):
        restore_output_manager(c.manifest_path, load_checkpoint(c.manifest_path))


def test_safe_branch_copy_preserves_original_visualization(tmp_path):
    baseline, out, c = _setup(tmp_path)
    branch_path = tmp_path / "branch"
    shutil.copytree(out.directory, branch_path)
    original_files = {p.name: p.read_bytes() for p in out.directory.iterdir()}
    branch = load_checkpoint(c.manifest_path)
    manager = restore_output_manager(c.manifest_path, branch, directory=branch_path)
    for _ in range(6):
        da, db = baseline.step(), branch.step()
        assert da == db
        out.on_step_completed(baseline, da)
        manager.on_step_completed(branch, db)
    _compare_states(baseline, branch)
    _check_output(out, manager)
    assert (branch_path / "particles.visit").read_bytes() != original_files["particles.visit"]
    # The original old snapshots themselves must be preserved.
    for name, contents in original_files.items():
        if name.endswith(".vtu"):
            assert (out.directory / name).read_bytes() == contents


def test_checkpoint_writer_will_not_overwrite_same_step(tmp_path):
    _, _, c = _setup(tmp_path)
    s = load_checkpoint(c.manifest_path)
    before = c.manifest_path.read_bytes(), c.data_path.read_bytes()
    with pytest.raises(FileExistsError):
        write_checkpoint(s, tmp_path / "checkpoint")
    assert before == (c.manifest_path.read_bytes(), c.data_path.read_bytes())


def test_checkpoint_without_visualization_still_restartable(tmp_path):
    a = make_case("cpdi", dynamic=True)
    for _ in range(15):
        a.step()
    c = write_checkpoint(a, tmp_path / "checkpoint")
    b = load_checkpoint(c.manifest_path)
    _compare_states(a, b)
    with pytest.raises(ValueError, match="does not contain"):
        restore_output_manager(c.manifest_path, b)
    for _ in range(15):
        assert a.step() == b.step()
    _compare_states(a, b)


def test_restart_in_independent_python_process(tmp_path):
    original = make_case("cpdi", dynamic=True)
    for _ in range(12):
        original.step()
    c = write_checkpoint(original, tmp_path / "checkpoint")
    npz_out = tmp_path / "process_result.npz"
    source = """
import sys, numpy as np
from geopympm.io import load_checkpoint
s=load_checkpoint(sys.argv[1])
for _ in range(8): s.step()
np.savez(sys.argv[2], pos=s.particles.position, F=s.particles.deformation_gradient,
         stress=s.particles.stress, velocity=s.particles.velocity)
"""
    subprocess.run([sys.executable, "-c", source, str(c.manifest_path), str(npz_out)],
                   check=True, capture_output=True, text=True)
    for _ in range(8):
        original.step()
    with np.load(npz_out, allow_pickle=False) as out:
        np.testing.assert_array_equal(out["pos"], original.particles.position)
        np.testing.assert_array_equal(out["F"], original.particles.deformation_gradient)
        np.testing.assert_array_equal(out["stress"], original.particles.stress)
        np.testing.assert_array_equal(out["velocity"], original.particles.velocity)


def test_cpdi_domain_corners_recovered_exactly(tmp_path):
    s = make_case("cpdi", dynamic=True)
    for _ in range(60):
        s.step()
    c = write_checkpoint(s, tmp_path)
    r = load_checkpoint(c.manifest_path)
    center = s.particles.position
    F = s.particles.deformation_gradient
    size0 = s.particles.domain_size0
    a = F[:, :, 0] * (size0[:, 0] / 2)[:, None]
    b = F[:, :, 1] * (size0[:, 1] / 2)[:, None]
    corners_a = np.stack((center-a-b, center+a-b, center+a+b, center-a+b), axis=1)
    center2, F2, size2 = r.particles.position, r.particles.deformation_gradient, r.particles.domain_size0
    a2 = F2[:, :, 0] * (size2[:, 0] / 2)[:, None]
    b2 = F2[:, :, 1] * (size2[:, 1] / 2)[:, None]
    corners_b = np.stack((center2-a2-b2, center2+a2-b2, center2+a2+b2, center2-a2+b2), axis=1)
    np.testing.assert_array_equal(corners_a, corners_b)
    assert float(np.max(np.abs(F[:, 0, 1]))) > .1
