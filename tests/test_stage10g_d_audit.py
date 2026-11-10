"""Stage 10G-D: benchmark reproducibility and known-integrity-gap regression."""

import pytest

from geopympm.io import load_checkpoint, restore_output_manager, write_checkpoint
from validation.stage10g_c.extended_restart import make_case, _make_output
from validation.stage10g_d.audit import run_audit


def test_audit_generates_machine_readable_results(tmp_path):
    data = run_audit(methods=("mpm",), sizes=(2,), steps=2, repeats=1,
                     output_dir=tmp_path / "audit")
    record, = data["measurements"]
    assert record["particles"] == 16
    assert record["checkpoint_bytes"] > 0
    assert record["checkpoint_write_median_sec"] > 0
    assert record["checkpoint_load_median_sec"] > 0
    assert (tmp_path / "audit" / "measurements.csv").is_file()


def test_changed_vtu_bytes_must_reject_output_restart(tmp_path):
    solver = make_case("mpm")
    manager = _make_output(solver, tmp_path / "visualization")
    checkpoint = write_checkpoint(solver, tmp_path / "checkpoint", output_manager=manager)
    snapshot = manager.directory / "particles_000000.vtu"
    snapshot.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="VTU|SHA-256|hash|integrity"):
        restore_output_manager(checkpoint.manifest_path, load_checkpoint(checkpoint.manifest_path))
