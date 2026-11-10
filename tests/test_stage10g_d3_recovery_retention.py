"""Stage 10G-D3: recover latest good checkpoint; conservative retention.

Use authentic Stage-10G snapshots and kernels rather than mock checkpoint files.
"""
from __future__ import annotations

import json
import shutil

import pytest

from geopympm.io import (
    CheckpointRecoveryError, load_checkpoint, write_checkpoint,
    scan_checkpoints, recover_latest_checkpoint, prune_checkpoints,
)
from validation.stage10g_c.extended_restart import make_case, _make_output, _compare_states


def _series(tmp_path, method='mpm', *, steps=(3, 6, 9), output=False, prefix='state'):
    solver = make_case(method)
    out = _make_output(solver, tmp_path / 'viz') if output else None
    checkpoints = []
    stop = max(steps)
    for i in range(1, stop + 1):
        diagnostic = solver.step()
        if out is not None:
            out.on_step_completed(solver, diagnostic)
        if i in steps:
            checkpoints.append(write_checkpoint(
                solver, tmp_path / 'ck', output_manager=out, prefix=prefix))
    return solver, out, checkpoints


@pytest.mark.parametrize('method', ['mpm', 'gimp', 'cpgimp', 'cpdi'])
def test_recover_newest_all_interpolations(tmp_path, method):
    solver, _, records = _series(tmp_path, method)
    found = recover_latest_checkpoint(tmp_path / 'ck')
    assert found.manifest_path == records[-1].manifest_path
    assert found.step == 9
    assert found.output_manager is None
    assert not found.skipped
    _compare_states(solver, found.solver)
    assert scan_checkpoints(tmp_path / 'ck').valid == tuple(
        r.manifest_path for r in records[::-1])


@pytest.mark.parametrize('damage', ['npz', 'json', 'missing_npz', 'bad_schema'])
def test_fallback_skips_corrupt_latest(tmp_path, damage):
    _, _, records = _series(tmp_path)
    last = records[-1]
    if damage == 'npz':
        with last.data_path.open('ab') as fh:
            fh.write(b'changed')
    elif damage == 'json':
        last.manifest_path.write_text('{bad json', encoding='utf-8')
    elif damage == 'missing_npz':
        last.data_path.unlink()
    else:
        meta = json.loads(last.manifest_path.read_text())
        meta['schema_version'] = 99
        last.manifest_path.write_text(json.dumps(meta))
    result = recover_latest_checkpoint(tmp_path / 'ck')
    assert result.step == 6
    assert result.manifest_path == records[1].manifest_path
    assert len(result.skipped) == 1
    assert result.skipped[0].manifest_path == last.manifest_path
    assert len(scan_checkpoints(tmp_path / 'ck').invalid) == 1


def test_missing_manifest_npz_is_orphan_and_ignored(tmp_path):
    _, _, records = _series(tmp_path, steps=(2, 4))
    (tmp_path / 'ck' / 'state_00000090.npz').write_bytes(b'partial-data')
    inventory = scan_checkpoints(tmp_path / 'ck')
    assert len(inventory.valid) == 2
    assert [p.name for p in inventory.orphan_data] == ['state_00000090.npz']
    assert recover_latest_checkpoint(tmp_path / 'ck').step == 4
    report = prune_checkpoints(tmp_path / 'ck', keep_last=1, dry_run=False)
    assert inventory.orphan_data[0].exists()
    assert not records[0].data_path.exists()
    assert report.deleted == (records[0].manifest_path,)


def test_manifest_filename_identity_mismatch_refused(tmp_path):
    _, _, records = _series(tmp_path, steps=(2, 4))
    fake = tmp_path / 'ck' / 'state_00000050.json'
    shutil.copyfile(records[-1].manifest_path, fake)
    shutil.copyfile(records[-1].data_path, fake.with_suffix(".npz"))
    assert recover_latest_checkpoint(tmp_path / 'ck').step == 4
    inventory = scan_checkpoints(tmp_path / 'ck')
    assert len(inventory.invalid) == 1
    assert 'filename' in inventory.invalid[0].reason.lower() or 'data filename' in inventory.invalid[0].reason.lower()


def test_only_requested_prefix_is_scanned(tmp_path):
    _series(tmp_path, steps=(2,), prefix='state')
    _, _, other = _series(tmp_path, steps=(4,), prefix='backup')
    assert recover_latest_checkpoint(tmp_path / 'ck', prefix='backup').step == 4
    assert scan_checkpoints(tmp_path / 'ck', prefix='state').valid[0].name == 'state_00000002.json'
    assert other[-1].manifest_path.exists()


def test_recovery_is_nonmutating(tmp_path):
    _, _, records = _series(tmp_path)
    snapshot = {p.name: p.read_bytes() for p in (tmp_path / 'ck').iterdir()}
    last = records[-1]
    last.manifest_path.write_bytes(b'broken')
    snapshot[last.manifest_path.name] = b'broken'
    assert recover_latest_checkpoint(tmp_path / 'ck').step == 6
    assert {p.name: p.read_bytes() for p in (tmp_path / 'ck').iterdir()} == snapshot


def test_output_restore_valid_latest_and_no_false_fallback(tmp_path):
    _, manager, records = _series(tmp_path, 'cpdi', steps=(4, 8, 17), output=True)
    good = recover_latest_checkpoint(tmp_path / 'ck', restore_output=True)
    assert good.step == 17
    assert good.output_manager.frames == manager.frames
    # Damaging last numeric archive requires fallback to step 8. But the same
    # live output directory has progressed beyond step 8, so D2 must
    # reject output continuation instead of deleting newer frames.
    records[-1].data_path.write_bytes(b'broken')
    assert recover_latest_checkpoint(tmp_path / 'ck').step == 8
    before = {p.name: p.read_bytes() for p in manager.directory.iterdir()}
    with pytest.raises(CheckpointRecoveryError, match='No restorable checkpoint'):
        recover_latest_checkpoint(tmp_path / 'ck', restore_output=True)
    assert {p.name: p.read_bytes() for p in manager.directory.iterdir()} == before


def test_output_content_tampering_rejected_numerical_only_still_recovers(tmp_path):
    _, manager, records = _series(tmp_path, output=True, steps=(4, 8))
    first = manager.directory / 'particles_000000.vtu'
    first.write_bytes(b'damaged')
    assert recover_latest_checkpoint(tmp_path / 'ck').step == 8
    with pytest.raises(CheckpointRecoveryError, match='VTU SHA-256 mismatch'):
        recover_latest_checkpoint(tmp_path / 'ck', restore_output=True)
    assert records[-1].manifest_path.exists()


def test_output_can_restore_from_exact_copy(tmp_path):
    _, manager, records = _series(tmp_path, 'gimp', steps=(3, 6, 9), output=True)
    branch = tmp_path / 'branch'
    shutil.copytree(manager.directory, branch)
    found = recover_latest_checkpoint(tmp_path / 'ck', restore_output=True,
                                      output_directory=branch)
    assert found.step == 9
    assert found.output_manager.directory == branch
    assert found.output_manager.frames == manager.frames



def test_output_fallback_allowed_when_no_additional_frames(tmp_path):
    _, manager, records = _series(tmp_path, steps=(3, 5), output=True)
    # Only the initial frame has been emitted. Both checkpoints describe the
    # identical series, so returning to step 3 is safe and should be allowed.
    records[-1].data_path.write_bytes(b'damaged latest')
    recovered = recover_latest_checkpoint(tmp_path / 'ck', restore_output=True)
    assert recovered.step == 3
    assert recovered.output_manager.frames == manager.frames
    assert len(recovered.skipped) == 1


def test_fallback_numerical_continuation_matches_direct_checkpoint(tmp_path):
    _, _, records = _series(tmp_path, 'cpdi', steps=(5, 10))
    reference = load_checkpoint(records[0].manifest_path)
    records[-1].data_path.write_bytes(b'bad newest state')
    recovered = recover_latest_checkpoint(tmp_path / 'ck')
    assert recovered.step == 5
    for _ in range(15):
        assert reference.step() == recovered.solver.step()
        _compare_states(reference, recovered.solver)


def test_retention_dry_run_is_non_mutating(tmp_path):
    _, _, records = _series(tmp_path, steps=(2, 4, 6, 8, 10))
    before = {p.name: p.read_bytes() for p in (tmp_path / 'ck').iterdir()}
    result = prune_checkpoints(tmp_path / 'ck', keep_last=2)
    assert result.dry_run is True
    assert result.retained == (records[-1].manifest_path, records[-2].manifest_path)
    assert result.would_delete == tuple(r.manifest_path for r in records[-3::-1])
    assert result.deleted == ()
    assert {p.name: p.read_bytes() for p in (tmp_path / 'ck').iterdir()} == before


def test_retention_deletes_only_older_valid_pairs(tmp_path):
    _, manager, records = _series(tmp_path, steps=(3, 6, 9, 12), output=True)
    before_viz = {p.name: p.read_bytes() for p in manager.directory.iterdir()}
    report = prune_checkpoints(tmp_path / 'ck', keep_last=2, dry_run=False)
    assert report.deleted == (records[1].manifest_path, records[0].manifest_path)
    for record in records[:2]:
        assert not record.manifest_path.exists()
        assert not record.data_path.exists()
    for record in records[2:]:
        assert record.manifest_path.exists() and record.data_path.exists()
        assert load_checkpoint(record.manifest_path).step_count == record.step
    assert recover_latest_checkpoint(tmp_path / 'ck').step == 12
    assert {p.name: p.read_bytes() for p in manager.directory.iterdir()} == before_viz


def test_retention_keeps_corrupt_pairs_and_uses_valid_count(tmp_path):
    _, _, records = _series(tmp_path, steps=(2, 4, 6, 8, 10))
    records[-1].data_path.write_bytes(b'corrupt newer archive')
    bad_before = records[-1].data_path.read_bytes()
    report = prune_checkpoints(tmp_path / 'ck', keep_last=2, dry_run=False)
    assert report.retained == (records[3].manifest_path, records[2].manifest_path)
    assert report.deleted == (records[1].manifest_path, records[0].manifest_path)
    assert len(report.invalid) == 1
    assert records[-1].manifest_path.exists()
    assert records[-1].data_path.read_bytes() == bad_before
    assert recover_latest_checkpoint(tmp_path / 'ck').step == 8


def test_retention_no_valid_pairs_is_no_op(tmp_path):
    folder = tmp_path / 'ck'
    folder.mkdir()
    (folder / 'state_00000001.json').write_bytes(b'broken')
    (folder / 'state_00000001.npz').write_bytes(b'broken')
    rep = prune_checkpoints(folder, keep_last=2, dry_run=False)
    assert rep.deleted == () and rep.retained == () and len(rep.invalid) == 1
    assert (folder / 'state_00000001.json').exists()
    assert (folder / 'state_00000001.npz').exists()


@pytest.mark.parametrize('bad', [0, -1, True, 1.3, '3', None])
def test_retention_invalid_keep_last_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match='positive integer'):
        prune_checkpoints(tmp_path, keep_last=bad)


@pytest.mark.parametrize('bad', [0, 'yes', None])
def test_flag_validation(tmp_path, bad):
    with pytest.raises(TypeError, match='dry_run must be a bool'):
        prune_checkpoints(tmp_path, dry_run=bad)
    with pytest.raises(TypeError, match='restore_output must be a bool'):
        recover_latest_checkpoint(tmp_path, restore_output=bad)


def test_invalid_prefix_and_symlink_directory_refused(tmp_path):
    folder = tmp_path / 'ck'
    folder.mkdir()
    link = tmp_path / 'link'
    link.symlink_to(folder, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        scan_checkpoints(link)
    with pytest.raises(ValueError, match='Invalid checkpoint prefix'):
        prune_checkpoints(folder, prefix='../weird')


def test_symlinked_newest_pair_skipped(tmp_path):
    _, _, records = _series(tmp_path, steps=(2, 4))
    # A symlinked manifest is never eligible as a committed pair.
    (tmp_path / 'ck' / 'state_00000050.json').symlink_to(records[-1].manifest_path)
    (tmp_path / 'ck' / 'state_00000050.npz').symlink_to(records[-1].data_path)
    found = recover_latest_checkpoint(tmp_path / 'ck')
    assert found.step == 4
    assert len(found.skipped) == 1
    assert 'symlink' in found.skipped[0].reason


def test_missing_directory_and_no_committed_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        recover_latest_checkpoint(tmp_path / 'not-created')
    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(CheckpointRecoveryError, match='No committed manifests'):
        recover_latest_checkpoint(empty)
