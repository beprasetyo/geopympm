"""Stage 10G-D3: conservative checkpoint discovery, recovery and retention.

The manifest JSON is the commit marker; a checkpoint is eligible only when
``load_checkpoint`` reconstructs its complete validated numerical state.
This module NEVER changes visualization snapshots, indexes or solver kernels.
No automatic repair of corrupt manifests/NPZ files is attempted.

Assumptions: a single writer, no concurrent checkpoint creation/pruning in the
same directory, and a trusted checkpoint directory (hashes detect accidental
changes; SHA-256 is not an authentication signature).
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING

from .checkpoint import _NAME
from .restart import load_checkpoint, restore_output_manager

if TYPE_CHECKING:
    from geopympm.solvers.explicit import ExplicitMPMSolver
    from .output_manager import OutputManager


@dataclass(frozen=True)
class CheckpointIssue:
    """Manifest rejected during discovery; reason is suitable for a log."""

    manifest_path: Path
    reason: str


@dataclass(frozen=True)
class CheckpointInventory:
    """Validated numerical checkpoints, newest first; invalid pairs untouched."""

    valid: tuple[Path, ...]
    invalid: tuple[CheckpointIssue, ...]
    orphan_data: tuple[Path, ...]


@dataclass(frozen=True)
class RecoveryResult:
    """Chosen restart state and skipped candidates (highest step first)."""

    solver: "ExplicitMPMSolver"
    manifest_path: Path
    output_manager: "OutputManager | None"
    skipped: tuple[CheckpointIssue, ...]

    @property
    def step(self) -> int:
        return self.solver.step_count


@dataclass(frozen=True)
class RetentionReport:
    """Retention intent and effects; dry-run is the default.

    ``would_delete`` lists committed manifests, not individual NPZ paths;
    each represents a JSON + NPZ pair. ``deleted`` is empty for dry-runs.
    """

    dry_run: bool
    keep_last: int
    retained: tuple[Path, ...]
    would_delete: tuple[Path, ...]
    deleted: tuple[Path, ...]
    invalid: tuple[CheckpointIssue, ...]
    orphan_data: tuple[Path, ...]


class CheckpointRecoveryError(RuntimeError):
    """No eligible checkpoint could be restored without compromising safety."""


def _folder_and_prefix(directory: str | os.PathLike[str], prefix: str) -> tuple[Path, re.Pattern[str]]:
    if not isinstance(prefix, str) or _NAME.fullmatch(prefix) is None:
        raise ValueError("Invalid checkpoint prefix")
    folder = Path(directory)
    if folder.is_symlink():
        raise ValueError("Checkpoint directory must not be a symlink")
    if not folder.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {folder}")
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d{{8,}})\.json$")
    return folder, pattern


def _candidates(folder: Path, pattern: re.Pattern[str]) -> list[tuple[int, Path]]:
    """Only filenames made by write_checkpoint are candidates."""
    candidates = []
    for path in folder.iterdir():
        match = pattern.fullmatch(path.name)
        if match is not None:
            step = int(match.group(1))
            if match.group(1) == f"{step:08d}":
                candidates.append((step, path))
    return sorted(candidates, key=lambda item: (item[0], item[1].name), reverse=True)


def _load_pair(path: Path, filename_step: int) -> "ExplicitMPMSolver":
    if path.is_symlink() or not path.is_file():
        raise ValueError("Manifest must be a regular non-symlinked file")
    data = path.with_suffix(".npz")
    if data.is_symlink() or not data.is_file():
        raise ValueError("Numeric archive must be a regular non-symlinked file")
    solver = load_checkpoint(path)
    if solver.step_count != filename_step:
        raise ValueError("Checkpoint filename step differs from its stored solver step")
    return solver


def scan_checkpoints(
    directory: str | os.PathLike[str], *, prefix: str = "state",
) -> CheckpointInventory:
    """Verify every committed numerical pair, returning newest valid first.

    Validity means a complete lossless solver restoration, not merely a valid
    NPZ hash. Dangling NPZ archives and invalid JSON/NPZ are never deleted.
    OutputManager series are NOT verified by this numerical-only scan; use
    ``recover_latest_checkpoint(..., restore_output=True)`` when required.
    """
    folder, pattern = _folder_and_prefix(directory, prefix)
    valid: list[Path] = []
    invalid: list[CheckpointIssue] = []
    for step, path in _candidates(folder, pattern):
        try:
            _load_pair(path, step)
        except Exception as exc:
            # Malformed external archives can raise several decoder errors;
            # preserve the rejected pair and report the exact exception.
            invalid.append(CheckpointIssue(path, f"{type(exc).__name__}: {exc}"))
        else:
            valid.append(path)

    # Uncommitted NPZ files must stay available for manual investigation.
    data_pattern = re.compile(rf"^{re.escape(prefix)}_(\d{{8,}})\.npz$")
    orphans = sorted((p for p in folder.iterdir()
                      if data_pattern.fullmatch(p.name) and
                      not p.with_suffix(".json").exists()), key=lambda p: p.name)
    return CheckpointInventory(tuple(valid), tuple(invalid), tuple(orphans))


def recover_latest_checkpoint(
    directory: str | os.PathLike[str], *, prefix: str = "state",
    restore_output: bool = False,
    output_directory: str | os.PathLike[str] | None = None,
    allow_unverified_legacy_vtu: bool = False,
) -> RecoveryResult:
    """Recover the newest *fully restorable* checkpoint, falling back safely.

    ``restore_output=False`` (default) recovers only the numerical solver.
    If ``restore_output=True``, the exact existing visualization series must
    ALSO satisfy Stage 10G-D2 checks. Older checkpoints cannot append to a
    directory with newer VTU frames: copy an exact checkpoint-era series to a
    separate directory first, or resume without OutputManager.

    The search reads files, but never modifies or deletes any of them.
    """
    if type(restore_output) is not bool:
        raise TypeError("restore_output must be a bool")
    if type(allow_unverified_legacy_vtu) is not bool:
        raise TypeError("allow_unverified_legacy_vtu must be a bool")
    folder, pattern = _folder_and_prefix(directory, prefix)
    skipped: list[CheckpointIssue] = []
    for step, path in _candidates(folder, pattern):
        try:
            solver = _load_pair(path, step)
            output = (restore_output_manager(
                path, solver, directory=output_directory,
                allow_unverified_legacy_vtu=allow_unverified_legacy_vtu,
            ) if restore_output else None)
        except Exception as exc:
            # Never mutate a failed candidate; try an earlier committed pair.
            skipped.append(CheckpointIssue(path, f"{type(exc).__name__}: {exc}"))
            continue
        return RecoveryResult(solver, path, output, tuple(skipped))
    raise CheckpointRecoveryError(
        f"No restorable checkpoint found in {folder} for prefix {prefix!r}"
        + (" (including output series)" if restore_output else "")
        + (". Rejected: " + "; ".join(
            f"{v.manifest_path.name}: {v.reason}" for v in skipped
        ) if skipped else ". No committed manifests found.")
    )


def prune_checkpoints(
    directory: str | os.PathLike[str], *, prefix: str = "state",
    keep_last: int = 3, dry_run: bool = True,
) -> RetentionReport:
    """Keep newest N *valid* numerical checkpoints; leave invalid/orphans.

    Default is a no-write dry run. With ``dry_run=False``, delete only older,
    individually validated **JSON+NPZ** pairs, manifest first (commit marker),
    data second. On interruption an NPZ orphan may remain, but there will not
    be a committed manifest pointing to a deleted NPZ. Never prune VTU frames,
    indexes, invalid pairs, or uncommitted NPZ files.

    Run without a concurrent checkpoint writer, and back up important data.
    ``keep_last`` protects N numeric checkpoints, not N independent backups
    of the complete visualization directory.
    """
    if type(keep_last) is not int or keep_last < 1:
        raise ValueError("keep_last must be a positive integer")
    if type(dry_run) is not bool:
        raise TypeError("dry_run must be a bool")
    inventory = scan_checkpoints(directory, prefix=prefix)
    retained = inventory.valid[:keep_last]
    would_delete = inventory.valid[keep_last:]
    deleted: list[Path] = []
    if not dry_run:
        for path in would_delete:
            # Reject file replacements between scan and prune. Still requires
            # single-writer operation: filesystem operations are not atomic
            # across a two-file pair.
            step = int(path.stem.rsplit("_", 1)[1])
            _load_pair(path, step)
            path.unlink()  # first remove commit marker
            path.with_suffix(".npz").unlink()  # then remove orphan data
            deleted.append(path)
    return RetentionReport(
        dry_run, keep_last, retained, would_delete, tuple(deleted),
        inventory.invalid, inventory.orphan_data,
    )


__all__ = [
    "CheckpointIssue", "CheckpointInventory", "RecoveryResult",
    "RetentionReport", "CheckpointRecoveryError", "scan_checkpoints",
    "recover_latest_checkpoint", "prune_checkpoints",
]
