"""Stage 10G-B1: lossless, inspectable *writer* for committed MPM states.

A checkpoint consists of ``state_XXXXXXXX.npz`` (numeric arrays only) and
``state_XXXXXXXX.json`` (versioned manifest). The JSON manifest is published
last and acts as the commit marker. B1 deliberately DOES NOT reconstruct a
solver; that is the responsibility of Stage 10G-B2.

Only the Stage-10G-A3 schema is supported. Unknown state is rejected rather
than silently omitted. In particular, constitutive internal variables will
need a registered serializer when elastoplastic models are introduced.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import TYPE_CHECKING

import numpy as np

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.materials.linear_elastic import LinearElastic, _particle_moduli
from geopympm.solvers.explicit import ExplicitMPMSolver

if TYPE_CHECKING:
    from .output_manager import OutputManager

FORMAT = "geopympm-checkpoint"
SCHEMA_VERSION = 1

# Explicit, versioned state manifest. Never silently serialize __dict__ without
# validating it: new plastic internal arrays require a schema change.
_PARTICLE_FIELDS = (
    "position", "position0", "mass", "volume", "volume0", "velocity",
    "material_id", "stress", "strain", "domain_size", "domain_size0",
    "deformation_gradient",
)
_GRID_FIELDS = (
    "origin", "cell_size", "node_coordinates", "mass", "momentum",
    "velocity", "acceleration", "force_internal", "force_external",
    "velocity_bc_mask", "velocity_bc_value",
)
_GRID_META = ("n_cells", "node_shape", "n_nodes")
_SOLVER_FIELDS = (
    "particles", "grid", "materials", "dt", "pic_fraction", "cfl",
    "gravity", "nodal_external_force", "stress_update", "kinematics_update",
    "advection_update", "interpolation", "time", "step_count",
)
_NAME = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")


@dataclass(frozen=True)
class CheckpointRecord:
    """Paths and basic information for a committed checkpoint pair."""

    data_path: Path
    manifest_path: Path
    step: int
    time: float
    n_particles: int
    array_count: int
    bytes_written: int


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_extra_attributes(obj: object, expected: tuple[str, ...], label: str) -> None:
    unexpected = set(vars(obj)) - set(expected)
    missing = set(expected) - set(vars(obj))
    if unexpected or missing:
        raise ValueError(
            f"Unregistered {label} state: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}. Update checkpoint schema first."
        )


def _collect_state(solver: ExplicitMPMSolver) -> tuple[dict[str, np.ndarray], dict]:
    if not isinstance(solver, ExplicitMPMSolver):
        raise TypeError("solver must be an ExplicitMPMSolver")
    p, g = solver.particles, solver.grid
    if not isinstance(p, ParticleSet) or not isinstance(g, BackgroundGrid):
        raise TypeError("solver must contain ParticleSet and BackgroundGrid")
    _reject_extra_attributes(p, _PARTICLE_FIELDS, "ParticleSet")
    _reject_extra_attributes(g, (*_GRID_FIELDS, *_GRID_META), "BackgroundGrid")
    _reject_extra_attributes(solver, _SOLVER_FIELDS, "ExplicitMPMSolver")
    if p.ndim != 2 or g.ndim != 2:
        raise ValueError("Only Stage-10G 2D MPM is supported")
    if type(solver.materials) is LinearElastic:
        material_spec = {"kind": "single", "model": "LinearElastic",
                         "parameters": {"young_modulus": solver.materials.young_modulus,
                                        "poisson_ratio": solver.materials.poisson_ratio}}
    elif isinstance(solver.materials, Mapping):
        items = []
        for key, model in sorted(solver.materials.items()):
            if (isinstance(key, (bool, np.bool_)) or not isinstance(key, (int, np.integer))
                    or not 0 <= int(key) <= np.iinfo(np.int32).max
                    or type(model) is not LinearElastic):
                raise TypeError("Only integer-ID LinearElastic materials are checkpointable")
            items.append({"id": int(key), "model": "LinearElastic",
                          "parameters": {"young_modulus": model.young_modulus,
                                         "poisson_ratio": model.poisson_ratio}})
        material_spec = {"kind": "mapping", "items": items}
    else:
        raise TypeError("Unsupported constitutive model; refusing incomplete checkpoint")
    _particle_moduli(p, solver.materials)
    if (isinstance(solver.step_count, (bool, np.bool_))
            or not isinstance(solver.step_count, (int, np.integer))
            or solver.step_count < 0 or not math.isfinite(float(solver.time))
            or solver.time < 0):
        raise ValueError("Solver must have a finite committed step/time")

    arrays: dict[str, np.ndarray] = {}
    for prefix, obj, fields in (("particle", p, _PARTICLE_FIELDS),
                                ("grid", g, _GRID_FIELDS)):
        for field in fields:
            values = getattr(obj, field)
            if not isinstance(values, np.ndarray):
                raise ValueError(f"{prefix}.{field} is not an ndarray")
            if values.dtype.hasobject or values.dtype.kind not in "fiub":
                raise ValueError(f"{prefix}.{field} uses unsupported dtype {values.dtype}")
            if values.dtype.kind == "f" and not np.all(np.isfinite(values)):
                raise ValueError(f"{prefix}.{field} contains non-finite values")
            arrays[f"{prefix}.{field}"] = np.array(values, order="C", copy=True)
    n, nn = p.n_particles, g.n_nodes
    particle_shapes = {
        "position": (n, 2), "position0": (n, 2),
        "mass": (n,), "volume": (n,), "volume0": (n,),
        "velocity": (n, 2), "material_id": (n,),
        "stress": (n, 6), "strain": (n, 6),
        "domain_size": (n, 2), "domain_size0": (n, 2),
        "deformation_gradient": (n, 2, 2),
    }
    grid_shapes = {
        "origin": (2,), "cell_size": (2,), "node_coordinates": (nn, 2),
        "mass": (nn,), "momentum": (nn, 2), "velocity": (nn, 2),
        "acceleration": (nn, 2), "force_internal": (nn, 2),
        "force_external": (nn, 2), "velocity_bc_mask": (nn, 2),
        "velocity_bc_value": (nn, 2),
    }
    for field, shape in particle_shapes.items():
        if arrays[f"particle.{field}"].shape != shape:
            raise ValueError(f"Particle state {field} must have shape {shape}")
    for field, shape in grid_shapes.items():
        if arrays[f"grid.{field}"].shape != shape:
            raise ValueError(f"Grid state {field} must have shape {shape}")
    if p.material_id.dtype != np.dtype("int32") or np.any(p.material_id < 0):
        raise ValueError("Particle material_id must be nonnegative int32")
    if g.n_nodes != (g.n_cells[0] + 1) * (g.n_cells[1] + 1):
        raise ValueError("Grid n_nodes and n_cells are inconsistent")
    if tuple(g.node_shape) != (g.n_cells[0] + 1, g.n_cells[1] + 1):
        raise ValueError("Grid node_shape and n_cells are inconsistent")
    if p.mass.dtype.kind != "f" or p.volume.dtype.kind != "f":
        raise ValueError("Particle mass and volume must be floating point")
    if np.any(p.mass <= 0) or np.any(p.volume <= 0) or np.any(p.volume0 <= 0):
        raise ValueError("Particle masses and volumes must be positive")
    if g.velocity_bc_mask.dtype.kind != "b":
        raise ValueError("Grid velocity_bc_mask must be bool")
    if g.node_coordinates.shape != (g.n_nodes, 2):
        raise ValueError("Grid node count mismatch")
    if p.deformation_gradient.shape != (p.n_particles, 2, 2):
        raise ValueError("Particle deformation gradient shape mismatch")
    ext = solver.nodal_external_force
    if ext is not None:
        ext = np.asarray(ext)
        if ext.shape != (g.n_nodes, 2) or not np.all(np.isfinite(ext)):
            raise ValueError("Invalid nodal external force state")
        arrays["solver.nodal_external_force"] = np.array(ext, order="C", copy=True)
    gravity = np.asarray(solver.gravity)
    if gravity.shape != (2,) or not np.all(np.isfinite(gravity)):
        raise ValueError("Invalid solver gravity state")
    arrays["solver.gravity"] = np.array(gravity, order="C", copy=True)

    config = {
        "dt": float(solver.dt), "pic_fraction": float(solver.pic_fraction),
        "cfl": float(solver.cfl), "interpolation": solver.interpolation,
        "stress_update": solver.stress_update,
        "kinematics_update": solver.kinematics_update,
        "advection_update": solver.advection_update,
        "has_nodal_external_force": ext is not None,
    }
    grid = {"n_cells": list(g.n_cells), "node_shape": list(g.node_shape),
            "n_nodes": int(g.n_nodes)}
    return arrays, {"config": config, "materials": material_spec, "grid": grid,
                    "n_particles": int(p.n_particles), "ndim": int(p.ndim)}


def _output_state(manager: "OutputManager", solver: ExplicitMPMSolver) -> dict:
    # Capture the scheduler, not the actual VTK files. Stage 10G-B2 will
    # reconstruct the scheduler and existing indexes as a separate task.
    from .output_manager import OutputManager
    if not isinstance(manager, OutputManager):
        raise TypeError("output_manager must be an OutputManager")
    if (manager._last_seen_step != solver.step_count
            or manager._last_seen_time != solver.time):
        raise ValueError("OutputManager must have observed the last committed step")
    return {
        "directory": str(manager.directory),
        "enabled": list(manager.enabled),
        "every_n_steps": manager.every_n_steps,
        "every_dt": manager.every_dt,
        "next_time": manager._next_time,
        "fields": {key: list(fields) if fields is not None else None
                   for key, fields in manager._fields.items()},
        "initial_written": manager._initial_written,
        "last_seen_step": manager._last_seen_step,
        "last_seen_time": manager._last_seen_time,
        "entries": {kind: [dict(step=frame.step, time=frame.time,
                                filename=frame.filename)
                            for frame in frames]
                    for kind, frames in manager.frames.items()},
    }


def write_checkpoint(
    solver: ExplicitMPMSolver,
    directory: str | os.PathLike[str],
    *,
    prefix: str = "state",
    output_manager: "OutputManager | None" = None,
) -> CheckpointRecord:
    """Write a complete Stage-10G solver snapshot without modifying it.

    The NPZ is written first and its SHA-256 recorded in the JSON manifest,
    which is atomically published LAST. Presence of the manifest marks a
    complete pair. A hard crash between file renames can leave an orphan NPZ,
    but never a published manifest pointing to an unfinished NPZ.

    Refuses to overwrite an existing checkpoint. Call only after committed
    ``solver.step()``, or on a newly initialized solver at step zero.
    """
    if not isinstance(prefix, str) or not _NAME.fullmatch(prefix):
        raise ValueError("prefix must be ASCII alphanumeric, underscore or dash, starting with a letter")
    arrays, state = _collect_state(solver)
    if output_manager is not None:
        state["output_manager"] = _output_state(output_manager, solver)
    else:
        state["output_manager"] = None
    folder = Path(directory)
    stem = f"{prefix}_{int(solver.step_count):08d}"
    data_path = folder / f"{stem}.npz"
    manifest_path = folder / f"{stem}.json"
    if data_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Checkpoint already exists: {stem}")
    folder.mkdir(parents=True, exist_ok=True)

    tmp_data: Path | None = None
    tmp_manifest: Path | None = None
    published_data = False
    try:
        fd, tmp = tempfile.mkstemp(dir=folder, prefix=f".{stem}.", suffix=".npz.tmp")
        tmp_data = Path(tmp)
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        hash_value = _hash_file(tmp_data)
        array_manifest = {
            name: {"dtype": value.dtype.str, "shape": list(value.shape)}
            for name, value in sorted(arrays.items())
        }
        manifest = {
            "format": FORMAT,
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "step": int(solver.step_count),
            "time": float(solver.time),
            "data_file": data_path.name,
            "data_sha256": hash_value,
            "arrays": array_manifest,
            **state,
        }
        # Reject NaN/Infinity in JSON. Ensure all entries are basic JSON types.
        encoded = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)
                   + "\n").encode("utf-8")
        fd, tmp = tempfile.mkstemp(dir=folder, prefix=f".{stem}.", suffix=".json.tmp")
        tmp_manifest = Path(tmp)
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # No overwriting even if another process has inserted one; normal use
        # is a single writer per directory. The manifest is the commit marker.
        if data_path.exists() or manifest_path.exists():
            raise FileExistsError(f"Concurrent checkpoint collision: {stem}")
        os.replace(tmp_data, data_path)
        tmp_data = None
        published_data = True
        os.replace(tmp_manifest, manifest_path)
        tmp_manifest = None
    except Exception:
        if published_data and not manifest_path.exists():
            data_path.unlink(missing_ok=True)
        raise
    finally:
        if tmp_data is not None:
            tmp_data.unlink(missing_ok=True)
        if tmp_manifest is not None:
            tmp_manifest.unlink(missing_ok=True)
    return CheckpointRecord(
        data_path, manifest_path, int(solver.step_count), float(solver.time),
        solver.particles.n_particles, len(arrays),
        data_path.stat().st_size + manifest_path.stat().st_size,
    )


def inspect_checkpoint(manifest_path: str | os.PathLike[str], *, verify: bool = True) -> dict:
    """Read B1 metadata, optionally validate the archived numeric arrays.

    Does NOT restore a ParticleSet, grid, material, OutputManager, or solver.
    This deliberately makes Stage 10G-B1 a writer/inspection phase only.
    """
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if (not isinstance(manifest, dict) or manifest.get("format") != FORMAT
            or manifest.get("schema_version") != SCHEMA_VERSION):
        raise ValueError("Unsupported or invalid checkpoint manifest version")
    filename = manifest.get("data_file")
    if (not isinstance(filename, str) or Path(filename).name != filename
            or filename != path.stem + ".npz"):
        raise ValueError("Invalid or unsafe checkpoint data filename")
    if not verify:
        return manifest
    data_file = path.with_name(filename)
    expected = manifest.get("data_sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Invalid checkpoint SHA-256 metadata")
    if _hash_file(data_file) != expected:
        raise ValueError("Checkpoint data SHA-256 mismatch (damaged or modified archive)")
    field_manifest = manifest.get("arrays")
    if not isinstance(field_manifest, dict):
        raise ValueError("Missing checkpoint array manifest")
    required = {f"particle.{key}" for key in _PARTICLE_FIELDS}
    required.update(f"grid.{key}" for key in _GRID_FIELDS)
    required.add("solver.gravity")
    if manifest.get("config", {}).get("has_nodal_external_force"):
        required.add("solver.nodal_external_force")
    if set(field_manifest) != required:
        raise ValueError("Checkpoint is missing required state or has unknown array fields")
    with np.load(data_file, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError("Checkpoint NPZ array list does not match manifest")
        for name in sorted(required):
            value = archive[name]
            spec = field_manifest[name]
            if (value.dtype.str != spec["dtype"] or list(value.shape) != spec["shape"]
                    or value.dtype.hasobject):
                raise ValueError(f"Checkpoint array dtype/shape mismatch: {name}")
            if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
                raise ValueError(f"Checkpoint has non-finite array: {name}")
    return manifest


__all__ = ["CheckpointRecord", "write_checkpoint", "inspect_checkpoint", "SCHEMA_VERSION"]
