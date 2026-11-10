"""Stage 10G-B2: validated restart from Stage 10G-B1 checkpoints.

The checkpoint manifest is the commit marker. Reconstruction accepts
schema 1 (2D LinearElastic) and schema 2 (2D CompressibleNeoHookean) and fails
closed on incompatible dimensions, material specifications or state arrays.

Two independent operations are offered:
    solver = load_checkpoint('state_00000006.json')
    output = restore_output_manager('state_00000006.json', solver)

Output-manager restoration is optional because many users only need to
continue numerical computation, not append an existing visualization series.
The output directory MUST still describe exactly the saved animation state;
B2 never truncates or overwrites newer results automatically.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import os
from pathlib import Path
import re

import numpy as np

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.materials.linear_elastic import LinearElastic
from geopympm.materials.neo_hookean import CompressibleNeoHookean
from geopympm.solvers.explicit import ExplicitMPMSolver
from .checkpoint import (
    _PARTICLE_FIELDS, _GRID_FIELDS, _collect_state, _hash_file, inspect_checkpoint,
)
from .output_manager import OutputManager
from .time_series import SeriesFrame, render_visit, render_pvd

# These are the exact dtypes and shapes produced by the current solver
# containers. Reject future schema variants explicitly rather than cast.
_CONFIG_KEYS = {
    "dt", "pic_fraction", "cfl", "interpolation", "stress_update",
    "kinematics_update", "advection_update", "has_nodal_external_force",
}


def _validate_checkpoint_identity(meta: dict) -> tuple[int, float, int, tuple[int, int]]:
    step, time, count = meta.get("step"), meta.get("time"), meta.get("n_particles")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("Invalid checkpoint step")
    if isinstance(time, bool) or not isinstance(time, (int, float)) or not math.isfinite(time) or time < 0:
        raise ValueError("Invalid checkpoint simulation time")
    if (step == 0) != (time == 0):
        raise ValueError("Checkpoint step zero and time zero must coincide")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Invalid checkpoint particle count")
    if meta.get("ndim") != 2:
        raise ValueError("Stage 10G-B2 supports only 2D checkpoints")
    grid = meta.get("grid")
    if not isinstance(grid, dict) or set(grid) != {"n_cells", "node_shape", "n_nodes"}:
        raise ValueError("Invalid checkpoint grid metadata")
    cells = grid["n_cells"]
    if (not isinstance(cells, list) or len(cells) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in cells)):
        raise ValueError("Invalid checkpoint grid dimensions")
    nx, ny = cells
    if (grid["node_shape"] != [nx + 1, ny + 1]
            or grid["n_nodes"] != (nx + 1) * (ny + 1)):
        raise ValueError("Checkpoint node counts do not match grid dimensions")
    return step, float(time), count, (nx, ny)


def _material_from_spec(spec: object) -> LinearElastic | CompressibleNeoHookean | dict:
    if not isinstance(spec, dict) or spec.get("kind") not in ("single", "mapping"):
        raise ValueError("Unsupported material checkpoint kind")

    def _make(entry: object) -> LinearElastic | CompressibleNeoHookean:
        if not isinstance(entry, dict) or set(entry) != {"model", "parameters"}:
            raise ValueError("Invalid material checkpoint")
        p = entry["parameters"]
        if not isinstance(p, dict):
            raise ValueError("Invalid material checkpoint parameters")
        if entry["model"] == "LinearElastic":
            if set(p) != {"young_modulus", "poisson_ratio"}:
                raise ValueError("Invalid LinearElastic material parameters")
            return LinearElastic(p["young_modulus"], p["poisson_ratio"])
        if entry["model"] == "CompressibleNeoHookean":
            if set(p) != {"bulk_modulus", "shear_modulus"}:
                raise ValueError("Invalid CNH material parameters")
            return CompressibleNeoHookean(p["bulk_modulus"], p["shear_modulus"])
        raise ValueError("Unsupported material checkpoint model")

    if spec["kind"] == "single":
        if set(spec) != {"kind", "model", "parameters"}:
            raise ValueError("Unexpected single material metadata")
        return _make({"model": spec["model"], "parameters": spec["parameters"]})
    if set(spec) != {"kind", "items"} or not isinstance(spec["items"], list) or not spec["items"]:
        raise ValueError("Invalid material mapping in checkpoint")
    materials: dict[int, LinearElastic | CompressibleNeoHookean] = {}
    for item in spec["items"]:
        if not isinstance(item, dict) or set(item) != {"id", "model", "parameters"}:
            raise ValueError("Invalid material mapping entry")
        key = item["id"]
        if (isinstance(key, bool) or not isinstance(key, int)
                or not 0 <= key <= np.iinfo(np.int32).max or key in materials):
            raise ValueError("Duplicate or invalid material mapping ID")
        materials[key] = _make({"model": item["model"], "parameters": item["parameters"]})
    return materials


def _checked_array(data: Mapping, name: str, shape: tuple[int, ...], dtype: str) -> np.ndarray:
    item = data[name]
    if item.shape != shape or item.dtype != np.dtype(dtype):
        raise ValueError(
            f"Checkpoint {name} expected {shape} {dtype}, got {item.shape} {item.dtype}"
        )
    # inspect_checkpoint already checked every array's finiteness and hash.
    return np.array(item, order="C", copy=True)


def load_checkpoint(manifest_path: str | os.PathLike[str]) -> ExplicitMPMSolver:
    """Restore a runnable solver, without modifying or deleting checkpoint files.

    Full SHA-256, manifest/NPZ schema, shapes, dtypes and constitutive
    configurations are checked. The solver is re-created by public class
    constructors before the exact saved state arrays and clock are restored.
    This deliberately restores *all* transient nodal arrays too, even though
    the current solver reassembles them at the next step.

    Output scheduling is separate: call ``restore_output_manager(...)`` if
    ``write_checkpoint(..., output_manager=...)`` captured a series.
    """
    meta = inspect_checkpoint(manifest_path, verify=True)
    step, time, n, cells = _validate_checkpoint_identity(meta)
    g_spec = meta["grid"]
    nn = g_spec["n_nodes"]
    config = meta.get("config")
    schema_version = meta["schema_version"]
    expected_keys = (_CONFIG_KEYS | {"scheme"}) if schema_version == 3 else _CONFIG_KEYS
    if not isinstance(config, dict) or set(config) != expected_keys:
        raise ValueError("Invalid checkpoint solver configuration")
    if type(config["has_nodal_external_force"]) is not bool:
        raise ValueError("Invalid nodal external force presence flag")
    materials = _material_from_spec(meta.get("materials"))
    p_shapes = {
        "position": (n, 2), "position0": (n, 2),
        "mass": (n,), "volume": (n,), "volume0": (n,),
        "velocity": (n, 2), "material_id": (n,),
        "stress": (n, 6), "strain": (n, 6),
        "domain_size": (n, 2), "domain_size0": (n, 2),
        "deformation_gradient": (n, 2, 2),
    }
    g_shapes = {
        "origin": (2,), "cell_size": (2,), "node_coordinates": (nn, 2),
        "mass": (nn,), "momentum": (nn, 2), "velocity": (nn, 2),
        "acceleration": (nn, 2), "force_internal": (nn, 2),
        "force_external": (nn, 2), "velocity_bc_mask": (nn, 2),
        "velocity_bc_value": (nn, 2),
    }
    archive_path = Path(manifest_path).with_name(meta["data_file"])
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {}
        for field in _PARTICLE_FIELDS:
            name = f"particle.{field}"
            arrays[name] = _checked_array(
                archive, name, p_shapes[field], "int32" if field == "material_id" else "float64",
            )
        for field in _GRID_FIELDS:
            name = f"grid.{field}"
            arrays[name] = _checked_array(
                archive, name, g_shapes[field], "bool" if field == "velocity_bc_mask" else "float64",
            )
        arrays["solver.gravity"] = _checked_array(archive, "solver.gravity", (2,), "float64")
        if config["has_nodal_external_force"]:
            arrays["solver.nodal_external_force"] = _checked_array(
                archive, "solver.nodal_external_force", (nn, 2), "float64",
            )

    # Constructors validate base geometry/material combinations. Then restore
    # every stored ndarray in place, not by assigning aliases to NPZ memory.
    p = ParticleSet(arrays["particle.position0"], arrays["particle.mass"],
                    arrays["particle.volume0"],
                    velocity=arrays["particle.velocity"],
                    material_id=arrays["particle.material_id"],
                    stress=arrays["particle.stress"], strain=arrays["particle.strain"])
    for field in _PARTICLE_FIELDS:
        getattr(p, field)[:] = arrays[f"particle.{field}"]

    g = BackgroundGrid(arrays["grid.origin"], arrays["grid.cell_size"], cells)
    if not np.array_equal(g.node_coordinates, arrays["grid.node_coordinates"]):
        raise ValueError("Archived background grid geometry is inconsistent with structured grid")
    for field in _GRID_FIELDS:
        getattr(g, field)[:] = arrays[f"grid.{field}"]

    solver = ExplicitMPMSolver(
        p, g, materials, dt=config["dt"],
        pic_fraction=config["pic_fraction"], cfl=config["cfl"],
        gravity=arrays["solver.gravity"],
        nodal_external_force=arrays.get("solver.nodal_external_force"),
        stress_update=config["stress_update"],
        kinematics_update=config["kinematics_update"],
        advection_update=config["advection_update"],
        interpolation=config["interpolation"],
        scheme=config.get("scheme", "usl"),
    )
    solver.step_count = step
    solver.time = time

    # The Stage B1 state inventory is also our exact post-reconstruction check.
    # This catches constructor changes that silently reset or normalize state.
    restored, state = _collect_state(solver)
    for key, original in arrays.items():
        if not np.array_equal(original, restored[key]):
            raise ValueError(f"Restored solver state differs from checkpoint: {key}")
    if (state["config"] != config or state["materials"] != meta["materials"]
            or state["grid"] != meta["grid"] or state["n_particles"] != n):
        raise ValueError("Restored solver configuration differs from checkpoint")
    return solver


def restore_output_manager(
    manifest_path: str | os.PathLike[str],
    solver: ExplicitMPMSolver,
    *,
    directory: str | os.PathLike[str] | None = None,
    allow_unverified_legacy_vtu: bool = False,
) -> OutputManager:
    """Resume the *exact* archived VisIt/ParaView series without overwriting it.

    Requires a checkpoint written WITH ``output_manager=...``. Existing
    .visit/.pvd indexes must match the checkpoint's frame lists byte-for-byte;
    an output folder that progressed past the checkpoint is deliberately
    rejected. To resume a branch, copy the checkpoint-era visualization folder
    and use ``directory=...``. New D2 checkpoints require SHA-256 verification
    of every VTU snapshot. To resume older checkpoints WITHOUT VTU hashes,
    explicitly set ``allow_unverified_legacy_vtu=True`` (unsafe for altered
    VTU contents). No files are changed by this function.
    """
    if type(allow_unverified_legacy_vtu) is not bool:
        raise TypeError("allow_unverified_legacy_vtu must be a bool")
    meta = inspect_checkpoint(manifest_path, verify=True)
    step, time, _, _ = _validate_checkpoint_identity(meta)
    if (not isinstance(solver, ExplicitMPMSolver) or solver.step_count != step
            or solver.time != time):
        raise ValueError("Output restart requires solver at the checkpoint step and time")
    state = meta.get("output_manager")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain OutputManager state")
    expected_keys = {"directory", "enabled", "every_n_steps", "every_dt", "next_time",
                     "fields", "initial_written", "last_seen_step", "last_seen_time", "entries"}
    has_integrity = "vtu_integrity" in state
    if set(state) != (expected_keys | {"vtu_integrity"} if has_integrity else expected_keys):
        raise ValueError("Invalid OutputManager checkpoint metadata")
    if state["last_seen_step"] != step or state["last_seen_time"] != time:
        raise ValueError("OutputManager checkpoint is not synchronized with solver")
    enabled = state["enabled"]
    if (not isinstance(enabled, list) or not enabled or len(set(map(str, enabled))) != len(enabled)
            or any(v not in ("particles", "grid", "domains") for v in enabled)
            or enabled != [v for v in ("particles", "grid", "domains") if v in enabled]):
        raise ValueError("Invalid OutputManager enabled streams")
    if "domains" in enabled and solver.interpolation != "cpdi":
        raise ValueError("CPDI domain output requires CPDI solver")
    every_n = state["every_n_steps"]
    every_dt = state["every_dt"]
    if (every_n is not None and (type(every_n) is not int or every_n <= 0)):
        raise ValueError("Invalid OutputManager step schedule")
    if (every_dt is not None and (isinstance(every_dt, bool) or not isinstance(every_dt, (int, float))
                                  or not math.isfinite(every_dt) or every_dt <= 0)):
        raise ValueError("Invalid OutputManager physical-time schedule")
    if every_n is None and every_dt is None:
        raise ValueError("OutputManager has no schedule")
    next_time = state["next_time"]
    if every_dt is None:
        if next_time is not None:
            raise ValueError("Unexpected next physical-time target")
    elif (isinstance(next_time, bool) or not isinstance(next_time, (int, float))
          or not math.isfinite(next_time) or next_time <= 0
          or next_time + 8 * math.ulp(max(abs(time), abs(next_time))) < time):
        raise ValueError("Invalid next physical-time target")
    if type(state["initial_written"]) is not bool:
        raise ValueError("Invalid initial_written flag")
    field_spec = state["fields"]
    if (not isinstance(field_spec, dict) or set(field_spec) != {"particles", "grid", "domains"}):
        raise ValueError("Invalid OutputManager field configuration")
    fields: dict[str, tuple[str, ...] | None] = {}
    for kind, items in field_spec.items():
        if items is not None and (not isinstance(items, list) or any(type(x) is not str for x in items)):
            raise ValueError(f"Invalid {kind} field selection")
        fields[kind] = None if items is None else tuple(items)

    raw_dir = state["directory"] if directory is None else directory
    if not isinstance(raw_dir, (str, os.PathLike)):
        raise ValueError("Invalid visualization directory")
    folder = Path(raw_dir)
    raw_entries = state["entries"]
    if not isinstance(raw_entries, dict) or set(raw_entries) != set(enabled):
        raise ValueError("Invalid OutputManager frame dictionary")
    # OutputManager need not create its folder before the first scheduled frame.
    # In that case restart can continue using the same nonexistent folder.
    if not folder.is_dir() and (folder.exists() or any(raw_entries.values())):
        raise FileNotFoundError(f"Visualization directory missing or invalid: {folder}")
    entries = {}
    for kind in enabled:
        raw_frames = raw_entries[kind]
        if not isinstance(raw_frames, list):
            raise ValueError("Invalid OutputManager frame list")
        frames = []
        for raw in raw_frames:
            if not isinstance(raw, dict) or set(raw) != {"step", "time", "filename"}:
                raise ValueError("Invalid output frame metadata")
            frame = SeriesFrame(**raw)
            if (frame.step > step or frame.time > time
                    or frame.filename != f"{kind}_{frame.step:06d}.vtu"):
                raise ValueError("Output frame is ahead of checkpoint or has wrong filename")
            frames.append(frame)
        if state["initial_written"]:
            if not frames or frames[0].step != 0 or frames[0].time != 0:
                raise ValueError("Missing output initial frame")
        elif frames and frames[0].step == 0:
            raise ValueError("Unexpected initial frame")
        actual_vtu = {p.name for p in folder.glob(f"{kind}_*.vtu") if p.is_file()}
        if actual_vtu != {f.filename for f in frames}:
            raise ValueError(f"Output directory has extra/missing {kind} snapshots; cannot restart safely")
        for ext, render in (("visit", render_visit), ("pvd", render_pvd)):
            pathname = folder / f"{kind}.{ext}"
            if frames:
                if not pathname.is_file() or pathname.read_bytes() != render(frames, folder):
                    raise ValueError(f"Existing {kind}.{ext} does not match checkpoint series")
            elif pathname.exists():
                raise ValueError(f"Unexpected {kind}.{ext} without any frames")
        entries[kind] = frames

    # Fail closed for VTU content integrity. Older B1/B2/C manifests had no
    # per-file hashes: continuing visualization is opt-in, not silently trusted.
    if has_integrity:
        integrity = state["vtu_integrity"]
        if (not isinstance(integrity, dict)
                or set(integrity) != {"version", "algorithm", "files"}
                or type(integrity["version"]) is not int or integrity["version"] != 1
                or integrity["algorithm"] != "sha256"
                or not isinstance(integrity["files"], dict)
                or set(integrity["files"]) != set(enabled)):
            raise ValueError("Invalid or unsupported VTU integrity metadata")
        for kind, frames in entries.items():
            hashes = integrity["files"][kind]
            expected_names = {f.filename for f in frames}
            if (not isinstance(hashes, dict) or set(hashes) != expected_names):
                raise ValueError(f"Missing or extra VTU SHA-256 hashes for {kind}")
            for filename, expected in hashes.items():
                if (not isinstance(expected, str)
                        or re.fullmatch(r"[0-9a-f]{64}", expected) is None):
                    raise ValueError(f"Invalid VTU SHA-256 for {filename}")
                path = folder / filename
                if path.is_symlink():
                    raise ValueError(f"Refusing symlinked VTU snapshot: {path}")
                if _hash_file(path) != expected:
                    raise ValueError(f"VTU SHA-256 mismatch: {filename} (modified/corrupted)")
    elif any(entries.values()) and not allow_unverified_legacy_vtu:
        raise ValueError(
            "Legacy checkpoint lacks VTU SHA-256 hashes; restore refused. "
            "Use allow_unverified_legacy_vtu=True only if the visualizations "
            "are trusted."
        )

    # Re-create ONLY after all files and scheduling constraints are checked.
    # __init__ intentionally rejects existing series; it is not a restart API.
    manager = OutputManager.__new__(OutputManager)
    manager.directory = folder
    manager.enabled = tuple(enabled)
    manager.every_n_steps = every_n
    manager.every_dt = None if every_dt is None else float(every_dt)
    manager._next_time = None if next_time is None else float(next_time)
    manager._fields = fields
    manager._entries = entries
    manager._last_seen_step = step
    manager._last_seen_time = time
    manager._initial_written = state["initial_written"]
    return manager


__all__ = ["load_checkpoint", "restore_output_manager"]
