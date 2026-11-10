"""VTK XML particle writer for GeoPyMPM (Stage 10G-A1).

Each material point is a VTK_VERTEX cell in a VTK UnstructuredGrid (.vtu).
The exporter is deliberately independent of solver integration and of VTK
Python bindings; its ASCII Float64 values round-trip without loss of precision.

The solver uses tension-positive Cauchy stress in Voigt order
[xx, yy, zz, xy, yz, xz]. Strain has the same order but ENGINEERING shears.
The full tensors exported to VTK use row-major 3x3 components, with strain
engineering shear divided by two. For 2D, vectors are padded with z=0 and
F is embedded in 3D using F_zz=1 (a plane-strain visualization convention).
"""

from __future__ import annotations

from collections.abc import Iterable
from io import StringIO
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import numpy as np

from geopympm.core.particles import ParticleSet


# Public names are explicit, stable, and used as VTK PointData arrays.
DEFAULT_FIELDS = (
    "particle_id", "material_id", "mass", "volume", "volume0", "density",
    "position0", "displacement", "velocity", "stress_voigt", "strain_voigt",
    "stress", "strain", "deformation_gradient", "jacobian", "domain_size",
    "domain_size0", "mean_stress", "pressure", "von_mises",
)


def _pad_vectors(values: np.ndarray, ndim: int) -> np.ndarray:
    result = np.zeros((values.shape[0], 3), dtype=np.float64)
    result[:, :ndim] = values
    return result


def _tensor_from_voigt(values: np.ndarray, *, engineering_shear: bool) -> np.ndarray:
    n = values.shape[0]
    result = np.zeros((n, 3, 3), dtype=np.float64)
    result[:, 0, 0] = values[:, 0]
    result[:, 1, 1] = values[:, 1]
    result[:, 2, 2] = values[:, 2]
    factor = 0.5 if engineering_shear else 1.0
    for i, j, k in ((0, 1, 3), (1, 2, 4), (0, 2, 5)):
        result[:, i, j] = result[:, j, i] = factor * values[:, k]
    return result.reshape(n, 9)


def _deformation_gradient_3d(values: np.ndarray, ndim: int) -> np.ndarray:
    n = values.shape[0]
    if ndim == 3:
        return values.reshape(n, 9)
    result = np.zeros((n, 3, 3), dtype=np.float64)
    result[:, :2, :2] = values
    result[:, 2, 2] = 1.0
    return result.reshape(n, 9)


def _validate_state(particles: ParticleSet) -> None:
    if not isinstance(particles, ParticleSet):
        raise TypeError("particles must be a ParticleSet")
    n, d = particles.n_particles, particles.ndim
    shapes = {
        "position": (n, d), "position0": (n, d), "velocity": (n, d),
        "mass": (n,), "volume": (n,), "volume0": (n,),
        "material_id": (n,), "stress": (n, 6), "strain": (n, 6),
        "domain_size": (n, d), "domain_size0": (n, d),
        "deformation_gradient": (n, d, d),
    }
    for name, shape in shapes.items():
        data = np.asarray(getattr(particles, name))
        if data.shape != shape:
            raise ValueError(f"{name}: expected shape {shape}, got {data.shape}")
        if not np.all(np.isfinite(data)):
            raise ValueError(f"{name} contains non-finite values")
    if np.any(particles.mass <= 0) or np.any(particles.volume <= 0) or np.any(particles.volume0 <= 0):
        raise ValueError("mass, volume, and volume0 must be positive")
    if not np.issubdtype(particles.material_id.dtype, np.integer) or np.any(particles.material_id < 0):
        raise ValueError("material_id must contain nonnegative integers")


def _field_arrays(particles: ParticleSet) -> dict[str, np.ndarray]:
    n, d = particles.n_particles, particles.ndim
    sig = particles.stress
    trace = sig[:, 0] + sig[:, 1] + sig[:, 2]
    equivalent = np.sqrt(np.maximum(0.0, (
        0.5 * ((sig[:, 0] - sig[:, 1]) ** 2
             + (sig[:, 1] - sig[:, 2]) ** 2
             + (sig[:, 2] - sig[:, 0]) ** 2)
        + 3.0 * (sig[:, 3] ** 2 + sig[:, 4] ** 2 + sig[:, 5] ** 2)
    )))
    data = {
        "particle_id": np.arange(n, dtype=np.int64),
        "material_id": particles.material_id,
        "mass": particles.mass,
        "volume": particles.volume,
        "volume0": particles.volume0,
        "density": particles.mass / particles.volume,
        "position0": _pad_vectors(particles.position0, d),
        "displacement": _pad_vectors(particles.displacement, d),
        "velocity": _pad_vectors(particles.velocity, d),
        "stress_voigt": sig,
        "strain_voigt": particles.strain,
        "stress": _tensor_from_voigt(sig, engineering_shear=False),
        "strain": _tensor_from_voigt(particles.strain, engineering_shear=True),
        "deformation_gradient": _deformation_gradient_3d(particles.deformation_gradient, d),
        "jacobian": np.linalg.det(particles.deformation_gradient),
        "domain_size": _pad_vectors(particles.domain_size, d),
        "domain_size0": _pad_vectors(particles.domain_size0, d),
        "mean_stress": trace / 3.0,
        "pressure": -trace / 3.0,  # compression positive, core stress tension positive
        "von_mises": equivalent,
    }
    return data


def _append_array(parent: ET.Element, name: str | None, data: np.ndarray) -> None:
    """Serialize one n-tuple array, using round-trip-safe float64 decimals."""
    array = np.asarray(data)
    if array.ndim == 1:
        matrix = array.reshape(-1, 1)
    elif array.ndim == 2:
        matrix = array
    else:
        raise ValueError("VTK DataArray must have shape (n,) or (n,components)")
    if np.issubdtype(matrix.dtype, np.floating):
        xml_type = "Float64"
        if matrix.shape[0] >= 256:
            # D4: format large float arrays through NumPy's C-backed formatter.
            # %.17g preserves the Float64 round-trip precision of format(x, '.17g').
            # In-memory StringIO avoids a Python list of one string per particle.
            buffer = StringIO()
            np.savetxt(buffer, matrix, fmt="%.17g")
            serialized = "\n" + buffer.getvalue()
        else:
            lines = [" ".join(format(float(x), ".17g") for x in row) for row in matrix]
            serialized = "\n" + "\n".join(lines) + "\n"
    elif np.issubdtype(matrix.dtype, np.integer):
        xml_type = "Int32" if matrix.dtype == np.int32 else ("UInt8" if matrix.dtype == np.uint8 else "Int64")
        lines = [" ".join(str(int(x)) for x in row) for row in matrix]
        serialized = "\n" + "\n".join(lines) + "\n"
    else:
        raise TypeError(f"Unsupported VTK numerical array dtype {matrix.dtype}")
    attrs = {"type": xml_type, "format": "ascii"}
    if name is not None:
        attrs["Name"] = name
    if parent.tag == "FieldData":
        attrs["NumberOfTuples"] = str(matrix.shape[0])
    if matrix.shape[1] != 1:
        attrs["NumberOfComponents"] = str(matrix.shape[1])
    elem = ET.SubElement(parent, "DataArray", attrs)
    elem.text = serialized


def write_particles_vtu(
    particles: ParticleSet,
    filename: str | Path,
    *,
    time: float | None = None,
    step: int | None = None,
    fields: Iterable[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write one particle snapshot to a VTK XML UnstructuredGrid file.

    The file contains n VTK_VERTEX cells, one per particle. By default every
    universal particle field is included. The optional ``fields`` argument
    selects a subset of PointData names. Does not change particle/solver state.

    ``time`` (s) and ``step`` are optional snapshot metadata; the writer does
    not manage an animation collection or schedule (planned Stage 10G-A3).
    Existing files are preserved unless ``overwrite=True``.
    """
    _validate_state(particles)
    path = Path(filename)
    if path.suffix.lower() != ".vtu":
        raise ValueError("Particle VTK output filename must end in .vtu")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing particle output: {path}")
    if time is not None:
        if isinstance(time, (bool, np.bool_)) or not math.isfinite(float(time)) or float(time) < 0:
            raise ValueError("time must be finite and nonnegative")
    if step is not None:
        if isinstance(step, (bool, np.bool_)) or not isinstance(step, (int, np.integer)) or step < 0:
            raise ValueError("step must be a nonnegative integer")

    selected = DEFAULT_FIELDS if fields is None else tuple(fields)
    if len(selected) != len(set(selected)) or any(name not in DEFAULT_FIELDS for name in selected):
        raise ValueError(f"fields must be unique names from {DEFAULT_FIELDS}")
    all_fields = _field_arrays(particles)
    n = particles.n_particles
    root = ET.Element("VTKFile", {"type": "UnstructuredGrid", "version": "1.0", "byte_order": "LittleEndian"})
    ug = ET.SubElement(root, "UnstructuredGrid")
    if time is not None or step is not None:
        fd = ET.SubElement(ug, "FieldData")
        if time is not None:
            _append_array(fd, "TimeValue", np.array([float(time)], dtype=np.float64))
        if step is not None:
            _append_array(fd, "Step", np.array([int(step)], dtype=np.int64))
    piece = ET.SubElement(ug, "Piece", {"NumberOfPoints": str(n), "NumberOfCells": str(n)})
    pd = ET.SubElement(piece, "PointData")
    for name in selected:
        values = all_fields[name]
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Export field {name} contains non-finite values")
        _append_array(pd, name, values)
    ET.SubElement(piece, "CellData")
    points = ET.SubElement(piece, "Points")
    _append_array(points, None, _pad_vectors(particles.position, particles.ndim))
    cells = ET.SubElement(piece, "Cells")
    _append_array(cells, "connectivity", np.arange(n, dtype=np.int64))
    _append_array(cells, "offsets", np.arange(1, n + 1, dtype=np.int64))
    _append_array(cells, "types", np.ones(n, dtype=np.uint8))  # VTK_VERTEX=1
    ET.indent(root, space="  ")
    xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb" if overwrite else "xb") as stream:
        stream.write(xml)
    return path


__all__ = ["DEFAULT_FIELDS", "write_particles_vtu"]
