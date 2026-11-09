"""Stage 10G-A3: VisIt and ParaView time-series indexes for .vtu snapshots.

Each stream contains one VTK UnstructuredGrid file per simulation time.
VisIt .visit uses !TIME directives; ParaView .pvd uses a VTK Collection.
Only relative files in the index's own directory are accepted. Files are
not read into memory or modified by these index writers.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class SeriesFrame:
    step: int
    time: float
    filename: str

    def __post_init__(self):
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError("frame step must be a nonnegative integer")
        if isinstance(self.time, bool) or not math.isfinite(float(self.time)) or float(self.time) < 0:
            raise ValueError("frame time must be finite and nonnegative")
        if not isinstance(self.filename, str) or not self.filename.endswith(".vtu"):
            raise ValueError("frame filename must be a .vtu basename")
        if self.filename != Path(self.filename).name or any(c in self.filename for c in ("\\", "\n", "\r")):
            raise ValueError("frame filename must be a simple, safe relative basename")
        if self.filename.startswith("."):
            raise ValueError("frame filename may not start with '.'")


def _validated(frames: Sequence[SeriesFrame], directory: Path) -> tuple[SeriesFrame, ...]:
    items = tuple(frames)
    if not items:
        raise ValueError("time series must contain at least one frame")
    previous_step, previous_time = -1, -1.0
    used_names = set()
    for frame in items:
        if not isinstance(frame, SeriesFrame):
            raise TypeError("frames must contain SeriesFrame objects")
        if frame.step <= previous_step or float(frame.time) <= previous_time:
            raise ValueError("frames must have strictly increasing step and simulation time")
        if frame.filename in used_names:
            raise ValueError("frames may not repeat .vtu filenames")
        if not (directory / frame.filename).is_file():
            raise FileNotFoundError(f"Missing indexed VTU file: {directory / frame.filename}")
        previous_step, previous_time = frame.step, float(frame.time)
        used_names.add(frame.filename)
    return items


def _atomic_write(path: Path, payload: bytes) -> None:
    """Replace one index after complete write; never leave a partial index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{path.name}.",
                                         suffix=".tmp", dir=path.parent, delete=False) as handle:
            name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if name is not None and os.path.exists(name):
            os.unlink(name)


def render_visit(frames: Sequence[SeriesFrame], directory: Path) -> bytes:
    """Render .visit with explicit physical times, without multi-block mixing."""
    frames = _validated(frames, directory)
    return ("!NBLOCKS 1\n" + "".join(
        f"!TIME {repr(float(f.time))}\n{f.filename}\n" for f in frames)).encode("utf-8")


def render_pvd(frames: Sequence[SeriesFrame], directory: Path) -> bytes:
    frames = _validated(frames, directory)
    root = ET.Element("VTKFile", {"type": "Collection", "version": "0.1",
                                  "byte_order": "LittleEndian"})
    collection = ET.SubElement(root, "Collection")
    for f in frames:
        ET.SubElement(collection, "DataSet", {
            "timestep": repr(float(f.time)), "group": "", "part": "0", "file": f.filename,
        })
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"


def write_time_series(frames: Sequence[SeriesFrame], directory: str | Path,
                      stem: str) -> tuple[Path, Path]:
    """Update <stem>.visit + <stem>.pvd after all referenced VTUs exist.

    Caller owns the snapshots and must coordinate rollback if filesystem I/O
    fails after the first of these two index replacements. No file containing
    simulation state is overwritten by this function.
    """
    directory = Path(directory)
    if not stem or stem in (".", "..") or any(c in stem for c in ("/", "\\", "\n", "\r")):
        raise ValueError("stem must be a simple filename base")
    visit_data = render_visit(frames, directory)
    pvd_data = render_pvd(frames, directory)
    visit_path = directory / f"{stem}.visit"
    pvd_path = directory / f"{stem}.pvd"
    _atomic_write(visit_path, visit_data)
    _atomic_write(pvd_path, pvd_data)
    return visit_path, pvd_path


__all__ = ["SeriesFrame", "render_visit", "render_pvd", "write_time_series"]
