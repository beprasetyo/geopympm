"""Stage 13A: explicit, versionable per-particle constitutive history arrays.

Model-specific history stays outside ParticleSet. All fields are owned float64
structure-of-arrays, with particle index on axis zero. A future plastic model
must declare and actually update its fields; adding fields here does NOT add
plasticity to the existing elastic material models.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

_FIELD_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class StateFieldSpec:
    """Definition of a history field, with non-particle component dimensions.

    ``StateFieldSpec('equivalent_plastic_strain')`` => shape (n,).
    ``StateFieldSpec('plastic_strain', (6,))`` => shape (n, 6).
    Only float64 is supported until a versioned schema explicitly adds dtypes.
    """
    name: str
    component_shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _FIELD_NAME.fullmatch(self.name):
            raise ValueError("History field name must match [a-z][a-z0-9_]*")
        shape = self.component_shape
        if not isinstance(shape, tuple) or len(shape) > 3 or any(
            type(dim) is not int or dim < 1 for dim in shape
        ):
            raise ValueError("component_shape must be a tuple of 1–3 positive dimensions, or ()")


class MaterialState:
    """Owned, validated collection of constitutive history arrays.

    Data are writable through ``state['name']`` for efficient Numba kernels.
    Call ``validate`` after updates. ``snapshot``/``restore`` preserve array
    identity, which is essential for solver rollback and external references.
    It is the caller's responsibility to define physical units and update law.
    """

    def __init__(
        self,
        n_particles: int,
        fields: tuple[StateFieldSpec, ...] | list[StateFieldSpec] = (),
        *,
        values: dict[str, np.ndarray] | None = None,
    ) -> None:
        if type(n_particles) is not int or n_particles < 1:
            raise ValueError("n_particles must be a positive Python integer")
        if not isinstance(fields, (list, tuple)) or any(
            type(f) is not StateFieldSpec for f in fields
        ):
            raise TypeError("fields must be a sequence of StateFieldSpec")
        specs = tuple(fields)
        names = [f.name for f in specs]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate material state field names")
        self.n_particles = n_particles
        self._fields = specs
        if values is not None and (not isinstance(values, dict) or set(values) != set(names)):
            raise ValueError("values must have exactly the declared history field names")
        self._arrays: dict[str, np.ndarray] = {}
        for field in specs:
            shape = (n_particles, *field.component_shape)
            if values is None:
                a = np.zeros(shape, dtype=np.float64, order="C")
            else:
                raw = np.asarray(values[field.name])
                if raw.dtype != np.dtype('float64') or raw.shape != shape or not np.all(np.isfinite(raw)):
                    raise ValueError(f"History {field.name} must be finite float64 with shape {shape}")
                a = np.array(raw, dtype=np.float64, order="C", copy=True)
            self._arrays[field.name] = a

    @property
    def fields(self) -> tuple[StateFieldSpec, ...]:
        return self._fields

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self._fields)

    def __getitem__(self, name: str) -> np.ndarray:
        return self._arrays[name]

    def validate(self, n_particles: int | None = None) -> None:
        if n_particles is not None and self.n_particles != n_particles:
            raise ValueError("Material state particle count differs from solver")
        if set(self._arrays) != set(self.names):
            raise ValueError("Material state field inventory was modified")
        for field in self._fields:
            arr = self._arrays[field.name]
            expected = (self.n_particles, *field.component_shape)
            if (not isinstance(arr, np.ndarray) or arr.dtype != np.dtype('float64')
                    or arr.shape != expected or not arr.flags.c_contiguous
                    or not np.all(np.isfinite(arr))):
                raise ValueError(f"Invalid material state field {field.name}: required finite C-contiguous float64 {expected}")

    def snapshot(self) -> dict[str, np.ndarray]:
        self.validate()
        return {name: a.copy() for name, a in self._arrays.items()}

    def restore(self, snapshot: dict[str, np.ndarray]) -> None:
        if not isinstance(snapshot, dict) or set(snapshot) != set(self.names):
            raise ValueError("Invalid material state rollback snapshot")
        # Validate whole snapshot first, then copy without replacing arrays.
        for field in self._fields:
            a = np.asarray(snapshot[field.name])
            if (a.dtype != np.dtype('float64')
                    or a.shape != (self.n_particles, *field.component_shape)
                    or not np.all(np.isfinite(a))):
                raise ValueError(f"Invalid rollback values for {field.name}")
        for name, a in snapshot.items():
            self._arrays[name][:] = a
        self.validate()

    def metadata(self) -> list[dict]:
        """Stable field schema for manifest encoding, not constitutive values."""
        return [{"name": f.name, "component_shape": list(f.component_shape),
                 "dtype": "float64"} for f in self._fields]

    @classmethod
    def from_metadata(cls, n_particles: int, metadata: object) -> "MaterialState":
        if not isinstance(metadata, list) or not metadata:
            raise ValueError("Stage 13A history metadata must be a nonempty list")
        fields: list[StateFieldSpec] = []
        for entry in metadata:
            if (not isinstance(entry, dict) or set(entry) != {"name", "component_shape", "dtype"}
                    or entry["dtype"] != "float64" or not isinstance(entry["component_shape"], list)):
                raise ValueError("Invalid material history metadata")
            fields.append(StateFieldSpec(entry["name"], tuple(entry["component_shape"])))
        return cls(n_particles, fields)


__all__ = ["StateFieldSpec", "MaterialState"]
