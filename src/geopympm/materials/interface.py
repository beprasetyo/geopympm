"""Stage 13A constitutive-dispatch contract for the existing material families.

The adapter keeps numerical material kernels outside Python per-particle loops.
Future models will provide compatible model-level dispatch and history schema.
This stage does NOT add an elastoplastic stress update or mixed-family support.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .material_state import StateFieldSpec, MaterialState
from .linear_elastic import update_linear_elastic_stress
from .objective_elastic import update_objective_linear_elastic_stress
from .neo_hookean import update_neo_hookean_stress
from .mohr_coulomb import MC_HISTORY, update_mohr_coulomb_stress, _particle_mc_parameters

@dataclass(frozen=True, slots=True)
class ConstitutiveAdapter:
    """Model-family operations consumed by explicit time-integration schemes.

    ``stress_update`` receives particle data, kinematics, the material mapping,
    dt, mode and explicit material history. Implementations must update
    particle stress in place or raise; the solver owns transactional rollback.
    ``wave_moduli`` returns per-particle (lambda, shear) for CFL screening.
    """
    family: str
    modes: tuple[str, ...]
    history_schema: tuple[StateFieldSpec, ...]
    stress_update: Callable
    wave_moduli: Callable

    def validate_state(self, state: MaterialState) -> None:
        state.validate()
        required = {f.name: f.component_shape for f in self.history_schema}
        actual = {f.name: f.component_shape for f in state.fields}
        for name, shape in required.items():
            if actual.get(name) != shape:
                raise ValueError(f"Missing or incompatible constitutive history {name}")


def _linear_stress(particles, kinematics, materials, dt, mode, state):
    if mode == "corotational":
        update_objective_linear_elastic_stress(
            particles, kinematics.strain_increment,
            kinematics.velocity_gradient, materials, dt)
    elif mode == "incremental":
        update_linear_elastic_stress(particles, kinematics.strain_increment, materials)
    else:
        raise ValueError("LinearElastic supports incremental or corotational stress update")


def _cnh_stress(particles, kinematics, materials, dt, mode, state):
    if mode != "hyperelastic":
        raise ValueError("CompressibleNeoHookean requires hyperelastic stress update")
    update_neo_hookean_stress(particles, materials)


def _mc_stress(particles, kinematics, materials, dt, mode, state):
    if mode != "incremental":
        raise ValueError("Stage 13B MohrCoulomb supports only incremental stress update")
    update_mohr_coulomb_stress(particles, kinematics.strain_increment, materials, state)


def _mc_wave_moduli(particles, materials):
    lam, mu, *_ = _particle_mc_parameters(particles, materials)
    return lam, mu


def resolve_constitutive_adapter(particles, materials) -> ConstitutiveAdapter:
    """Validate materials and return a whole-array constitutive dispatcher."""
    from .registry import material_family
    from .linear_elastic import _particle_moduli
    from .neo_hookean import _particle_cnh_moduli
    family = material_family(particles, materials)
    if family == "mc":
        return ConstitutiveAdapter(
            family="mc", modes=("incremental",),
            history_schema=MC_HISTORY, stress_update=_mc_stress,
            wave_moduli=_mc_wave_moduli,
        )
    if family == "linear":
        return ConstitutiveAdapter(
            family="linear", modes=("incremental", "corotational"),
            history_schema=(), stress_update=_linear_stress,
            wave_moduli=_particle_moduli,
        )
    if family == "cnh":
        def moduli(p, models):
            K, G = _particle_cnh_moduli(p, models)
            return K - (2.0 / 3.0) * G, G
        return ConstitutiveAdapter(
            family="cnh", modes=("hyperelastic",),
            history_schema=(), stress_update=_cnh_stress,
            wave_moduli=moduli,
        )
    raise TypeError("Unsupported material model family")


__all__ = ["ConstitutiveAdapter", "resolve_constitutive_adapter"]
