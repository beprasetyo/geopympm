"""Stage 10G-A3: non-invasive simulation-step visualization scheduler.

Call after solver.step() has COMMITTED, never from within solver.step().
This module does not alter grid/particle arrays, integration or rollback.
There is deliberately no resume/appending existing series here: checkpoint
and restart coordination belong to Stage 10G-B.
"""
from __future__ import annotations

import math
from pathlib import Path
from collections.abc import Iterable

from geopympm.solvers.explicit import ExplicitMPMSolver, StepDiagnostics
from .vtk_writer import write_particles_vtu
from .grid_vtk_writer import write_grid_vtu
from .cpdi_vtk_writer import write_cpdi_domains_vtu
from .time_series import SeriesFrame, write_time_series, _atomic_write


class OutputManager:
    """Emit VTU snapshots and synchronize VisIt/ParaView animations.

    Parameters
    ----------
    directory : path
        Run's *new* visualization directory (e.g. results/run/visualization).
    every_n_steps : int or None
        Emit at multiples of this step count, 1-based (default 10).
    every_dt : float or None
        Emit at first committed timestep crossing each physical-time target,
        measured from t=0. Physical times in files remain the *actual* solver
        times, NOT the target time. Either schedule triggers output (OR).
    particles, grid, cpdi_domains : bool
        Enable independent time series (CPDI geometry requires solver 'cpdi').
    particle_fields, grid_fields, cpdi_fields : iterable[str] or None
        Field selection; None means writer's full default set.

    Notes
    -----
    - ``write_initial(solver)`` optionally saves the t=0 snapshot once.
    - Call ``on_step_completed(solver, diagnostics)`` after EVERY successful
      step. Duplicate, skipped, and failed steps are rejected.
    - Existing series are never resumed or overwritten. Stage 10G-B adds
      explicit restart semantics later.
    - Initial grid transient nodal quantities are zero; grid data after step
      reflect last assembly and are NOT particle-interpolated at x_(n+1).
    """

    def __init__(self, directory: str | Path, *, every_n_steps: int | None = 10,
                 every_dt: float | None = None, particles: bool = True,
                 grid: bool = True, cpdi_domains: bool = False,
                 particle_fields: Iterable[str] | None = None,
                 grid_fields: Iterable[str] | None = None,
                 cpdi_fields: Iterable[str] | None = None):
        if every_n_steps is not None and (isinstance(every_n_steps, bool)
           or not isinstance(every_n_steps, int) or every_n_steps <= 0):
            raise ValueError("every_n_steps must be a positive integer or None")
        if every_dt is not None:
            if isinstance(every_dt, bool) or not math.isfinite(float(every_dt)) or float(every_dt) <= 0:
                raise ValueError("every_dt must be positive finite or None")
        if every_n_steps is None and every_dt is None:
            raise ValueError("At least one output schedule must be provided")
        for label, val in (("particles", particles), ("grid", grid), ("cpdi_domains", cpdi_domains)):
            if not isinstance(val, bool):
                raise TypeError(f"{label} must be a bool")
        self.enabled = tuple(name for name, active in
                             (("particles", particles), ("grid", grid), ("domains", cpdi_domains))
                             if active)
        if not self.enabled:
            raise ValueError("At least one data stream must be enabled")
        self.directory = Path(directory)
        self.every_n_steps = every_n_steps
        self.every_dt = float(every_dt) if every_dt is not None else None
        self._next_time = self.every_dt
        self._fields = {"particles": None if particle_fields is None else tuple(particle_fields),
                        "grid": None if grid_fields is None else tuple(grid_fields),
                        "domains": None if cpdi_fields is None else tuple(cpdi_fields)}
        self._entries: dict[str, list[SeriesFrame]] = {kind: [] for kind in self.enabled}
        self._last_seen_step = 0
        self._last_seen_time = 0.0
        self._initial_written = False
        # Do not silently merge into previous output. Restart support is a
        # separate feature and will need to recreate scheduler state.
        for kind in self.enabled:
            for suffix in (".visit", ".pvd"):
                if (self.directory / f"{kind}{suffix}").exists():
                    raise FileExistsError("Existing time series found; use a new output directory")

    @property
    def frames(self) -> dict[str, tuple[SeriesFrame, ...]]:
        """Read-only snapshot of the registered time-series frame lists."""
        return {name: tuple(records) for name, records in self._entries.items()}

    def _check_solver(self, solver: ExplicitMPMSolver) -> None:
        if not isinstance(solver, ExplicitMPMSolver):
            raise TypeError("solver must be ExplicitMPMSolver")
        if "domains" in self.enabled and solver.interpolation != "cpdi":
            raise ValueError("CPDI domain export requires interpolation='cpdi'")

    def write_initial(self, solver: ExplicitMPMSolver) -> dict[str, Path]:
        """Optionally output initial t=0, step=0 exactly once."""
        self._check_solver(solver)
        if self._initial_written or self._last_seen_step or solver.step_count != 0 or solver.time != 0.0:
            raise RuntimeError("write_initial is only allowed once at solver's initial state")
        written = self._emit(solver)
        self._initial_written = True
        return written

    def on_step_completed(self, solver: ExplicitMPMSolver,
                          diagnostics: StepDiagnostics) -> dict[str, Path]:
        """Process one *committed* sequential step; return paths if emitted."""
        self._check_solver(solver)
        if not isinstance(diagnostics, StepDiagnostics):
            raise TypeError("diagnostics must be the returned StepDiagnostics")
        if (solver.step_count != self._last_seen_step + 1
                or diagnostics.step != solver.step_count):
            raise ValueError("OutputManager requires each committed step exactly once")
        if (not math.isfinite(solver.time) or diagnostics.time != solver.time
                or solver.time <= self._last_seen_time):
            raise ValueError("diagnostics must match committed, increasing solver time")

        due_step = (self.every_n_steps is not None
                    and solver.step_count % self.every_n_steps == 0)
        tol = 8 * math.ulp(max(abs(solver.time), abs(self._next_time or 0.0)))
        due_time = (self._next_time is not None and solver.time + tol >= self._next_time)
        written = self._emit(solver) if due_step or due_time else {}
        # Advance time targets only after the emission succeeds.
        if due_time and self.every_dt is not None:
            index = max(1, math.floor((solver.time + tol) / self.every_dt) + 1)
            self._next_time = index * self.every_dt
        self._last_seen_step = solver.step_count
        self._last_seen_time = solver.time
        return written

    def _emit(self, solver: ExplicitMPMSolver) -> dict[str, Path]:
        """Write all selected streams and only then publish their indexes."""
        step, time = solver.step_count, float(solver.time)
        paths = {kind: self.directory / f"{kind}_{step:06d}.vtu" for kind in self.enabled}
        for path in paths.values():
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite VTK snapshot: {path}")
        # Save previous index contents so any partial collection update can be
        # rolled back if disk I/O fails. Newly emitted VTUs are also removed.
        previous = {}
        for kind in self.enabled:
            for ext in (".visit", ".pvd"):
                index_path = self.directory / f"{kind}{ext}"
                previous[index_path] = index_path.read_bytes() if index_path.exists() else None
        try:
            for kind, path in paths.items():
                if kind == "particles":
                    write_particles_vtu(solver.particles, path, time=time, step=step,
                                        fields=self._fields[kind])
                elif kind == "grid":
                    write_grid_vtu(solver.grid, path, time=time, step=step,
                                   fields=self._fields[kind])
                else:
                    write_cpdi_domains_vtu(solver.particles, solver.grid, path,
                                           time=time, step=step, fields=self._fields[kind])
            new_frame = {kind: SeriesFrame(step=step, time=time, filename=path.name)
                         for kind, path in paths.items()}
            for kind in self.enabled:
                write_time_series([*self._entries[kind], new_frame[kind]],
                                  self.directory, kind)
        except Exception:
            for path in paths.values():
                path.unlink(missing_ok=True)
            for index_path, content in previous.items():
                if content is None:
                    index_path.unlink(missing_ok=True)
                else:
                    _atomic_write(index_path, content)
            raise
        for kind in self.enabled:
            self._entries[kind].append(new_frame[kind])
        return paths


__all__ = ["OutputManager"]
