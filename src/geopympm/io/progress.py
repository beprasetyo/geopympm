"""Small, opt-in terminal progress reporter for long-running MPM examples.

This module does not access or mutate solver state.  Call ``start()`` before a
simulation and ``update()`` only *after* a successfully committed step.

Wall-clock ETA is a rolling, last-interval estimate; it is informational, not
physical simulation time.  No third-party dependencies are required.
"""
from __future__ import annotations

import math
import sys
import time
from typing import Mapping


def _clock_text(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ProgressReporter:
    """Write flushed, line-based progress updates without altering the solver.

    ``every=0`` disables all progress output. The first report's ETA is shown as
    warming up because Numba's initial JIT compilation can heavily skew it.
    Subsequent ETAs use the most recent reporting interval, not the run average.
    """

    def __init__(self, *, total_steps: int, every: int = 500,
                 label: str = "GeoPyMPM", stream=None, clock=None):
        if type(total_steps) is not int or total_steps < 0:
            raise ValueError("total_steps must be a nonnegative integer")
        if type(every) is not int or every < 0:
            raise ValueError("every must be a nonnegative integer")
        self.total_steps = total_steps
        self.every = every
        self.label = str(label)
        self.stream = sys.stdout if stream is None else stream
        self._clock = time.perf_counter if clock is None else clock
        self._started_at = None
        self._last_at = None
        self._last_step = 0

    @property
    def enabled(self) -> bool:
        return self.every > 0

    def _emit(self, text: str) -> None:
        if self.enabled:
            print(f"[{self.label}] {text}", file=self.stream, flush=True)

    def start(self) -> None:
        if not self.enabled:
            return
        if self._started_at is not None:
            raise RuntimeError("progress reporter already started")
        self._started_at = self._last_at = self._clock()
        self._emit(f"START | target={self.total_steps:,} steps | progress-every={self.every:,} "
                   "| initial Numba compilation may take extra time")

    def should_report(self, step: int) -> bool:
        if not self.enabled:
            return False
        return step > 0 and (step % self.every == 0 or step == self.total_steps)

    def update(self, *, step: int, time_s: float,
               metrics: Mapping[str, float] | None = None) -> None:
        if not self.should_report(step):
            return
        if self._started_at is None:
            raise RuntimeError("start() must precede progress updates")
        if step <= self._last_step or step > self.total_steps:
            raise ValueError("reported steps must be increasing and <= total_steps")
        now = self._clock()
        elapsed = max(0.0, now - self._started_at)
        delta_steps = step - self._last_step
        delta_wall = now - self._last_at
        if self._last_step > 0 and delta_wall > 0:
            rate = delta_steps / delta_wall
            eta = _clock_text((self.total_steps - step) / rate)
            speed = f"{rate:,.1f} steps/s"
        else:
            eta = "warming up"
            speed = "warming up"
        percent = 100.0 if self.total_steps == 0 else 100 * step / self.total_steps
        extras = ""
        if metrics:
            for name, value in metrics.items():
                extras += f" | {name}={float(value):.5g}"
        self._emit(f"step={step:,}/{self.total_steps:,} ({percent:5.1f}%)"
                   f" | t={time_s:.5f} s | wall={_clock_text(elapsed)}"
                   f" | speed={speed} | ETA={eta}{extras}")
        self._last_step = step
        self._last_at = now

    def finish(self, *, step: int, time_s: float,
               status: str = "COMPLETED", detail: str | None = None) -> None:
        if not self.enabled:
            return
        if self._started_at is None:
            raise RuntimeError("start() must precede finish()")
        elapsed = max(0.0, self._clock() - self._started_at)
        msg = (f"{status} | committed={step:,}/{self.total_steps:,} steps"
               f" | t={time_s:.5f} s | wall={_clock_text(elapsed)}")
        if detail:
            msg += f" | {detail}"
        self._emit(msg)
