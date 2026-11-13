"""Nonintrusive terminal progress reporting (Stage 13D_V1)."""
import io
import json
import math

import numpy as np
import pytest

from geopympm.io import ProgressReporter
from validation.stage13d_v1.cantilever import BeamConfig, run_case


class FakeClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value


def test_progress_reporter_uses_completed_steps_and_recent_eta():
    buffer = io.StringIO()
    clock = FakeClock()
    reporter = ProgressReporter(total_steps=1000, every=200, label='Test',
                                stream=buffer, clock=clock)
    reporter.start()
    assert not reporter.should_report(199)
    assert reporter.should_report(200)
    clock.value += 8
    reporter.update(step=200, time_s=0.02, metrics={'uy': -0.02})
    clock.value += 2
    reporter.update(step=400, time_s=0.04)
    reporter.finish(step=400, time_s=0.04, status='STOPPED')
    out=buffer.getvalue()
    assert 'START' in out
    assert 'step=200/1,000 ( 20.0%)' in out
    assert 'ETA=warming up' in out
    assert 'step=400/1,000 ( 40.0%)' in out
    assert '100.0 steps/s' in out
    assert 'ETA=00:00:06' in out
    assert 'uy=-0.02' in out
    assert 'STOPPED' in out


def test_progress_can_be_disabled_and_requires_nonnegative_interval():
    buffer=io.StringIO()
    rep=ProgressReporter(total_steps=5, every=0, stream=buffer)
    rep.start()
    assert not rep.should_report(5)
    rep.update(step=5, time_s=1.0)
    rep.finish(step=5, time_s=1.0)
    assert buffer.getvalue()==''
    with pytest.raises(ValueError):
        ProgressReporter(total_steps=10, every=-1)
    with pytest.raises(ValueError):
        ProgressReporter(total_steps=-1, every=5)


def test_v1_outputs_real_time_progress_at_requested_steps(tmp_path, capsys):
    report=run_case(output_dir=tmp_path/'progress', config=BeamConfig(), steps=7,
                    sample_every=3, progress_every=2)
    stdout=capsys.readouterr().out
    assert report['steps']==7
    assert stdout.count('step=')==4
    for label in ('step=2/7', 'step=4/7', 'step=6/7', 'step=7/7', 'COMPLETED'):
        assert label in stdout
    assert 'tip_uy(m)=' in stdout
    assert 'vmax(m/s)=' in stdout
    assert 'Jmin=' in stdout
    assert json.loads((tmp_path/'progress'/'summary.json').read_text())['steps']==7


def test_reporting_is_read_only_no_change_to_numerical_results(tmp_path, capsys):
    args=dict(config=BeamConfig(interpolation='gimp'), steps=9, sample_every=3)
    with_progress=run_case(output_dir=tmp_path/'with', progress_every=2, **args)
    silent=run_case(output_dir=tmp_path/'without', progress_every=0, **args)
    assert with_progress['tip_uy_m']==silent['tip_uy_m']
    assert with_progress['physical_time_s']==silent['physical_time_s']
    assert (tmp_path/'with'/'final_particles.csv').read_bytes()==(tmp_path/'without'/'final_particles.csv').read_bytes()
    assert (tmp_path/'with'/'diagnostics.csv').read_bytes()==(tmp_path/'without'/'diagnostics.csv').read_bytes()
    assert 'COMPLETED' in capsys.readouterr().out


def test_failure_is_reported_without_reporting_failed_step(tmp_path, monkeypatch, capsys):
    from geopympm.solvers import ExplicitMPMSolver
    def fail_step(self):
        raise ValueError('synthetic CFL limit')
    monkeypatch.setattr(ExplicitMPMSolver, 'step', fail_step)
    s=run_case(output_dir=tmp_path/'failing', config=BeamConfig(), steps=5,
               sample_every=2, progress_every=2)
    out=capsys.readouterr().out
    assert s['failed_requested_step']==1
    assert s['steps']==0
    assert 'NUMERICAL FAILURE' in out
    assert 'committed=0/5' in out
    assert 'synthetic CFL limit' in out
    assert (tmp_path/'failing'/'summary.json').is_file()


def test_invalid_progress_interval_rejected_before_output(tmp_path):
    with pytest.raises(ValueError, match='progress_every'):
        run_case(output_dir=tmp_path/'bad', config=BeamConfig(), steps=0,
                 progress_every=-3)
    assert not (tmp_path/'bad').exists()
