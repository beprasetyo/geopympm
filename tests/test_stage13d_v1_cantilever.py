"""Stage 13D_V1 regression checks; not an independent static validation pass."""
import json
import math

import numpy as np
import pytest

from validation.stage13d_v1.cantilever import (
    BeamConfig, analytical_beam_deflection, analytical_deflection_at, build_solver, run_case,
    LENGTH_M, HEIGHT_M, DENSITY_KG_M3, THICKNESS_M,
)


def test_analytical_reference_independent():
    ref=analytical_beam_deflection()
    assert abs(ref['total_m']-(-0.6780672)) < 1e-10
    assert abs(ref['total_m']-(-0.67806)) < 1e-4
    assert ref['bending_m'] < 0 and ref['shear_m'] < 0


@pytest.mark.parametrize('scheme', ['usl','usf','musl'])
@pytest.mark.parametrize('interpolation', ['mpm','gimp','cpgimp','cpdi'])
def test_build_and_short_solver_step(scheme,interpolation):
    config=BeamConfig(cell_size=0.5,particles_per_axis=1,scheme=scheme,interpolation=interpolation)
    solver,monitor=build_solver(config)
    assert math.isclose(solver.particles.mass.sum(), DENSITY_KG_M3*LENGTH_M*HEIGHT_M*THICKNESS_M)
    assert monitor.sum() == int(round(HEIGHT_M/config.cell_size))*config.particles_per_axis
    assert np.any(solver.grid.velocity_bc_mask)
    assert np.all(solver.grid.velocity_bc_value[solver.grid.velocity_bc_mask] == 0.0)
    solver.gravity[1]=-9.81
    diagnostic=solver.step()
    assert diagnostic.step==1
    assert np.all(np.isfinite(solver.particles.position))
    assert np.all(np.isfinite(solver.particles.stress))


def test_smoke_run_has_labels_and_no_static_claim(tmp_path):
    target=tmp_path/'one'
    report=run_case(output_dir=target,config=BeamConfig(cell_size=0.5,particles_per_axis=2),
                    steps=12,sample_every=4,vtu_every=6)
    assert report['steps']==12
    assert not report['quasi_static_screen_passed']
    assert report['relative_difference_to_published_static'] is None
    assert (target/'diagnostics.csv').exists()
    assert (target/'final_particles.csv').exists()
    assert report['failure_reason'] is None
    assert (target/'summary.json').exists()
    assert (target/'visualization'/'particles.pvd').exists()
    assert (target/'visualization'/'particles_000012.vtu').exists()
    assert json.loads((target/'summary.json').read_text())['status'].startswith('dynamic')
    with pytest.raises(FileExistsError):
        run_case(output_dir=target, config=BeamConfig(), steps=0)


@pytest.mark.parametrize('broken', [0, -1, 0.6])
def test_reject_invalid_grid_size(broken):
    with pytest.raises(ValueError):
        build_solver(BeamConfig(cell_size=broken))


@pytest.mark.parametrize('material', ['linear', 'cnh'])
def test_material_variants_are_selectable(material):
    cfg=BeamConfig(material=material, particles_per_axis=1)
    solver, _ = build_solver(cfg)
    assert solver.stress_update == ('incremental' if material=='linear' else 'hyperelastic')
    solver.gravity[1]=-cfg.gravity
    solver.step()
    assert np.all(np.isfinite(solver.particles.stress))


def test_independent_beam_deflection_at_monitor():
    at_tip=analytical_beam_deflection(g=0.981)['total_m']
    at_probe=analytical_deflection_at(7.875, g=0.981)['total_m']
    assert math.isclose(at_tip, -0.06780672, abs_tol=1e-10)
    assert math.isclose(at_probe, -0.06641131695556642, abs_tol=1e-10)
    assert abs(at_probe) < abs(at_tip)
    with pytest.raises(ValueError):
        analytical_deflection_at(9.0)
