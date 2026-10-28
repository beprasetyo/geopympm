"""Stage 10A numerical validation regressions and non-overclaim checks."""

import numpy as np
import pytest

from validation.stage10.validate_gimp import (
    single_particle_probe,
    moving_prestressed_block,
    prescribed_strain_crossing,
    force_driven_elastic_wave,
)


def test_single_particle_exact_domain_average_and_gradient_jump():
    rows, stats = single_particle_probe()
    assert len(rows) == 201
    assert stats['max_gimp_exact_domain_error_n'] < 1e-10
    assert stats['max_adjacent_mpm_force_change_n'] > 19.9
    assert stats['max_adjacent_gimp_force_change_n'] < 0.11
    assert stats['rms_gimp_vs_domain_force_error_n'] < 1e-10
    assert stats['rms_mpm_vs_domain_force_error_n'] > 1.0


@pytest.mark.parametrize('ppc', (2,4))
def test_tiling_domain_gimp_exact_moving_block(ppc):
    _, point = moving_prestressed_block('mpm', ppc=ppc, steps=110)
    _, gimp = moving_prestressed_block('gimp', ppc=ppc, steps=110)
    assert point['rms_force_error_n'] > 1.0
    assert gimp['rms_force_error_n'] < 1e-8
    assert gimp['max_force_increment_noise_n'] < 1e-8
    for result in (point, gimp):
        assert result['max_stress_drift_pa'] < 1e-7
        assert result['max_position_error_m'] < 1e-9
        assert result['max_mass_relative_error'] < 1e-12


def test_strain_driven_crossing_computes_nonzero_stress_but_does_not_assume_gimp_is_better():
    for method in ('mpm','gimp'):
        records, info = prescribed_strain_crossing(method, steps=160)
        assert info['particle_crossed_x2']
        assert abs(records[-1]['sigma_xx_pa']) > 10
        assert np.isfinite(info['rms_stress_error_pa'])
        assert info['rms_stress_error_pa'] < 1500


def test_force_driven_wave_both_modes_reproduce_continuum_stress():
    point, pstats = force_driven_elastic_wave('mpm', steps=150)
    gimp, gstats = force_driven_elastic_wave('gimp', steps=150)
    for stats in (pstats,gstats):
        assert stats['max_stress_relative_l2'] < 0.08
        assert stats['max_mass_relative_error'] < 1e-12
    # In this wave the small particle domains remain inside single cells.
    # Hence uGIMP matches point MPM by construction: equality is EXPECTED.
    np.testing.assert_allclose([r['modal_sigma_pa'] for r in point],
                               [r['modal_sigma_pa'] for r in gimp], atol=1e-8)
