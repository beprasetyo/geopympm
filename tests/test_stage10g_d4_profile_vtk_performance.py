"""D4 numerical-format regression and repeatable profiling smoke tests.

Timings are deliberately NOT asserted: they depend on CPU, system load, Numba
caches, disk state, and chosen particle count.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from geopympm.core import ParticleSet
from geopympm.io import write_particles_vtu
from geopympm.io.vtk_writer import _append_array
from validation.stage10g_d4.profile import run_profile


@pytest.mark.parametrize('size', [1, 255, 256, 512])
@pytest.mark.parametrize('width', [1, 2, 3, 6, 9])
def test_ascii_float_serializer_preserves_legacy_precision_and_layout(size, width):
    rng = np.random.default_rng(123)
    values = rng.standard_normal((size, width))
    # Include floats sensitive to rounding, negative zero and extreme exponents.
    if size >= 1:
        values[0, 0] = np.nextafter(0.1, 1.0)
    if size >= 2:
        values[1, 0] = -0.0
    if size >= 3:
        values[2, 0] = np.finfo(np.float64).tiny
    if size >= 4:
        values[3, 0] = np.finfo(np.float64).max
    root = ET.Element('PointData')
    _append_array(root, 'test', values)
    expected = '\n' + '\n'.join(
        ' '.join(format(float(value), '.17g') for value in row)
        for row in values) + '\n'
    data = root.find('DataArray')
    assert data is not None
    assert data.text == expected
    assert data.attrib['type'] == 'Float64'
    parsed = np.fromstring(data.text, sep=' ').reshape(values.shape)
    np.testing.assert_array_equal(parsed, values)
    np.testing.assert_array_equal(np.signbit(parsed), np.signbit(values))


def test_large_particles_vtu_field_roundtrip(tmp_path):
    n = 512
    rng = np.random.default_rng(34)
    positions = np.column_stack((np.linspace(0.1, 0.9, n), np.linspace(0.3, 0.7, n)))
    p = ParticleSet(positions, np.ones(n), np.full(n, 0.1),
                    velocity=rng.standard_normal((n, 2)),
                    stress=rng.standard_normal((n, 6)))
    file = write_particles_vtu(p, tmp_path / 'particle.vtu', time=.1, step=10,
                               fields=('mass', 'velocity', 'stress_voigt'))
    root = ET.parse(file)
    fields = {x.attrib['Name']: x for x in root.findall('.//PointData/DataArray')}
    assert set(fields) == {'mass', 'velocity', 'stress_voigt'}
    np.testing.assert_array_equal(np.fromstring(fields['mass'].text, sep=' '), p.mass)
    np.testing.assert_array_equal(np.fromstring(fields['stress_voigt'].text, sep=' ').reshape(n, 6), p.stress)
    velocity3 = np.fromstring(fields['velocity'].text, sep=' ').reshape(n, 3)
    np.testing.assert_array_equal(velocity3[:, :2], p.velocity)
    np.testing.assert_array_equal(velocity3[:, 2], 0.0)


def test_profiler_writes_unique_reproducible_outputs(tmp_path):
    target = tmp_path / 'stage10gd4'
    results = run_profile(methods=('mpm',), sizes=(2,), steps=1, repeats=1,
                          profile_steps=1, output_dir=target)
    row = results['measurements'][0]
    assert row['particles'] == 16
    assert row['warm_step_median_sec'] > 0
    assert row['vtu_hashed_files'] == 6  # particle/grid streams: 3 frames each
    assert row['vtu_hashed_bytes'] > 0
    assert row['checkpoint_with_vtu_sec'] > 0
    assert (target / 'measurements.csv').is_file()
    assert (target / 'profile_summary.json').is_file()
    assert (target / 'mpm' / 'cells_2' / 'solver_profile.txt').is_file()
    with pytest.raises(FileExistsError):
        run_profile(methods=('mpm',), sizes=(2,), steps=1, repeats=1,
                    output_dir=target)
