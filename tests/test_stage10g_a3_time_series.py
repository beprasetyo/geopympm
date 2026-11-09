"""Stage 10G-A3: time indexes, schedule, committed-steps, failure safety."""
import xml.etree.ElementTree as ET
import numpy as np
import pytest

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.io import OutputManager, SeriesFrame, write_time_series
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver


def solver_fixture(*, cpdi=True, dt=.0005):
    pts = np.array([[.75, .75], [1.25, .75], [.75, 1.25], [1.25, 1.25]])
    p = ParticleSet(pts, np.ones(4), np.full(4, .04),
                    velocity=np.tile([.01, 0.], (4, 1)),
                    domain_size=np.full((4, 2), .20))
    g = BackgroundGrid((0., 0.), (.5, .5), (5, 5))
    return ExplicitMPMSolver(p, g, LinearElastic(1000., .25), dt=dt,
                             interpolation="cpdi" if cpdi else "mpm")


def _pvd(index):
    root = ET.parse(index).getroot()
    assert root.tag == 'VTKFile' and root.attrib['type'] == 'Collection'
    return root.findall('./Collection/DataSet')


def test_series_visit_pvd_names_and_exact_times(tmp_path):
    (tmp_path / 'particles_000000.vtu').write_text('test')
    (tmp_path / 'particles_000005.vtu').write_text('test')
    frames = [SeriesFrame(0, 0., 'particles_000000.vtu'),
              SeriesFrame(5, .014, 'particles_000005.vtu')]
    visit, pvd = write_time_series(frames, tmp_path, 'particles')
    assert visit.read_text() == ('!NBLOCKS 1\n!TIME 0.0\nparticles_000000.vtu\n'
                                 '!TIME 0.014\nparticles_000005.vtu\n')
    datasets = _pvd(pvd)
    assert [(float(a.attrib['timestep']), a.attrib['file']) for a in datasets] == [
        (0., 'particles_000000.vtu'), (.014, 'particles_000005.vtu')]
    assert all(a.attrib['part'] == '0' for a in datasets)


@pytest.mark.parametrize('bad', ['../outside.vtu', '/tmp/abs.vtu', 'x\\y.vtu',
                                 'broken.txt', '.hidden.vtu', 'x\ny.vtu'])
def test_unsafe_filename_rejected(bad):
    with pytest.raises(ValueError):
        SeriesFrame(1, 1.0, bad)


def test_series_requires_existing_files_and_monotonicity(tmp_path):
    name = 'frame_000001.vtu'
    with pytest.raises(FileNotFoundError):
        write_time_series([SeriesFrame(1, .1, name)], tmp_path, 'particles')
    (tmp_path / name).write_text('test')
    with pytest.raises(ValueError, match='strictly increasing'):
        write_time_series([SeriesFrame(1, .1, name), SeriesFrame(1, .2, name)],
                          tmp_path, 'particles')


def test_step_schedule_and_all_three_streams(tmp_path):
    solver = solver_fixture()
    output = OutputManager(tmp_path / 'vis', every_n_steps=2,
                           particles=True, grid=True, cpdi_domains=True)
    snapshot = {name: getattr(solver.particles, name).copy() for name in
                ('position', 'stress', 'velocity', 'deformation_gradient')}
    first = output.write_initial(solver)
    assert set(first) == {'particles', 'grid', 'domains'}
    assert all(p.exists() for p in first.values())
    for name, array in snapshot.items():
        np.testing.assert_array_equal(getattr(solver.particles, name), array)
    emitted = []
    for _ in range(5):
        diag = solver.step()
        if output.on_step_completed(solver, diag):
            emitted.append(diag.step)
    assert emitted == [2, 4]
    for kind in output.enabled:
        visit = output.directory / f'{kind}.visit'
        pvd = output.directory / f'{kind}.pvd'
        assert [float(a.attrib['timestep']) for a in _pvd(pvd)] == [0., .001, .002]
        assert [a.attrib['file'] for a in _pvd(pvd)] == [
            f'{kind}_{i:06d}.vtu' for i in (0, 2, 4)]
        lines = visit.read_text().splitlines()
        assert [line for line in lines if line.startswith('!TIME ')] == [
            '!TIME 0.0', '!TIME 0.001', '!TIME 0.002']
        assert len(output.frames[kind]) == 3


def test_time_schedule_uses_actual_simulation_time_and_no_duplicate(tmp_path):
    solver = solver_fixture(dt=.002)
    output = OutputManager(tmp_path, every_n_steps=None, every_dt=.003,
                           particles=True, grid=False)
    actual = []
    for _ in range(7):
        result = solver.step()
        if output.on_step_completed(solver, result):
            actual.append((result.step, result.time))
    assert [step for step, _ in actual] == [2, 3, 5, 6]
    assert [a.attrib['file'] for a in _pvd(tmp_path/'particles.pvd')] == [
        'particles_000002.vtu', 'particles_000003.vtu',
        'particles_000005.vtu', 'particles_000006.vtu']
    assert [float(a.attrib['timestep']) for a in _pvd(tmp_path/'particles.pvd')] == [
        .004, .006, .01, .012]


def test_or_schedule_deduplicates_same_step(tmp_path):
    solver = solver_fixture(dt=.002)
    output = OutputManager(tmp_path, every_n_steps=2, every_dt=.004,
                           grid=False)
    for _ in range(5):
        output.on_step_completed(solver, solver.step())
    assert [frame.step for frame in output.frames['particles']] == [2, 4]


def test_emission_refuses_duplicate_or_skipped_steps(tmp_path):
    solver = solver_fixture()
    output = OutputManager(tmp_path, every_n_steps=1)
    d = solver.step()
    output.on_step_completed(solver, d)
    with pytest.raises(ValueError, match='exactly once'):
        output.on_step_completed(solver, d)
    solver.step()
    d = solver.step()
    with pytest.raises(ValueError, match='exactly once'):
        output.on_step_completed(solver, d)


def test_initial_only_once_and_protect_prior_series(tmp_path):
    solver = solver_fixture()
    output = OutputManager(tmp_path)
    output.write_initial(solver)
    with pytest.raises(RuntimeError):
        output.write_initial(solver)
    with pytest.raises(FileExistsError, match='Existing time series'):
        OutputManager(tmp_path)


def test_no_output_for_failed_solver_step(tmp_path):
    solver = solver_fixture(dt=1.)
    output = OutputManager(tmp_path, every_n_steps=1)
    with pytest.raises(ValueError, match='exceeds'):
        solver.step()
    assert solver.step_count == 0
    assert not list(tmp_path.iterdir())
    assert output.frames['particles'] == ()


def test_cpdi_export_requires_cpdi_interpolation(tmp_path):
    solver = solver_fixture(cpdi=False)
    output = OutputManager(tmp_path, cpdi_domains=True)
    with pytest.raises(ValueError, match="interpolation='cpdi'"):
        output.write_initial(solver)
    assert not list(tmp_path.iterdir())


def test_failed_write_does_not_publish_partial_collection(tmp_path, monkeypatch):
    solver = solver_fixture()
    output = OutputManager(tmp_path, every_n_steps=1, cpdi_domains=True)
    output.write_initial(solver)
    old_indices = {p.name: p.read_bytes() for p in tmp_path.glob('*.visit')}
    import geopympm.io.output_manager as mod
    real = mod.write_cpdi_domains_vtu
    def broken(*args, **kwargs):
        path = args[2]
        path.write_text('partial file')
        raise OSError('simulated disk failure during CPDI snapshot')
    monkeypatch.setattr(mod, 'write_cpdi_domains_vtu', broken)
    d = solver.step()
    with pytest.raises(OSError, match='simulated disk failure'):
        output.on_step_completed(solver, d)
    assert set(p.name for p in tmp_path.glob('*.vtu')) == {
        'particles_000000.vtu', 'grid_000000.vtu', 'domains_000000.vtu'}
    assert {p.name: p.read_bytes() for p in tmp_path.glob('*.visit')} == old_indices
    monkeypatch.setattr(mod, 'write_cpdi_domains_vtu', real)
    output.on_step_completed(solver, d)
    assert (tmp_path / 'domains_000001.vtu').is_file()


def test_field_selection_only_requested_fields(tmp_path):
    solver = solver_fixture()
    output = OutputManager(tmp_path, particles=True, grid=True,
                           particle_fields=('velocity', 'pressure'),
                           grid_fields=('mass',), every_n_steps=1)
    output.write_initial(solver)
    root = ET.parse(tmp_path / 'particles_000000.vtu')
    assert {d.attrib['Name'] for d in root.findall('./UnstructuredGrid/Piece/PointData/DataArray')} == {
        'velocity', 'pressure'}
    root = ET.parse(tmp_path / 'grid_000000.vtu')
    assert {d.attrib['Name'] for d in root.findall('./UnstructuredGrid/Piece/PointData/DataArray')} == {'mass'}


def test_reader_optional_can_follow_series(tmp_path):
    vtk = pytest.importorskip('vtk', reason='Independent VTK Python reader optional')
    solver = solver_fixture()
    output = OutputManager(tmp_path, every_n_steps=1)
    output.write_initial(solver)
    output.on_step_completed(solver, solver.step())
    # Independent VTK reader validates the two actual VTU snapshots.
    reader = vtk.vtkXMLUnstructuredGridReader()
    for frame in output.frames['particles']:
        reader.SetFileName(str(tmp_path / frame.filename))
        reader.Update()
        assert reader.GetOutput().GetNumberOfCells() == solver.particles.n_particles
    assert [float(x.attrib['timestep']) for x in _pvd(tmp_path/'particles.pvd')] == [0., solver.time]


def test_small_times_do_not_trigger_early(tmp_path):
    # An absolute tolerance scaled to one second would spuriously emit here.
    solver = solver_fixture(dt=1e-17)
    output = OutputManager(tmp_path, every_n_steps=None, every_dt=2e-17)
    emitted = []
    for _ in range(4):
        diag = solver.step()
        if output.on_step_completed(solver, diag):
            emitted.append(diag.step)
    assert emitted == [2, 4]
