"""Stage 10G-A3 standalone output manager and series verification.

Usage: python -m validation.stage10g_a3.validate_time_series
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.io import OutputManager
from geopympm.materials import LinearElastic
from geopympm.solvers import ExplicitMPMSolver


def _make_solver():
    xy = np.array([[.75, .75], [1.25, .75], [.75, 1.25], [1.25, 1.25]])
    p = ParticleSet(xy, np.ones(len(xy)), .04*np.ones(len(xy)),
                    domain_size=np.full((len(xy),2), .2),
                    velocity=np.tile([.01,0],(len(xy),1)))
    g = BackgroundGrid((0,0),(.5,.5),(5,5))
    return ExplicitMPMSolver(p,g,LinearElastic(1000.,.25),dt=.001,
                             interpolation='cpdi',pic_fraction=0.,
                             stress_update='corotational',kinematics_update='exponential')


def run_validation(result_dir: Path = Path('validation/results/stage10g_a3')):
    checks = {}
    independent_reader = 'not_installed'
    with tempfile.TemporaryDirectory(prefix='geopympm_stage10g_a3_') as workspace:
        directory = Path(workspace) / 'visualization'
        solver = _make_solver()
        reference = _make_solver()
        manager = OutputManager(directory, every_n_steps=2, every_dt=.0015,
                                particles=True, grid=True, cpdi_domains=True)
        manager.write_initial(solver)
        steps = [0]
        for _ in range(6):
            diag = solver.step()
            written = manager.on_step_completed(solver, diag)
            reference.step()
            if written:
                steps.append(diag.step)
        checks['expected_schedule'] = steps == [0, 2, 3, 4, 5, 6]
        checks['solver_clock_unchanged_by_output'] = (solver.time == reference.time and
                                                         solver.step_count == reference.step_count)
        for name in ('position','velocity','strain','stress','volume','deformation_gradient','domain_size'):
            checks['same_solution_'+name] = np.array_equal(getattr(solver.particles,name),
                                                            getattr(reference.particles,name))
        for kind in manager.enabled:
            visit = directory/f'{kind}.visit'
            pvd = directory/f'{kind}.pvd'
            files = [f.filename for f in manager.frames[kind]]
            lines = visit.read_text().splitlines()
            listed = [line for line in lines if not line.startswith('!TIME ') and not line.startswith('!NBLOCKS ')]
            expected_times = [float(f.time) for f in manager.frames[kind]]
            visit_times = [float(line.removeprefix('!TIME ')) for line in lines if line.startswith('!TIME ')]
            rows = ET.parse(pvd).getroot().findall('./Collection/DataSet')
            pvd_times = [float(row.attrib['timestep']) for row in rows]
            checks[f'{kind}_frame_count'] = len(files) == len(steps)
            checks[f'{kind}_visit_order'] = (lines[0] == '!NBLOCKS 1' and
                                              listed == files and visit_times == expected_times)
            checks[f'{kind}_pvd_time'] = pvd_times == expected_times
            checks[f'{kind}_files_exist'] = all((directory/name).is_file() for name in files)
            expected_cells = {'particles':solver.particles.n_particles,
                              'grid':solver.grid.n_elements,
                              'domains':solver.particles.n_particles}[kind]
            checks[f'{kind}_correct_vtk_cells'] = all(
                int(ET.parse(directory/name).getroot().find('./UnstructuredGrid/Piece').attrib['NumberOfCells'])
                == expected_cells for name in files)
            checks[f'{kind}_metadata_times'] = all(
                float(ET.parse(directory/f.filename).getroot().find(
                    './UnstructuredGrid/FieldData/DataArray[@Name="TimeValue"]').text) == f.time
                for f in manager.frames[kind])
        try:
            import vtk
            independent_reader = vtk.vtkVersion.GetVTKVersion()
            reader = vtk.vtkXMLUnstructuredGridReader()
            for kind in manager.enabled:
                valid = True
                for f in manager.frames[kind]:
                    reader.SetFileName(str(directory/f.filename))
                    reader.Update()
                    valid = valid and reader.GetOutput().GetNumberOfCells() > 0
                checks[f'{kind}_vtk_reader'] = bool(valid)
        except ImportError:
            independent_reader = 'not_installed'

    results = {'stage':'10G-A3','reader':independent_reader,'checks':checks,
               'passed':sum(checks.values()),'total':len(checks),
               'all_checks_passed':all(checks.values()),
               'caveat':'VisIt/ParaView GUI and restart/checkpoint not tested.'}
    result_dir.mkdir(parents=True,exist_ok=True)
    (result_dir/'summary.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    return results


if __name__ == '__main__':
    report = run_validation()
    print(json.dumps(report, indent=2))
    if not report['all_checks_passed']:
        raise SystemExit(1)
