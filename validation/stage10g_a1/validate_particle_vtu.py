"""Reproducible Stage 10G-A1 VTK export and structure/reader validation.

Run: python -m validation.stage10g_a1.validate_particle_vtu
VTK Python bindings are optional: the XML checks run without them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from geopympm.core import ParticleSet
from geopympm.io import DEFAULT_FIELDS, write_particles_vtu


def run(output_dir: str | Path) -> dict:
    folder = Path(output_dir)
    p = ParticleSet(
        position=[[1., 2.], [3., 4.]], mass=[2., 3.], volume=[.2, .3],
        material_id=[0, 1], velocity=[[.4, .5], [-.1, .8]],
        stress=[[100., 70., 20., 15., 0., 0.], [-50., -60., -10., 5., 0., 0.]],
        strain=[[.01, .02, 0., .04, 0., 0.], [0., 0., 0., -.05, 0., 0.]],
        domain_size=[[.5, .6], [.7, .4]],
    )
    p.position[0, 0] += .3
    p.deformation_gradient[0] = [[1., .2], [0., 1.]]
    path = write_particles_vtu(p, folder/'particles_000012.vtu',time=.012,step=12,overwrite=True)
    root = ET.parse(path).getroot()
    piece = root.find('./UnstructuredGrid/Piece')
    data_names = {a.get('Name') for a in piece.findall('./PointData/DataArray')}
    points = np.fromstring(piece.find('./Points/DataArray').text, sep=' ').reshape(-1,3)
    checks = {
        'xml_unstructured_grid': root.attrib.get('type') == 'UnstructuredGrid',
        'particle_vertices': piece.attrib.get('NumberOfCells') == '2' and piece.attrib.get('NumberOfPoints') == '2',
        'all_universal_fields': data_names == set(DEFAULT_FIELDS),
        'coordinates_correct': bool(np.allclose(points, [[1.3,2.,0.],[3.,4.,0.]])),
        'timestamp_present': root.find("./UnstructuredGrid/FieldData/DataArray[@Name='TimeValue']") is not None,
    }
    reader_status = 'not_installed'
    try:
        import vtk  # optional test that exercises the real VTK file reader
    except ImportError:
        pass
    else:
        r = vtk.vtkXMLUnstructuredGridReader()
        r.SetFileName(str(path))
        r.Update()
        mesh = r.GetOutput()
        time_key = vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS()
        info = r.GetOutputInformation(0)
        checks['vtk_reader_points_cells'] = mesh.GetNumberOfPoints() == 2 and mesh.GetNumberOfCells() == 2
        checks['vtk_reader_stress_tensor'] = (mesh.GetPointData().GetArray('stress') is not None and
                                              mesh.GetPointData().GetArray('stress').GetNumberOfComponents() == 9)
        checks['vtk_reader_time'] = bool(info.Has(time_key)) and abs(info.Get(time_key, 0) - .012) < 1e-14
        reader_status = 'tested'
    summary = {
        'stage': '10G-A1', 'file': str(path), 'particle_count': p.n_particles,
        'field_count': len(data_names), 'reader': reader_status,
        'checks': checks, 'all_checks_passed': all(checks.values()),
        'caveat': 'VisIt GUI not exercised; VTU checked against XML and optional independent VTK reader. Single-frame only.',
    }
    (folder/'validation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('validation/results/stage10g_a1'))
    args = parser.parse_args()
    report = run(args.output_dir)
    print(json.dumps(report, indent=2))
    if not report['all_checks_passed']:
        raise SystemExit('Stage 10G-A1 validation did not pass')


if __name__ == '__main__':
    main()
