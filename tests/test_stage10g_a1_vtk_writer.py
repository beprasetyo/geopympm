"""Stage 10G-A1: structure, field conventions, and real VTK reader tests."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from geopympm.core import ParticleSet
from geopympm.io import DEFAULT_FIELDS, write_particles_vtu


def _particles_2d():
    p = ParticleSet(
        position=[[1.25, 2.5], [2.25, 3.5]],
        mass=[2.0, 4.0], volume=[0.5, 1.0],
        material_id=np.array([0, 17], dtype=np.int64),
        velocity=[[1.0, -2.0], [0.5, 3.0]],
        stress=[[100, 80, 40, 12, 9, 7], [2, 1, 3, -4, 0, 0]],
        strain=[[.1, .2, 0, .6, .8, 1.0], [0, 0, 0, .4, 0, 0]],
        domain_size=[[.5, .5], [.4, .8]],
    )
    p.position[0, 0] += .2
    p.deformation_gradient[0] = [[1.2, .3], [.2, .9]]
    return p


def _array(root, section, name):
    return root.find(f"./UnstructuredGrid/{section}/DataArray[@Name='{name}']") if section == "FieldData" else root.find(f"./UnstructuredGrid/Piece/{section}/DataArray[@Name='{name}']")


def _values(root, section, name, columns=1):
    entry = _array(root, section, name)
    assert entry is not None
    return np.fromstring(entry.text, sep=" ").reshape(-1, columns)


def test_vtu_structure_and_data_types(tmp_path):
    p = _particles_2d()
    path = write_particles_vtu(p, tmp_path / "particle cloud.vtu", time=.25, step=4)
    tree = ET.parse(path)
    root = tree.getroot()
    assert root.attrib['type'] == 'UnstructuredGrid'
    piece = root.find('./UnstructuredGrid/Piece')
    assert piece.attrib['NumberOfPoints'] == '2'
    assert piece.attrib['NumberOfCells'] == '2'
    point_data = piece.find('PointData')
    assert {elem.attrib['Name'] for elem in point_data} == set(DEFAULT_FIELDS)
    assert _array(root, 'PointData', 'material_id').attrib['type'] == 'Int32'
    assert _array(root, 'PointData', 'stress').attrib['NumberOfComponents'] == '9'
    np.testing.assert_array_equal(_values(root, 'Cells', 'types'), [[1], [1]])
    np.testing.assert_array_equal(_values(root, 'Cells', 'connectivity'), [[0], [1]])
    np.testing.assert_array_equal(_values(root, 'Cells', 'offsets'), [[1], [2]])
    np.testing.assert_array_equal(_values(root, 'FieldData', 'Step'), [[4]])
    np.testing.assert_allclose(_values(root, 'FieldData', 'TimeValue'), [[.25]])
    # Points and vector fields are padded to 3D without discarding the 2D data.
    pts = np.fromstring(piece.find('./Points/DataArray').text, sep=' ').reshape(-1,3)
    np.testing.assert_array_equal(pts, [[1.45, 2.5, 0], [2.25, 3.5, 0]])
    np.testing.assert_allclose(_values(root, 'PointData', 'displacement', 3)[0], [.2, 0, 0])
    np.testing.assert_allclose(_values(root, 'PointData', 'density'), [[4], [4]])
    # Row-major tensor: stress_xy=12, strain_xy=engineering_gamma_xy/2=0.3.
    sig = _values(root, 'PointData', 'stress', 9)[0].reshape(3,3)
    eps = _values(root, 'PointData', 'strain', 9)[0].reshape(3,3)
    np.testing.assert_array_equal(sig, [[100,12,7],[12,80,9],[7,9,40]])
    np.testing.assert_array_equal(eps, [[.1,.3,.5],[.3,.2,.4],[.5,.4,0]])
    F = _values(root, 'PointData', 'deformation_gradient', 9)[0].reshape(3,3)
    np.testing.assert_array_equal(F, [[1.2,.3,0],[.2,.9,0],[0,0,1]])
    np.testing.assert_allclose(_values(root, 'PointData', 'pressure')[0], [-220/3])
    np.testing.assert_allclose(_values(root, 'PointData', 'jacobian')[0], [1.02])


def test_3d_tensor_not_padded_or_reordered(tmp_path):
    p = ParticleSet([[1.,2.,3.]],[1.],[.5],stress=[[1,2,3,4,5,6]],strain=[[1,2,3,4,5,6]])
    p.deformation_gradient[0] = [[1,.1,.2],[.3,2,.4],[.5,.6,3]]
    root = ET.parse(write_particles_vtu(p, tmp_path/'a.vtu')).getroot()
    np.testing.assert_array_equal(_values(root,'PointData','stress',9)[0].reshape(3,3),
                                  [[1,4,6],[4,2,5],[6,5,3]])
    np.testing.assert_array_equal(_values(root,'PointData','strain',9)[0].reshape(3,3),
                                  [[1,2,3],[2,2,2.5],[3,2.5,3]])
    np.testing.assert_allclose(_values(root,'PointData','deformation_gradient',9),
                               [[1,.1,.2,.3,2,.4,.5,.6,3]])


def test_roundtrip_float64_precision_and_no_mutation(tmp_path):
    p = _particles_2d()
    p.stress[0, 3] = np.nextafter(0.1, 1.0)
    p.velocity[1,1] = np.nextafter(1.0, 2.0)
    snap = {field: getattr(p, field).copy() for field in (
        'position','position0','velocity','stress','strain','mass','volume',
        'deformation_gradient','domain_size')}
    root = ET.parse(write_particles_vtu(p,tmp_path/'b.vtu')).getroot()
    np.testing.assert_array_equal(_values(root,'PointData','stress_voigt',6),p.stress)
    np.testing.assert_array_equal(_values(root,'PointData','velocity',3)[:,:2],p.velocity)
    for field, arr in snap.items():
        np.testing.assert_array_equal(getattr(p,field),arr)


def test_field_selection_and_no_overwrite(tmp_path):
    p = _particles_2d()
    out = tmp_path/'c.vtu'
    write_particles_vtu(p,out,fields=('mass','stress'))
    assert set(e.attrib['Name'] for e in ET.parse(out).findall('.//PointData/DataArray')) == {'mass','stress'}
    old = out.read_bytes()
    with pytest.raises(FileExistsError):
        write_particles_vtu(p,out)
    assert out.read_bytes() == old
    write_particles_vtu(p,out,fields=('velocity',),overwrite=True)
    assert 'velocity' in out.read_text()


@pytest.mark.parametrize('bad', (('mass','mass'),('not_a_field',)))
def test_reject_unknown_or_duplicate_fields(tmp_path,bad):
    with pytest.raises(ValueError,match='fields'):
        write_particles_vtu(_particles_2d(),tmp_path/'bad.vtu',fields=bad)
    assert not (tmp_path/'bad.vtu').exists()


def test_reject_invalid_file_and_nonfinite_state(tmp_path):
    p = _particles_2d()
    with pytest.raises(ValueError,match='.vtu'):
        write_particles_vtu(p,tmp_path/'not_vtu.csv')
    with pytest.raises(ValueError,match='time'):
        write_particles_vtu(p,tmp_path/'a.vtu',time=float('nan'))
    with pytest.raises(ValueError,match='step'):
        write_particles_vtu(p,tmp_path/'a.vtu',step=-1)
    p.stress[0,0] = float('nan')
    with pytest.raises(ValueError,match='stress'):
        write_particles_vtu(p,tmp_path/'a.vtu')
    assert not (tmp_path/'a.vtu').exists()


def test_vtk_reader_independent_interoperability(tmp_path):
    vtk = pytest.importorskip('vtk',reason='independent VTK Python reader optional')
    p = _particles_2d()
    path = write_particles_vtu(p,tmp_path/'interop.vtu',time=.5,step=5)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    mesh = reader.GetOutput()
    assert mesh.GetNumberOfPoints() == p.n_particles
    assert mesh.GetNumberOfCells() == p.n_particles
    assert all(mesh.GetCellType(i) == vtk.VTK_VERTEX for i in range(p.n_particles))
    assert mesh.GetPointData().GetArray('stress').GetNumberOfComponents() == 9
    assert mesh.GetPointData().GetArray('material_id').GetTuple1(1) == 17
    time_key = vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS()
    assert reader.GetOutputInformation(0).Has(time_key)
    assert reader.GetOutputInformation(0).Get(time_key, 0) == pytest.approx(.5)
    assert mesh.GetFieldData().GetArray('TimeValue').GetValue(0) == pytest.approx(.5)
    np.testing.assert_allclose(mesh.GetPoint(0),[1.45,2.5,0],rtol=0,atol=1e-14)
    np.testing.assert_allclose(mesh.GetPointData().GetArray('strain').GetTuple(0)[1],.3)

