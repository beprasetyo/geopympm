"""Stage 10G-A2: grid and CPDI domain VTK mesh writer tests."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.io import (
    CPDI_CELL_FIELDS, CPDI_POINT_FIELDS, GRID_CELL_FIELDS, GRID_POINT_FIELDS,
    write_cpdi_domains_vtu, write_grid_vtu,
)
from geopympm.interpolation.cpdi import candidate_cpdi_geometry


def _read(path):
    root = ET.parse(path).getroot()
    assert root.attrib['type'] == 'UnstructuredGrid'
    piece = root.find('./UnstructuredGrid/Piece')
    assert piece is not None
    return root, piece


def _array(piece, where, name=None):
    path = f'{where}/DataArray'
    if name is not None:
        path += f"[@Name='{name}']"
    item = piece.find(path)
    assert item is not None, (where, name)
    ncomp = int(item.attrib.get('NumberOfComponents', 1))
    return np.fromstring(item.text, sep=' ').reshape(-1, ncomp)


def _fixture():
    g = BackgroundGrid((0.0, 0.0), (.5, .75), (5, 4))
    g.mass[:] = np.arange(g.n_nodes) + .25
    g.velocity[:, 0] = np.arange(g.n_nodes) * .1
    g.acceleration[:, 1] = -9.81
    g.velocity_bc_mask[0, 0] = True
    g.velocity_bc_mask[-1, 1] = True
    g.velocity_bc_value[-1, 1] = 3.
    p = ParticleSet(
        position=[[1., 1.], [1.7, 1.7]], mass=[2., 3.], volume=[.1, .2],
        material_id=[0, 2], domain_size=[[.3, .45], [.3, .5]],
    )
    # Complete shear+stretch rotation test, both domain corners remain in grid.
    p.deformation_gradient[:] = [
        [[1.1, .3], [.2, .95]],
        [[.9, -.4], [.5, 1.1]],
    ]
    p.stress[:, 0] = -125.0
    p.stress[:, 3] = 30.0
    return p, g


def test_grid_vtu_connectivity_and_nodal_fields(tmp_path):
    _, g = _fixture()
    snap = {name: getattr(g, name).copy() for name in
            ('node_coordinates','mass','momentum','velocity','velocity_bc_mask',
             'velocity_bc_value','force_internal','force_external')}
    root, piece = _read(write_grid_vtu(g, tmp_path/'grid.vtu', time=.25, step=12))
    assert int(piece.attrib['NumberOfPoints']) == g.n_nodes
    assert int(piece.attrib['NumberOfCells']) == g.n_elements
    assert set(a.attrib['Name'] for a in piece.findall('./PointData/DataArray')) == set(GRID_POINT_FIELDS)
    assert set(a.attrib['Name'] for a in piece.findall('./CellData/DataArray')) == set(GRID_CELL_FIELDS)
    np.testing.assert_allclose(_array(piece,'Points'), np.column_stack((g.node_coordinates,np.zeros(g.n_nodes))))
    connectivity = _array(piece,'Cells','connectivity')[:,0].reshape(-1,4)
    for iy in range(g.n_cells[1]):
        for ix in range(g.n_cells[0]):
            np.testing.assert_array_equal(connectivity[iy*g.n_cells[0]+ix],g.cell_node_ids(ix,iy))
    np.testing.assert_array_equal(_array(piece,'Cells','types'),9*np.ones((g.n_elements,1)))
    np.testing.assert_array_equal(_array(piece,'Cells','offsets')[:,0],4*np.arange(1,g.n_elements+1))
    np.testing.assert_array_equal(_array(piece,'PointData','velocity_bc_mask')[0],[1,0,0])
    np.testing.assert_array_equal(_array(piece,'PointData','velocity_bc_mask')[-1],[0,1,0])
    np.testing.assert_allclose(_array(piece,'PointData','velocity')[:,0],g.velocity[:,0])
    np.testing.assert_allclose(_array(piece,'PointData','acceleration')[:,1],-9.81)
    np.testing.assert_allclose(np.fromstring(root.find('./UnstructuredGrid/FieldData/DataArray[@Name="TimeValue"]').text,sep=' '),[.25])
    for key, values in snap.items():
        np.testing.assert_array_equal(getattr(g,key),values)


def test_cpdi_exact_four_corners_and_cell_fields(tmp_path):
    p,g = _fixture()
    snap = {name: getattr(p,name).copy() for name in (
        'position','deformation_gradient','domain_size','domain_size0','stress')}
    corners,_,_,_ = candidate_cpdi_geometry(p,g)
    root,piece = _read(write_cpdi_domains_vtu(p,g,tmp_path/'domains.vtu',time=2.,step=200))
    assert int(piece.attrib['NumberOfCells']) == p.n_particles
    assert int(piece.attrib['NumberOfPoints']) == 4*p.n_particles
    np.testing.assert_allclose(_array(piece,'Points')[:,:2],corners.reshape(-1,2))
    np.testing.assert_array_equal(_array(piece,'Cells','connectivity')[:,0],np.arange(4*p.n_particles))
    np.testing.assert_array_equal(_array(piece,'Cells','types')[:,0],9)
    np.testing.assert_array_equal(_array(piece,'PointData','corner_id')[:,0],np.tile([0,1,2,3],p.n_particles))
    np.testing.assert_array_equal(_array(piece,'CellData','particle_id')[:,0],np.arange(p.n_particles))
    assert {a.attrib['Name'] for a in piece.findall('./CellData/DataArray')} == set(CPDI_CELL_FIELDS)
    assert {a.attrib['Name'] for a in piece.findall('./PointData/DataArray')} == set(CPDI_POINT_FIELDS)
    ratio = np.linalg.det(p.deformation_gradient)
    np.testing.assert_allclose(_array(piece,'CellData','area_ratio')[:,0],ratio,rtol=1e-14)
    np.testing.assert_allclose(_array(piece,'CellData','domain_area')[:,0],ratio*np.prod(p.domain_size0,axis=1),rtol=1e-14)
    np.testing.assert_allclose(_array(piece,'CellData','jacobian')[:,0],ratio,rtol=1e-14)
    np.testing.assert_allclose(_array(piece,'CellData','pressure')[:,0],125./3.)
    for key,values in snap.items():
        np.testing.assert_array_equal(getattr(p,key),values)


def test_cpdi_sheared_domain_is_not_axis_aligned_bounding_box(tmp_path):
    p,g = _fixture()
    corners, *_ = candidate_cpdi_geometry(p,g)
    piece = _read(write_cpdi_domains_vtu(p,g,tmp_path/'cpdi.vtu'))[1]
    observed = _array(piece,'Points')[:,:2].reshape(-1,4,2)
    np.testing.assert_array_equal(observed,corners)
    assert abs(observed[0,1,1] - observed[0,0,1]) > 1e-4
    assert abs(observed[0,3,0] - observed[0,0,0]) > 1e-4


def test_field_selection_and_overwrite(tmp_path):
    p,g = _fixture()
    output = tmp_path/'grid.vtu'
    write_grid_vtu(g,output,fields=['mass'],cell_fields=['cell_id'])
    piece = _read(output)[1]
    assert [e.attrib['Name'] for e in piece.findall('./PointData/DataArray')] == ['mass']
    assert [e.attrib['Name'] for e in piece.findall('./CellData/DataArray')] == ['cell_id']
    contents = output.read_bytes()
    with pytest.raises(FileExistsError):
        write_grid_vtu(g,output)
    assert output.read_bytes() == contents
    write_grid_vtu(g,output,overwrite=True,fields=[])
    assert _read(output)[1].find('./PointData/DataArray') is None
    output2 = tmp_path/'cpdi.vtu'
    write_cpdi_domains_vtu(p,g,output2,fields=['domain_area'],point_fields=['corner_id'])
    piece = _read(output2)[1]
    assert [e.attrib['Name'] for e in piece.findall('./CellData/DataArray')] == ['domain_area']
    assert [e.attrib['Name'] for e in piece.findall('./PointData/DataArray')] == ['corner_id']
    with pytest.raises(FileExistsError):
        write_cpdi_domains_vtu(p,g,output2)


@pytest.mark.parametrize('bad', [['mass','mass'], ['does_not_exist']])
def test_invalid_grid_field_selection(tmp_path,bad):
    _,g = _fixture()
    with pytest.raises(ValueError, match='fields'):
        write_grid_vtu(g,tmp_path/'bad.vtu',fields=bad)
    assert not (tmp_path/'bad.vtu').exists()


def test_invalid_cpdi_field_selection(tmp_path):
    p,g = _fixture()
    with pytest.raises(ValueError, match='fields'):
        write_cpdi_domains_vtu(p,g,tmp_path/'bad.vtu',fields=['x'])
    assert not (tmp_path/'bad.vtu').exists()


def test_invalid_inputs_fail_before_writing(tmp_path):
    p,g = _fixture()
    with pytest.raises(ValueError,match='.vtu'):
        write_grid_vtu(g,tmp_path/'grid.csv')
    with pytest.raises(ValueError,match='time'):
        write_grid_vtu(g,tmp_path/'bad.vtu',time=float('nan'))
    with pytest.raises(ValueError,match='step'):
        write_grid_vtu(g,tmp_path/'bad.vtu',step=-1)
    g.mass[0] = np.nan
    with pytest.raises(ValueError,match='mass'):
        write_grid_vtu(g,tmp_path/'bad.vtu')
    assert not (tmp_path/'bad.vtu').exists()
    # Inverted CPDI F must not get visualized as an apparently valid cell.
    p.deformation_gradient[0] = [[-1,0],[0,1]]
    with pytest.raises(ValueError,match='domain'):
        write_cpdi_domains_vtu(p,BackgroundGrid((0,0),(.5,.5),(5,5)),tmp_path/'cpdi.vtu')
    assert not (tmp_path/'cpdi.vtu').exists()


def test_outside_grid_and_missing_domains_are_rejected(tmp_path):
    p,g = _fixture()
    p.position[0] = [-1.,0.]
    with pytest.raises(ValueError,match='outside'):
        write_cpdi_domains_vtu(p,g,tmp_path/'bad.vtu')
    p,g = _fixture()
    p.domain_size0[0,0] = 0
    with pytest.raises(ValueError,match='domain_size0'):
        write_cpdi_domains_vtu(p,g,tmp_path/'bad.vtu')


def test_grid_renders_as_vtk_quads_using_real_reader(tmp_path):
    vtk = pytest.importorskip('vtk', reason='VTK Python reader optional')
    _,g = _fixture()
    path = write_grid_vtu(g,tmp_path/'grid.vtu',time=1.5,step=7)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path)); reader.Update()
    mesh = reader.GetOutput()
    assert mesh.GetNumberOfPoints() == g.n_nodes
    assert mesh.GetNumberOfCells() == g.n_elements
    assert mesh.GetCellType(0) == vtk.VTK_QUAD
    np.testing.assert_array_equal([mesh.GetCell(0).GetPointId(i) for i in range(4)],g.cell_node_ids(0,0))
    assert mesh.GetPointData().GetArray('velocity_bc_mask').GetNumberOfComponents() == 3
    assert mesh.GetFieldData().GetArray('TimeValue').GetValue(0) == pytest.approx(1.5)
    key = vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS()
    assert reader.GetOutputInformation(0).Get(key,0) == pytest.approx(1.5)


def test_cpdi_renders_as_vtk_quads_using_real_reader(tmp_path):
    vtk = pytest.importorskip('vtk', reason='VTK Python reader optional')
    p,g = _fixture()
    path = write_cpdi_domains_vtu(p,g,tmp_path/'domains.vtu')
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path)); reader.Update()
    mesh = reader.GetOutput()
    assert mesh.GetNumberOfPoints() == 4*p.n_particles
    assert mesh.GetNumberOfCells() == p.n_particles
    assert mesh.GetCellType(0) == vtk.VTK_QUAD
    assert mesh.GetCellData().GetArray('stress').GetNumberOfComponents() == 9
    np.testing.assert_allclose(mesh.GetCellData().GetArray('area_ratio').GetTuple1(0),
                               np.linalg.det(p.deformation_gradient[0]))
