"""Independent XML and optionally VTK-reader validation of 2D grid and CPDI meshes.

Run: python -m validation.stage10g_a2.validate_mesh_vtu
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.io import write_grid_vtu, write_cpdi_domains_vtu


def data(piece, where, name=None):
    path = f'{where}/DataArray'
    if name:
        path += f"[@Name='{name}']"
    e = piece.find(path)
    if e is None:
        raise AssertionError(f'missing {where}/{name}')
    components=int(e.attrib.get('NumberOfComponents','1'))
    return np.fromstring(e.text,sep=' ').reshape(-1,components)


def validate(outdir: Path) -> dict:
    outdir.mkdir(parents=True,exist_ok=True)
    g=BackgroundGrid((0.,0.),(.5,.5),(4,4))
    g.mass[0]=3.125
    p=ParticleSet([[1.,1.],[1.5,1.5]],[1.,2.],[.1,.2],
                  domain_size=[[.4,.6],[.4,.5]])
    p.deformation_gradient[0]=[[1.2,.5],[.2,1.1]]
    p.deformation_gradient[1]=[[.95,-.3],[.3,1.]]
    p.stress[0]=[-12.,-8.,-3.,4.,0.,0.]
    gridpath=write_grid_vtu(g,outdir/'grid.vtu',time=.25,step=25,overwrite=True)
    domainpath=write_cpdi_domains_vtu(p,g,outdir/'domains.vtu',time=.25,step=25,overwrite=True)
    gp=ET.parse(gridpath).getroot().find('./UnstructuredGrid/Piece')
    cp=ET.parse(domainpath).getroot().find('./UnstructuredGrid/Piece')
    checks={}
    checks['grid_point_count']=int(gp.attrib['NumberOfPoints'])==g.n_nodes
    checks['grid_quad_count']=int(gp.attrib['NumberOfCells'])==g.n_elements
    checks['grid_cells_are_quad']=bool(np.all(data(gp,'Cells','types')==9))
    checks['grid_node_order']=bool(np.array_equal(data(gp,'Cells','connectivity')[:4,0],g.cell_node_ids(0,0)))
    checks['nodal_mass_roundtrip']=data(gp,'PointData','mass')[0,0]==3.125
    checks['domain_point_count']=int(cp.attrib['NumberOfPoints'])==4*p.n_particles
    checks['domain_quad_count']=int(cp.attrib['NumberOfCells'])==p.n_particles
    checks['domain_cells_are_quad']=bool(np.all(data(cp,'Cells','types')==9))
    # Reference corners computed independently from CPDI geometry function.
    hx,hy=.2,.3
    a=p.deformation_gradient[0]@np.array([hx,0.])
    b=p.deformation_gradient[0]@np.array([0.,hy])
    expected=np.array([p.position[0]-a-b,p.position[0]+a-b,
                       p.position[0]+a+b,p.position[0]-a+b])
    checks['corners_match_affine_geometry']=bool(np.allclose(data(cp,'Points')[:4,:2],expected,atol=2e-15,rtol=0))
    checks['domain_area_matches_detF']=bool(np.allclose(data(cp,'CellData','area_ratio')[:,0],
        np.linalg.det(p.deformation_gradient),atol=2e-15,rtol=0))
    checks['pressure_convention']=bool(np.isclose(data(cp,'CellData','pressure')[0,0],23./3.))
    try:
        import vtk
    except ImportError:
        reader='not_installed'
    else:
        reader='vtkXMLUnstructuredGridReader'
        for path, expected_cells in ((gridpath,g.n_elements),(domainpath,p.n_particles)):
            vtkreader=vtk.vtkXMLUnstructuredGridReader()
            vtkreader.SetFileName(str(path))
            vtkreader.Update()
            m=vtkreader.GetOutput()
            checks[f'{path.stem}_vtk_reader']=m.GetNumberOfCells()==expected_cells and all(
                m.GetCellType(i)==vtk.VTK_QUAD for i in range(expected_cells))
            checks[f'{path.stem}_time_field']=bool(np.isclose(m.GetFieldData().GetArray('TimeValue').GetValue(0),.25))
    checks = {name: bool(value) for name, value in checks.items()}
    report={
        'stage':'10G-A2', 'files':[str(gridpath),str(domainpath)],
        'reader':reader, 'checks':checks, 'all_checks_passed':all(checks.values()),
        'caveat':'VisIt GUI not exercised; individual frames only, time series scheduled for 10G-A3.',
    }
    (outdir/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,default=Path('validation/results/stage10g_a2'))
    args=ap.parse_args()
    result=validate(args.output_dir)
    print(json.dumps(result,indent=2))
    if not result['all_checks_passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
