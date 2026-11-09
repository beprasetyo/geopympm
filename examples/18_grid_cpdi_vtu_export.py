"""Stage 10G-A2: static grid + manufactured affine CPDI deformation snapshots.

These are PRESCRIBED affine geometries for visualization, NOT force-driven MPM
solutions. Open grid_000.vtu and domains_000/001/002.vtu in VisIt/ParaView.
Automatic multiframe .visit/.pvd time series is reserved for Stage 10G-A3.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from geopympm.core import BackgroundGrid, ParticleSet
from geopympm.io import write_grid_vtu, write_cpdi_domains_vtu
from geopympm.interpolation.cpdi import update_cpdi_domain


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output-dir',type=Path,default=Path('results/stage10g_a2'))
    args=cli.parse_args()
    grid = BackgroundGrid(origin=(0.,0.),cell_size=(.25,.25),n_cells=(32,32))
    x,y = np.meshgrid(np.linspace(2.4,4.6,10), np.linspace(2.4,4.6,10), indexing='xy')
    p0 = np.column_stack((x.ravel(),y.ravel()))
    p = ParticleSet(p0, np.ones(len(p0)), np.full(len(p0), .22**2),
                    domain_size=np.full((len(p0),2),.22),
                    material_id=np.where(p0[:,0]>3.5,1,0).astype(np.int32))
    grid.velocity_bc_mask[:grid.n_cells[0]+1,1]=True
    grid.velocity_bc_mask[-(grid.n_cells[0]+1):,1]=True
    gp=write_grid_vtu(grid,args.output_dir/'grid_000.vtu',time=0.,step=0,overwrite=True)
    print(f'Wrote {gp} ({grid.n_elements} QUAD cells, {grid.n_nodes} nodes)')
    center=np.array([3.5,3.5])
    states=[(0., np.eye(2)),
            (.5, np.array([[1.,.4],[0.,1.]])),
            (1., np.array([[np.cos(.4),-np.sin(.4)],[np.sin(.4),np.cos(.4)]])
                 @ np.array([[1.1,.6],[0.,.9]]))]
    for i,(time,F) in enumerate(states):
        p.deformation_gradient[:]=F
        p.position[:]=center+(p0-center)@F.T
        update_cpdi_domain(p,grid)
        path=write_cpdi_domains_vtu(p,grid,args.output_dir/f'domains_{i:03d}.vtu',
                                    time=time,step=i,overwrite=True)
        print(f'Wrote {path} ({p.n_particles} CPDI quadrilaterals, detF={np.linalg.det(F):.6g})')
    print('Each file can be opened separately in VisIt. Time collections are Stage 10G-A3.')


if __name__=='__main__':
    main()
