"""Stage 10A parameter-sensitivity regressions."""
from __future__ import annotations

import numpy as np
import pytest

from validation.stage10.force_driven_crossing import make_case, one_run
from validation.stage10.sensitivity_gimp import run


def test_domain_factor_and_pic_are_passed_into_solver():
    X,p,g,solver=make_case('gimp',drift=1,ppc_x=4,domain_factor=2.0,pic_fraction=0.1)
    np.testing.assert_allclose(p.domain_size[:,0],g.cell_size[0]/2)
    np.testing.assert_allclose(p.domain_size[:,1],g.cell_size[1]/2)
    assert solver.interpolation == 'gimp'
    assert solver.pic_fraction == 0.1
    assert len(X)==640


@pytest.mark.parametrize('bad_factor',(-1.0,0.0,5.0,float('nan')))
def test_invalid_gimp_domain_scale_rejected(bad_factor):
    with pytest.raises(ValueError,match='domain_factor'):
        make_case('gimp',ppc_x=4,domain_factor=bad_factor)


@pytest.mark.parametrize('bad_pic',(-0.1,1.1,float('nan')))
def test_invalid_pic_rejected(bad_pic):
    with pytest.raises(ValueError,match='pic_fraction'):
        make_case('gimp',ppc_x=4,pic_fraction=bad_pic)


def test_default_benchmark_regression_with_explicit_parameters():
    _,_,a,stress_a,_=one_run('gimp',ppc_x=2,drift=1,steps=10)
    _,_,b,stress_b,_=one_run('gimp',ppc_x=2,drift=1,steps=10,
                             domain_factor=1.0,pic_fraction=0.0)
    assert a['time_rms_stress_relative_l2']==b['time_rms_stress_relative_l2']
    np.testing.assert_array_equal(stress_a,stress_b)


def test_sensitivity_outputs_and_conservation(tmp_path):
    output=tmp_path/'sensitivity'
    result=run(output,steps=35,plot=False,verbose=False,
               ppcs=(2,),pics=(0.0,),domain_factors=(0.5,1.0,2.0))
    assert result['summary']['n_cases']==4
    assert result['summary']['n_dynamics_runs']==8
    assert all(result['summary']['checks'].values())
    assert (output/'10A_SENS_all_cases.csv').exists()
    assert (output/'10A_SENS_summary.json').exists()
    assert any(r['method']=='gimp' and r['domain_factor']==2.0 for r in result['records'])
