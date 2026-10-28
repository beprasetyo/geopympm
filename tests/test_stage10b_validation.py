"""End-to-end Stage 10B reproducible verification without plotting dependency."""
import json
from validation.stage10b.validate_cpgimp import run


def test_stage10b_benchmark_artifacts(tmp_path):
    summary=run(tmp_path,steps=40,ppc_x=2,plot=False)
    assert all(summary['checks'].values())
    for name in ('10B_summary.json','10B_affine_history.csv',
                 '10B_force_driven_history.csv'):
        assert (tmp_path/name).is_file()
    data=json.loads((tmp_path/'10B_summary.json').read_text())
    assert len(data['wave']) == 6
    assert data['affine_domain_max_abs_error_m'] < 5.e-12
