"""Stage 10C lightweight end-to-end reproducibility checks."""
import json
from validation.stage10c.validate_cpdi import run


def test_stage10c_outputs_and_mass_momentum(tmp_path):
    summary=run(tmp_path,steps=40,ppc_x=2,plot=False)
    assert all(summary['checks'].values()), summary
    assert summary['affine_max_F_error']<1e-11
    for name in ('10C_affine_history.csv','10C_force_driven_history.csv',
                 '10C_summary.json'):
        assert (tmp_path/name).is_file()
    assert len(json.loads((tmp_path/'10C_summary.json').read_text())['wave_final'])==4
