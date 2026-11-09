"""Build a compact, traceable Stage-10 history from archived benchmark JSONs.

Only metrics present in their original source JSON are extracted. Missing
optional historical reports are reported, never replaced by invented values.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

RESULTS = Path('validation/results')


def export_history(output='validation/results/stage10f/10F_historical_validation.csv',
                   results_root=RESULTS):
    root = Path(results_root)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []

    def load(relative):
        path = root / relative
        if not path.exists():
            missing.append(str(path))
            return None
        return json.loads(path.read_text(encoding='utf8'))

    def record(stage, case, method, metric, value, unit, source, note):
        rows.append(dict(stage=stage, case=case, method=method, metric=metric,
                         value=float(value), unit=unit, source_json=str(source),
                         applicability=note))

    rel = 'stage10a/10A_summary.json'
    s = load(rel)
    if s:
        for method, d in s['moving_block'].items():
            record('10A', 'moving_block', method, 'rms_nodal_force_error',
                   d['rms_force_error_n'], 'N', rel,
                   'prescribed uniform translation; tiling physical quadrature domains')

    rel = 'stage10a_sensitivity/10A_SENS_summary.json'
    s = load(rel)
    if s:
        for d in s['default_cases']:
            if d['ppc_x'] == 4 and d['pic_fraction'] == 0.0 and d['domain_factor'] == 1.0:
                record('10A', 'moving_frame_elastic_wave', d['method'],
                       'time_rms_relative_stress_error', d['drift_time_rms_stress_rel_l2'],
                       'fraction', rel, 'h=0.25 m, FLIP, 4 ppc-x')
                record('10A', 'moving_frame_elastic_wave', d['method'],
                       'final_galilean_difference', d['galilean_final_stress_rel_l2'],
                       'fraction', rel, 'h=0.25 m, drift 1 versus 0 m/s')

    rel = 'stage10d/10D_summary.json'
    s = load(rel)
    if s:
        for mode, d in s['affine_end'].items():
            record('10D', f'prescribed_affine_{mode}', 'cpdi',
                   'maximum_deformation_gradient_error', d['max_F_error'],
                   'dimensionless', rel, 'prescribed velocity; NOT force-driven')
            record('10D', f'prescribed_affine_{mode}', 'cpdi',
                   'maximum_stress_error', d['max_stress_error_pa'], 'Pa', rel,
                   'Jaumann specific analytical reference')

    rel = 'stage10e/10E_summary.json'
    s = load(rel)
    if s:
        for d in s['cases']:
            if d['cell_dx_m'] == .25:
                record('10E-A', 'nonlinear_axial_characteristics', d['method'],
                       'final_relative_stress_error', d['final_stress_rel_l2'],
                       'fraction', rel, 'exact pre-shock 1D finite-stretch reduction')
                record('10E-A', 'nonlinear_axial_characteristics', d['method'],
                       'max_relative_energy_drift', d['maximum_energy_relative_drift'],
                       'fraction', rel, 'axial-reduction energy only')

    rel = 'stage10eb/10EB_summary.json'
    s = load(rel)
    if s:
        for d in s['cases']:
            if d['grid_spacing_m'] == .25:
                record('10E-B', 'constrained_shear_characteristics', d['method'],
                       'final_relative_shear_stress_error', d['shear_stress_relative_l2'],
                       'fraction', rel, 'constrained vy=0; core window only')

    rel = 'stage10eb2/10EB2_summary.json'
    s = load(rel)
    if s:
        for d in s['cases']:
            if d['h_m'] == .25 and d['amplitude_scale'] == .25:
                record('10E-B2', 'mixed_PS_2D', d['method'],
                       'final_relative_stress_error', d['stress_relative_l2'],
                       'fraction', rel, 'small-amplitude linear continuum comparator')

    if rows:
        with out.open('w', newline='', encoding='utf8') as f:
            writer = csv.DictWriter(f, fieldnames=tuple(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    return dict(n_rows=len(rows), missing_source_json=missing, output=str(out))


if __name__ == '__main__':
    print(json.dumps(export_history(), indent=2))
