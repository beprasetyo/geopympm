"""Stage 10E: minimal reproducer for force-driven 55%-stretch longitudinal wave.

From repository root (with editable GeoPyMPM install):
    python examples/14_force_driven_nonlinear_wave.py

For full charts/data/refinement use:
    python -m validation.stage10e.nonlinear_wave --refinement
"""
from pathlib import Path
import sys

# The validation/ directory is a repository-level package, not part of the
# editable src/geopympm installation. Allow direct `python examples/...`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation.stage10e.nonlinear_wave import run_one

if __name__ == '__main__':
    print('Stage 10E: force-driven nonlinear elastic wave (t=0.45 s)')
    print('Method      Stress error    Stretch error   Energy drift')
    for method in ('mpm', 'gimp', 'cpgimp', 'cpdi'):
        result, _, _ = run_one(method)
        print(f'{method:<11} {100*result["final_stress_rel_l2"]:8.3f}%'
              f'       {100*result["final_stretch_rel_l2"]:7.3f}%'
              f'        {100*result["maximum_energy_relative_drift"]:7.4f}%')
