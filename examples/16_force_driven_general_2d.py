"""Stage 10E-B2: unconstrained two-component force-driven plane-strain wave.

Run from project root: python examples/16_force_driven_general_2d.py
The independent reference is exact only for LINEAR plane-strain elasticity.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validation.stage10eb2.general_2d_waves import METHODS, run_one

print('Stage 10E-B2: oblique mixed P/S wave, t = 0.30 s, h = 0.25 m')
print('Method    stress L2     F RMS       rotation max    max |fx|    max |fy|')
for method in METHODS:
    r,_,_=run_one(method,h=.25,steps=150,amplitude_scale=.25)
    print(f'{method:7s}   {100*r["stress_relative_l2"]:7.3f}%  '
          f'{r["F_absolute_rms"]:10.5f}   '
          f'{r["peak_polar_rotation_deg"]:7.3f} deg   '
          f'{r["max_nodal_force_x_N"]:8.3f}   {r["max_nodal_force_y_N"]:8.3f}')
print('Both nodal velocity components remain unconstrained during the run.')
print('The linear-wave reference is NOT an exact finite-strain Jaumann solution.')
