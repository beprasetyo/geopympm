"""Stage 10E-B force-driven constrained nonlinear shear/rotation wave.

Run from the repository root:
    python examples/15_force_driven_2d_shear.py
"""
# The validation/ directory is deliberately outside the installed src/ package.
# Make the project root importable when this example is executed as a script.
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validation.stage10eb.nonlinear_shear import run_one

if __name__=='__main__':
    print('Stage 10E-B: force-driven nonlinear constrained plane-strain shear (t=0.9 s)')
    print('Method   shear stress error    Fxy RMS error   polar-angle RMS error')
    for method in ('mpm','gimp','cpgimp','cpdi'):
        row,_,_=run_one(method)
        print(f"{method:7s} {100*row['shear_stress_relative_l2']:12.4f}%"
              f" {row['Fxy_absolute_rms']:17.5f}"
              f" {row['rotation_absolute_rms_rad']*180/3.141592653589793:17.4f} deg")
    print('Notes: v_y constrained; no v_x grid boundary conditions; central strip only.')
