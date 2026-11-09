"""Stage 10G-C example: 120-step CPDI dynamics with three restarts.

Run:
    python examples/22_extended_restart_verification.py
    python examples/22_extended_restart_verification.py --steps 160 \\
       --output-dir results/stage10g_c_long_run

A continuous reference and a three-checkpoint chain are compared at EVERY
step, including particle/grid state, diagnostics, and VisIt/ParaView indexes.
This is a restart verification, not a high-strain constitutive benchmark.
Open <output-dir>/cpdi/restart/domains.visit in VisIt to inspect deformed CPDI
quadrilaterals; particles.visit and grid.visit are beside it.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Running `python examples/22_...py` sets sys.path[0] to examples/, not root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validation.stage10g_c.extended_restart import run_chain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage10g_c_example"))
    args = parser.parse_args()
    result = run_chain("cpdi", args.output_dir / "cpdi", steps=args.steps, dynamic=True)
    print(f"PASS: {result['steps']} steps, {len(result['checkpoints'])} restarts; identical at each step")
    print(f"CPDI peak |Fxy| = {result['max_abs_Fxy']:.6g}, peak polar angle = {result['max_polar_angle_deg']:.3f} deg")
    for cp in result["checkpoints"]:
        print(f"Checkpoint step {cp['step']}: {cp['path']}")
    print(f"VisIt CPDI domains: {args.output_dir / 'cpdi/restart/domains.visit'}")
    print(f"VisIt particles: {args.output_dir / 'cpdi/restart/particles.visit'}")
    print(f"VisIt background grid: {args.output_dir / 'cpdi/restart/grid.visit'}")


if __name__ == "__main__":
    main()
