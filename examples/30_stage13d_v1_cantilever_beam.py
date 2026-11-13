"""Stage 13D_V1 self-weight cantilever; run with --help for choices.

Quick: python examples/30_stage13d_v1_cantilever_beam.py --steps 200
Full (10% g):  python examples/30_stage13d_v1_cantilever_beam.py --steps 30000 --plot --vtu-every 3000
Published load (unvalidated): use --gravity-scale 1.0 with a separate output folder
Each run must have its own output directory (refuses overwriting).
"""
from pathlib import Path
import sys

# examples/ is not a Python package; make the repository root importable when
# this file is invoked directly as a script from the repository directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validation.stage13d_v1.cantilever import main

if __name__ == '__main__':
    main()
