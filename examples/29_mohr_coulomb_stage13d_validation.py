"""Stage 13D reproducible MC analytical & MPM patch validation.

Run at repository root:
    python examples/29_mohr_coulomb_stage13d_validation.py

To configure output or case selection directly:
    python -m validation.stage13d.run --help
"""
from pathlib import Path
import sys

# The validation/ tree is beside examples/ (not part of geopympm package).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation.stage13d.run import main

if __name__ == "__main__":
    raise SystemExit(main())
