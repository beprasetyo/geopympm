"""Stage 12: compare USL-like, USF and MUSL for Sulsky elastic-disk surrogate.

Examples:
  python examples/25_compare_time_integration_schemes.py --steps 200 --no-plot
  python examples/25_compare_time_integration_schemes.py --steps 6000 --output-dir results/stage12_sulsky

Uses exactly the CNH disk seeding and energy diagnostics of Stage 11E. Time
integration is the ONLY changed numerical option across runs. The problem
is *not* a quantitative Uintah validation (2D, fixed dt, no artificial viscosity).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import runpy
import time

import numpy as np
from geopympm.solvers import ExplicitMPMSolver


def run(*, output_dir, steps=6000, dt=0.0005, methods=("usl","usf","musl"),
        interpolation="mpm", plot=True):
    module = runpy.run_path(str(Path(__file__).with_name("24_sulsky_colliding_disks_cnh.py")))
    source_solver = module["make_solver"]
    snapshot = module["snapshot"]
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing nonempty directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for method in methods:
        original = source_solver(dt=dt, interpolation=interpolation)
        s = ExplicitMPMSolver(
            original.particles, original.grid, original.materials, dt=original.dt,
            cfl=original.cfl, pic_fraction=original.pic_fraction,
            gravity=original.gravity, stress_update=original.stress_update,
            kinematics_update=original.kinematics_update,
            advection_update=original.advection_update,
            interpolation=original.interpolation, scheme=method,
        )
        rows = [snapshot(s)]
        started = time.perf_counter()
        for _ in range(steps):
            s.step()
            rows.append(snapshot(s))
        elapsed = time.perf_counter() - started
        with (out / f"{method}_diagnostics.csv").open("w", newline="", encoding="utf8") as fd:
            writer = csv.DictWriter(fd, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        initial = rows[0]["total_mechanical_energy_j"]
        terminal = rows[-1]["total_mechanical_energy_j"]
        drift = max(np.hypot(r["momentum_total_x"] - rows[0]["momentum_total_x"],
                             r["momentum_total_y"] - rows[0]["momentum_total_y"]) for r in rows)
        results[method] = {
            "steps":steps, "dt_s":dt, "end_time_s":s.time,
            "cpu_walltime_s":elapsed,
            "end_energy_error_percent":100*(terminal-initial)/initial,
            "max_total_momentum_drift_kg_m_s":float(drift),
            "minimum_J":float(min(row["min_J"] for row in rows)),
            "rebound_a":bool(any(row["momentum_a_x"]<0 for row in rows)),
            "rebound_b":bool(any(row["momentum_b_x"]>0 for row in rows)),
        }
        print(f"{method:4s}: time={s.time:.3f}s dE={results[method]['end_energy_error_percent']:+.4f}% "
              f"dp={drift:.3e}, runtime={elapsed:.2f}s")
        results[method]["_rows"] = rows

    if plot:
        import matplotlib.pyplot as plt
        for field,name,ylabel in (
            ("total_mechanical_energy_j","energy","Total mechanical energy (J)"),
            ("momentum_a_x","momentum_disk_a","Disk A momentum x (kg m/s)"),
            ("center_distance_m","center_distance","Disk center distance (m)")):
            fig,ax=plt.subplots(figsize=(8,4.5))
            for label, entry in results.items():
                data=entry["_rows"]
                ax.plot([row["time_s"] for row in data], [row[field] for row in data], label=label)
            ax.set(xlabel="Time (s)",ylabel=ylabel)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out/f"{name}.png", dpi=150)
            plt.close(fig)
    for entry in results.values():
        entry.pop("_rows")
    (out / "comparison.json").write_text(json.dumps({
        "case":"Stage11E Sulsky CNH disk surrogate", "interpolation":interpolation,
        "schemes":results,
        "caution":"Results do not establish agreement with Uintah or conservation of energy."
    },indent=2)+"\n",encoding="utf8")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("results/stage12_sulsky"))
    parser.add_argument("--steps",type=int,default=6000)
    parser.add_argument("--dt",type=float,default=0.0005)
    parser.add_argument("--schemes",nargs="+",choices=("usl","usf","musl"),default=("usl","usf","musl"))
    parser.add_argument("--interpolation",choices=("mpm","gimp","cpgimp","cpdi"),default="mpm")
    parser.add_argument("--no-plot",action="store_true")
    args=parser.parse_args()
    if args.steps<0 or not (0<args.dt):
        parser.error("steps must be nonnegative and dt positive")
    run(output_dir=args.output_dir,steps=args.steps,dt=args.dt,
        methods=args.schemes,interpolation=args.interpolation,plot=not args.no_plot)

if __name__=="__main__":
    main()
