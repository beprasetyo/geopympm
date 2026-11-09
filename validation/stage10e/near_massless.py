"""Stage 10E-A.1: non-intrusive near-massless nodal diagnostic.

Observe the *unmodified* Stage 10E solver. Interpolation is captured at x_n,
BEFORE solver.step(), because the grid arrays after a step refer to that x_n.
No mass cutoff is applied to P2G, grid momentum, G2P or constitutive kernels.

Example:
  python -m validation.stage10e.near_massless --methods cpgimp --sample-step 98
  python -m validation.stage10e.near_massless --methods all
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from geopympm.interpolation.mpm import evaluate_quad4
from geopympm.interpolation.gimp import evaluate_gimp
from geopympm.interpolation.cpdi import evaluate_cpdi
from validation.stage10e.nonlinear_wave import make_case, reference, scalar_rms, stretch_initial, MODULUS

DEFAULT_THRESHOLDS = (1e-8, 1e-12, 1e-16)
METHODS = ("mpm", "gimp", "cpgimp", "cpdi")


def evaluate_at_start(method, particles, grid):
    """Use the same interpolation selection as ExplicitMPMSolver.step()."""
    if method == "cpdi":
        return evaluate_cpdi(particles, grid)
    if method in ("gimp", "cpgimp"):
        return evaluate_gimp(particles, grid)
    return evaluate_quad4(particles, grid)


def analyze_step(mass, internal_force, acceleration, nodal_velocity,
                 node_ids, weights, gradients, *, dt, particle_mass,
                 thresholds=DEFAULT_THRESHOLDS, bc_mask=None):
    """Analyze a single completed grid step with its PRE-STEP interpolation.

    Reference nodal mass is TOTAL PARTICLE MASS / count(m_i > 0), not an
    absolute kilogram threshold. This is an *observational* threshold.
    The contributions from masked nodes are reported as hypothetical parts
    of the ACTUAL, unchanged FLIP / PIC and velocity-gradient sums.

    The maximum x-acceleration is reported separately from y DOFs; in the
    Stage-10E benchmark all grid y velocities are prescribed zero.
    """
    mass = np.asarray(mass, dtype=np.float64)
    internal_force = np.asarray(internal_force, dtype=np.float64)
    acceleration = np.asarray(acceleration, dtype=np.float64)
    nodal_velocity = np.asarray(nodal_velocity, dtype=np.float64)
    node_ids = np.asarray(node_ids)
    weights = np.asarray(weights, dtype=np.float64)
    gradients = np.asarray(gradients, dtype=np.float64)
    particle_mass = np.asarray(particle_mass, dtype=np.float64)
    if mass.ndim != 1 or not np.all(np.isfinite(mass)) or np.any(mass < 0):
        raise ValueError("Nodal masses must be finite and nonnegative.")
    n_nodes = len(mass)
    n_particles, support = weights.shape
    if (internal_force.shape != (n_nodes, 2) or acceleration.shape != (n_nodes, 2)
        or nodal_velocity.shape != (n_nodes, 2) or node_ids.shape != (n_particles, support)
        or gradients.shape != (n_particles, support, 2)
        or particle_mass.shape != (n_particles,)):
        raise ValueError("Inconsistent nodal/interpolation/particle array shapes.")
    if not np.issubdtype(node_ids.dtype, np.integer) or np.any(node_ids < 0) or np.any(node_ids >= n_nodes):
        raise ValueError("Invalid interpolation node indices.")
    if not (np.all(np.isfinite(internal_force)) and np.all(np.isfinite(acceleration))
        and np.all(np.isfinite(nodal_velocity)) and np.all(np.isfinite(weights))
        and np.all(np.isfinite(gradients)) and np.all(particle_mass > 0)):
        raise ValueError("All inputs must be finite and particle masses positive.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")
    if bc_mask is not None and np.asarray(bc_mask).shape != (n_nodes, 2):
        raise ValueError("BC mask shape must be (n_nodes,2).")
    if not np.isclose(mass.sum(), particle_mass.sum(), rtol=1e-10, atol=1e-12):
        raise ValueError("P2G mass must agree with particle mass.")
    active = mass > 0
    if not np.any(active):
        raise ValueError("No nodes received positive mass.")
    m_ref = float(particle_mass.sum() / int(np.count_nonzero(active)))
    a_x = np.abs(acceleration[:, 0])
    maximum_id = int(np.argmax(a_x))
    total_dv_x = dt * np.sum(weights * acceleration[node_ids, 0], axis=1)
    total_pic_vx = np.sum(weights * nodal_velocity[node_ids, 0], axis=1)
    total_Lxx = np.sum(nodal_velocity[node_ids, 0] * gradients[:, :, 0], axis=1)
    summary = {
        "mass_reference_kg": m_ref,
        "grid_mass_sum_kg": float(mass.sum()),
        "total_active_nodes": int(active.sum()),
        "raw_max_ax_m_s2": float(a_x[maximum_id]),
        "raw_max_node_id": maximum_id,
        "raw_max_node_mass_kg": float(mass[maximum_id]),
        "raw_max_node_internal_fx_N": float(internal_force[maximum_id, 0]),
        "raw_max_node_mass_ratio": float(mass[maximum_id] / m_ref),
        "unfiltered_max_flip_dvx_m_s": float(np.max(np.abs(total_dv_x))),
        "unfiltered_max_pic_vx_m_s": float(np.max(np.abs(total_pic_vx))),
        "unfiltered_max_Lxx_s_inv": float(np.max(np.abs(total_Lxx))),
        "thresholds": {},
    }
    for tau in thresholds:
        tau = float(tau)
        if not np.isfinite(tau) or not (0 < tau < 1):
            raise ValueError("Relative mass thresholds must lie strictly between 0 and 1.")
        low = active & (mass <= tau * m_ref)
        regular = active & ~low
        is_low = low[node_ids]
        # Includes ALL appearances of repeated CPDI node IDs by design.
        local_a = weights * acceleration[node_ids, 0]
        local_v = weights * nodal_velocity[node_ids, 0]
        local_L = nodal_velocity[node_ids, 0] * gradients[:, :, 0]
        near_dv_x = dt * np.sum(np.where(is_low, local_a, 0.0), axis=1)
        # Sum of absolute contributions bounds cancellations in the G2P sum.
        near_dv_upper = dt * np.sum(np.where(is_low, np.abs(local_a), 0.0), axis=1)
        near_pic_vx = np.sum(np.where(is_low, local_v, 0.0), axis=1)
        near_Lxx = np.sum(np.where(is_low, local_L, 0.0), axis=1)
        near_Lxx_upper = np.sum(np.where(is_low, np.abs(local_L), 0.0), axis=1)
        filtered_max = float(np.max(a_x[regular])) if np.any(regular) else 0.0
        summary["thresholds"][str(tau)] = {
            "tau": tau,
            "mass_cutoff_kg": float(tau * m_ref),
            "low_node_count": int(np.count_nonzero(low)),
            "low_nodal_mass_fraction": float(np.sum(mass[low]) / np.sum(mass)),
            "low_internal_fx_abs_fraction": float(np.sum(np.abs(internal_force[low, 0])) / max(np.sum(np.abs(internal_force[:, 0])), np.finfo(float).tiny)),
            "filtered_max_ax_m_s2": filtered_max,
            "max_abs_low_flip_dvx_m_s": float(np.max(np.abs(near_dv_x))),
            "rms_low_flip_dvx_m_s": scalar_rms(near_dv_x),
            "max_abs_low_flip_dvx_upper_m_s": float(np.max(near_dv_upper)),
            "max_abs_low_pic_vx_m_s": float(np.max(np.abs(near_pic_vx))),
            "max_abs_low_Lxx_s_inv": float(np.max(np.abs(near_Lxx))),
            "max_abs_low_Lxx_upper_s_inv": float(np.max(near_Lxx_upper)),
            "near_force_impulse_x_N_s": float(dt * np.sum(internal_force[low, 0])),
        }
    return summary


def run_diagnostic(method="cpgimp", *, cell_dx=.25, ppc_x=2, dt=.003,
                   steps=150, thresholds=DEFAULT_THRESHOLDS, sample_step=98):
    """Run unchanged original solver, observing nodal state after each step."""
    if method not in METHODS:
        raise ValueError("Unknown interpolation method.")
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer.")
    XY, p, g, solver = make_case(method, cell_dx=cell_dx, ppc_x=ppc_x, dt=dt)
    X = XY[:, 0]
    m0 = float(p.mass.sum())
    p0 = float(np.dot(p.mass, p.velocity[:, 0]))
    J0 = stretch_initial(X)
    reference_vol = p.mass / 1200.0
    from validation.stage10e.nonlinear_wave import RHO0
    def energy(J, v):
        return float(np.sum(reference_vol*(.5*RHO0*v*v + MODULUS*(J*np.log(J)-J+1.))))
    E0 = energy(J0, p.velocity[:, 0])
    summary_rows = []
    tail_rows = []
    sampled = None
    for k in range(steps):
        # Capture interpolation used by THIS step, before position/domain/F move.
        ip = evaluate_at_start(method, p, g)
        result = solver.step()
        metrics = analyze_step(
            g.mass, g.force_internal, g.acceleration, g.velocity,
            ip.node_ids, ip.weights, ip.gradients, dt=dt, particle_mass=p.mass,
            thresholds=thresholds, bc_mask=g.velocity_bc_mask,
        )
        j_full = J0 * np.linalg.det(p.deformation_gradient)
        _, _, sigma_ref = reference(X, result.time)
        stress_error = scalar_rms(p.stress[:, 0] - sigma_ref) / scalar_rms(sigma_ref)
        Edrift = (energy(j_full, p.velocity[:, 0]) - E0) / E0
        row = {"method":method, "step":result.step, "t_s":result.time,
               "stress_rel_l2":stress_error, "energy_relative_drift":Edrift,
               "mass_relative_error":(p.mass.sum() - m0)/m0,
               "momentum_x_abs_error":abs(float(np.dot(p.mass, p.velocity[:, 0])) - p0)}
        for key, val in metrics.items():
            if key != "thresholds":
                row[key] = val
        for tau_key, values in metrics["thresholds"].items():
            tag = f"tau{float(tau_key):.0e}_"
            for key, value in values.items():
                if key not in ("tau",): row[tag + key] = value
        summary_rows.append(row)
        if result.step == sample_step:
            sampled = {"step":result.step, "time_s":result.time,
                       "metrics":metrics,
                       "mass_kg":g.mass.copy(),
                       "acceleration_x_m_s2":g.acceleration[:, 0].copy(),
                       "internal_fx_N":g.force_internal[:, 0].copy(),
                       "node_xy_m":g.node_coordinates.copy()}
        # For diagnosis only, record the top acceleration node at every step.
        i = metrics["raw_max_node_id"]
        tail_rows.append({"method":method, "step":result.step, "time_s":result.time,
                          "node_id":i, "node_x_m":float(g.node_coordinates[i, 0]),
                          "node_y_m":float(g.node_coordinates[i, 1]),
                          "mass_kg":metrics["raw_max_node_mass_kg"],
                          "fx_N":metrics["raw_max_node_internal_fx_N"],
                          "accel_x_m_s2":float(g.acceleration[i, 0]),
                          "mass_ratio":metrics["raw_max_node_mass_ratio"]})
    worst = max(summary_rows, key=lambda r:r["raw_max_ax_m_s2"])
    result = {
        "method":method, "cell_dx_m":cell_dx,"ppc_x":ppc_x, "dt_s":dt,
        "steps":steps, "particles":p.n_particles,
        "mass_reference_definition":"total particle mass / number of nodes with m_i>0, evaluated each time step",
        "worst_raw_acceleration":{
            "step":worst["step"],"time_s":worst["t_s"],
            "node_id":worst["raw_max_node_id"],
            "accel_x_m_s2":worst["raw_max_ax_m_s2"],
            "nodal_mass_kg":worst["raw_max_node_mass_kg"],
            "internal_force_x_N":worst["raw_max_node_internal_fx_N"],
            "mass_ratio":worst["raw_max_node_mass_ratio"]},
        "max_stress_rel_l2":max(r["stress_rel_l2"] for r in summary_rows),
        "final_stress_rel_l2":summary_rows[-1]["stress_rel_l2"],
        "max_abs_energy_relative_drift":max(abs(r["energy_relative_drift"]) for r in summary_rows),
        "max_momentum_x_abs_error":max(r["momentum_x_abs_error"] for r in summary_rows),
        "max_mass_rel_error":max(abs(r["mass_relative_error"]) for r in summary_rows),
        "thresholds":{},
    }
    for tau in thresholds:
        tag = f"tau{float(tau):.0e}_"
        result["thresholds"][str(float(tau))] = {
            "tau":float(tau),
            "max_excluded_nodes_per_step":max(r[tag+"low_node_count"] for r in summary_rows),
            "max_filtered_ax_m_s2":max(r[tag+"filtered_max_ax_m_s2"] for r in summary_rows),
            "max_low_nodal_mass_fraction":max(r[tag+"low_nodal_mass_fraction"] for r in summary_rows),
            "max_low_force_abs_fraction":max(r[tag+"low_internal_fx_abs_fraction"] for r in summary_rows),
            "max_abs_low_flip_dvx_m_s":max(r[tag+"max_abs_low_flip_dvx_m_s"] for r in summary_rows),
            "max_abs_low_flip_dvx_upper_m_s":max(r[tag+"max_abs_low_flip_dvx_upper_m_s"] for r in summary_rows),
            "max_abs_low_pic_vx_m_s":max(r[tag+"max_abs_low_pic_vx_m_s"] for r in summary_rows),
            "max_abs_low_Lxx_s_inv":max(r[tag+"max_abs_low_Lxx_s_inv"] for r in summary_rows),
            "max_abs_low_Lxx_upper_s_inv":max(r[tag+"max_abs_low_Lxx_upper_s_inv"] for r in summary_rows),
        }
    return result, summary_rows, tail_rows, sampled


def _csv(path, rows):
    if not rows: return
    with Path(path).open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for method in dict.fromkeys(r["method"] for r in rows):
        r = [row for row in rows if row["method"] == method]
        ax.plot([a["t_s"] for a in r],[a["raw_max_ax_m_s2"] for a in r],label=method)
    ax.set(xlabel="Time (s)", ylabel="Max |nodal a_x| (m/s²)", title="Unfiltered nodal acceleration (diagnostic only)")
    ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path,dpi=170);plt.close(fig)


def plot_mass_accel(sampled, path):
    if sampled is None: return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m=sampled["mass_kg"];a=np.abs(sampled["acceleration_x_m_s2"])
    good=(m>0) & (a>0)
    fig,ax=plt.subplots(figsize=(7.6,4.5))
    ax.scatter(m[good], a[good], s=11,alpha=.65)
    ax.set_xscale("log");ax.set_yscale("log")
    threshold=sampled["metrics"]["thresholds"].get(str(1e-12))
    if threshold is not None:
        ax.axvline(threshold["mass_cutoff_kg"],ls="--",lw=1.2,
                   label=r"diagnostic cutoff: $10^{-12}m_{ref}$")
        ax.legend(loc="lower left",fontsize=8)
    worst_id=sampled["metrics"]["raw_max_node_id"]
    ax.annotate(f"Node {worst_id}", xy=(m[worst_id],a[worst_id]),
                xytext=(12,8),textcoords="offset points",fontsize=9)
    ax.set(xlabel="Positive nodal mass m_i (kg)", ylabel="|a_x| (m/s²)",
           title=f"Near-massless nodes at step {sampled['step']}")
    ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(path,dpi=170);plt.close(fig)


def validate(*, methods=METHODS, output="validation/results/stage10e_near_massless",
             steps=150, sample_step=98, dt=.003, ppc_x=2,
             thresholds=DEFAULT_THRESHOLDS, plot=True):
    methods=tuple(methods)
    if not methods or any(m not in METHODS for m in methods):
        raise ValueError("methods must be drawn from mpm,gimp,cpgimp,cpdi.")
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    cases=[];records=[];tail=[];samples={}
    for method in methods:
        print(f"Diagnosing {method} (solver unchanged)...",flush=True)
        result,rows,tail_rows,sample=run_diagnostic(method,steps=steps,dt=dt,ppc_x=ppc_x,
                                   thresholds=thresholds,sample_step=sample_step)
        cases.append(result);records.extend(rows);tail.extend(tail_rows)
        if sample is not None:
            samples[method]=sample
        worst=result["worst_raw_acceleration"]
        ref=result["thresholds"][str(float(thresholds[1] if len(thresholds)>1 else thresholds[0]))]
        print(f"  raw max={worst['accel_x_m_s2']:.6g} m/s²; node={worst['node_id']}; "
              f"mass={worst['nodal_mass_kg']:.3e} kg; "
              f"max low-node FLIP contribution={ref['max_abs_low_flip_dvx_m_s']:.3e} m/s",flush=True)
    checks={
        "unchanged_solver_run_all_requested":len(cases)==len(methods),
        "finite_nodal_diagnostics":all(np.isfinite(r["raw_max_ax_m_s2"]) for r in records),
        "mass_conserved":all(c["max_mass_rel_error"]<1e-12 for c in cases),
        "momentum_conserved":all(c["max_momentum_x_abs_error"]<1e-6 for c in cases),
        "nonnegative_filtered_mass":all(v["max_low_nodal_mass_fraction"]>=0
                             for c in cases for v in c["thresholds"].values()),
        "hypothetical_contributions_finite":all(np.isfinite(v["max_abs_low_flip_dvx_m_s"])
                             for c in cases for v in c["thresholds"].values()),
    }
    summary={"checks":checks,"cases":cases,
             "meaning":"Observational separation ONLY. No node was deleted or suppressed in the actual solver.",
             "thresholds_relative_to":"step-wise mean nodal mass among all m_i>0 nodes",
             "caveat":"A small masked contribution is evidence for this benchmark, not a universal near-massless-node guarantee."}
    _csv(out/"10E_A1_step_metrics.csv",records)
    _csv(out/"10E_A1_worst_nodes.csv",tail)
    (out/"10E_A1_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
    if plot:
        plot_results(records,out/"10E_A1_acceleration_time.png")
        if "cpgimp" in samples:
            plot_mass_accel(samples["cpgimp"],out/"10E_A1_cpgimp_mass_acceleration.png")
    print("CHECKS:",checks)
    print("RESULTS:",out)
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods",default="all",help="all or comma-separated mpm,gimp,cpgimp,cpdi")
    ap.add_argument("--steps",type=int,default=150)
    ap.add_argument("--sample-step",type=int,default=98)
    ap.add_argument("--dt",type=float,default=.003)
    ap.add_argument("--ppc-x",type=int,default=2)
    ap.add_argument("--thresholds",default="1e-8,1e-12,1e-16")
    ap.add_argument("--output",default="validation/results/stage10e_near_massless")
    ap.add_argument("--no-plot",action="store_true")
    args=ap.parse_args()
    methods=METHODS if args.methods=="all" else tuple(a.strip() for a in args.methods.split(","))
    thresholds=tuple(float(a) for a in args.thresholds.split(","))
    result=validate(methods=methods,steps=args.steps,sample_step=args.sample_step,
           dt=args.dt,ppc_x=args.ppc_x,thresholds=thresholds,
           output=args.output,plot=not args.no_plot)
    if not all(result["checks"].values()):
        raise SystemExit("Diagnostic sanity checks did not all pass.")


if __name__=="__main__":
    main()
