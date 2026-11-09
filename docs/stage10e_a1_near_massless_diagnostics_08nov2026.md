# GeoPyMPM — Stage 10E-A.1: Near-Massless Nodal Diagnostics

**Date:** 8 November 2026  
**Baseline:** Stage 10E (7 November 2026)  
**Scope:** Diagnostic only; the `src/geopympm/` solver and its kernels are **unchanged**.

## 1. Motivation

The Stage 10E nonlinear-wave benchmark reported a maximum axial nodal acceleration of
94.05851059136754 m/s² for `cpgimp`, although the final stress error was only 1.345%.
The question is whether a very low-mass grid node affects particle motion or
merely dominates the reported **unfiltered nodal maximum**.

The underlying solver uses the positive-mass condition `m_i > 0` and divides
`nodal momentum change / (m_i * dt)` for every such node. Neither its current
P2G nor its grid integration uses a *relative* mass cutoff. The diagnostic
does not change that behavior.

### Core equations

- P2G: `m_i = sum_p S_ip m_p`.
- Internal force: `f_int,i = - sum_p V_p sigma_p grad(S_ip)`.
- Grid (free x DOFs): `a_ix = (p_ix,new - p_ix,old)/(m_i dt)`.
- Actual FLIP x-velocity increment: `dv_px = dt sum_i S_ip a_ix`.
- PIC contribution to grid-interpolated velocity: `v_PIC,px = sum_i S_ip v_ix`.
- Axial velocity-gradient contribution: `L_p,xx = sum_i v_ix dS_ip/dx`.

The shape functions `S_ip` and gradients used for this analysis are sampled
**before** each `solver.step()`, while nodal mass, velocity and acceleration
are read **after** the step. These are precisely the nodal quantities from
that step, which used the old interpolation and old particle positions.

## 2. Diagnostic threshold (not a solver setting)

Define a characteristic nodal mass at each step:

`m_ref = sum_p m_p / #{i : m_i > 0}`.

A node is classified as **near-massless for observation** when
`0 < m_i <= tau * m_ref`, with `tau = 1e-8, 1e-12, 1e-16`.

At each threshold the report contains:

1. Count, mass fraction and absolute internal-force fraction of low-mass nodes.
2. Largest `|a_ix|` among the **remaining** nodes (a filtered *statistic*).
3. Largest signed and absolute-sum upper bounds for the actual low-node part
   of the FLIP `dv_px` sum, using the same pre-step `S_ip` as the solver.
4. Largest low-node term in the PIC velocity and axial velocity gradient.
5. Mass, momentum and energy checks using the **original, unaltered** solver.

These metrics do **not** simulate replacing nodal accelerations with zero;
they only isolate terms in the existing particle update. The upper bound
uses `dt * sum(|S_ip * a_ix|)` to detect cancellation that a signed sum could
hide. CPDI repeated node IDs are retained consistently with actual kernels.

## 3. Reproduction commands

From the repository root with the project's virtual environment activated:

```bash
source .venv/bin/activate
python -m validation.stage10e.near_massless --methods all
python -m pytest tests/test_stage10e_near_massless.py -q
```

Focused replication of the cpGIMP outlier:

```bash
python -m validation.stage10e.near_massless --methods cpgimp --sample-step 98
```

Outputs are stored in `validation/results/stage10e_near_massless/`:

- `10E_A1_summary.json`: raw and filtered maxima, influence bounds, conservation checks.
- `10E_A1_step_metrics.csv`: step-by-step data for all tested methods.
- `10E_A1_worst_nodes.csv`: the highest-acceleration node at each step.
- `10E_A1_acceleration_time.png`: unfiltered axial acceleration history.
- `10E_A1_cpgimp_mass_acceleration.png`: node-mass-versus-acceleration snapshot at step 98.

## 4. Results from unchanged Stage 10E benchmark

Parameters: 512 particles, grid `h=0.25 m`, 2 particles per cell in each
direction, `dt=0.003 s`, 150 steps (`t=0.45 s`), `pic_fraction=0`,
corotational Jaumann stress update, exponential kinematics and affine-exponential
particle advection. Transverse grid velocities are prescribed zero.

| Method | Raw max `|a_x|`, m/s² | Max `|a_x|` at significant-mass nodes, m/s² (`tau=1e-12`) | Final stress relative L2 |
|---|---:|---:|---:|
| MPM | 4.81139 | 4.81139 | 0.264620 |
| uGIMP | 0.605262 | 0.605262 | 0.015455 |
| cpGIMP | **94.05851** | **0.501977** | **0.013452** |
| CPDI1 | 0.468713 | 0.468713 | 0.015035 |

The raw cpGIMP maximum occurs at **step 98**, `t=0.294 s`, node **656**:

| Quantity | Measured value |
|---|---:|
| Nodal mass | `3.937708418319498e-34 kg` |
| Internal force x | `3.7037498897022167e-32 N` |
| Acceleration x | `94.05851059136754 m/s²` |
| `m_i / m_ref` | `8.039488020735642e-36` |

For cpGIMP, among thresholds `tau=1e-8, 1e-12, 1e-16`, the results
were identical in this experiment (the low-node masses are much smaller
than any of those cutoffs):

| Influence over all steps | Largest observed value |
|---|---:|
| Classified low-mass nodes in one step | 28 |
| Low-node mass fraction | `6.741e-34` |
| Low-node absolute `f_int,x` fraction | `4.351e-33` |
| `max_p |dt sum_low S_ip a_ix|` | `4.957e-35 m/s` |
| `max_p dt sum_low |S_ip a_ix|` (cancellation-free bound) | `6.323e-34 m/s` |
| `max_p |sum_low S_ip v_ix|` | `2.096e-32 m/s` |
| `max_p |sum_low v_ix dS_ip/dx|` | `9.077e-33 1/s` |
| `max_p sum_low |v_ix dS_ip/dx|` | `1.454e-31 1/s` |

The near-empty nodes were classified by a **relative mass reference**,
not by a numerical constant in kilograms. The numerically tiny mass and
force are consistent with vanishing shape-function support (potentially
created by floating-point effects in the domain-integrated interpolation).
The exact origin of each tiny coefficient was **not** independently proved
by this diagnostic; root-cause investigation requires a targeted basis-weight
trace, if desired.

**Interpretation:** The 94 m/s² number dominates the raw *nodal* maximum
but its weighted contributions to this benchmark's particle FLIP velocity
and axial strain-rate are negligible. It is not evidence of a physical
particle acceleration of 94 m/s². In this specific scenario, Stage 10E's
low cpGIMP stress error is compatible with the very small influence bounds.

This does NOT prove near-massless nodes are harmless in other materials,
strong boundary loads, extreme rotations, contact configurations or
later elastoplastic models.

## 5. Checks and decision

All six script sanity checks passed: method execution; finite diagnostics;
mass and x-momentum conservation; valid nonnegative relative mass fraction;
and finite hypothetical weighted contributions.

Seven additional pytest cases verify: exact synthetic FLIP/gradient
contributions; input immutability; invalid input rejection; reproduction
of the step-98/node-656 outlier; and equality of diagnostic and
non-diagnostic particle results.

**Decision:** Keep the existing solver unchanged. Do not globally remove
low-mass nodes or introduce a new `mass_tol` based only on this example.
Keep a *mass-aware acceleration diagnostic* separate from the actual
momentum update. If future force-driven 2D tests show materially large
low-node contributions, examine robust numerical handling in a separate,
fully regression-tested solver change.
