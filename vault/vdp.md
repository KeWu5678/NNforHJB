# VDP Open-Loop Smooth Benchmark

The Van der Pol open-loop problem provides the smooth value-function dataset
for PDAP tests. Pendulum covers the nonsmooth case; VDP should therefore use one
fixed terminal target when generating the default training dataset.

## Solver Structure

The cleaned VDP implementation should separate three concerns:

`VdpOptimalControlProblem`: the mathematical optimal-control problem. It owns
the VDP dynamics, running cost, terminal condition `p(T)=0`, and stationarity
residual.

`VdpOpenLoopSolver`: the numerical pipeline for one or many initial states. It
parameterizes the open-loop control, evaluates objective and gradient by
forward state integration plus backward adjoint integration, calls a coefficient
optimizer backend, and returns value samples for PDAP.
The cleaned implementation should support two explicit numerical profiles:

`"fast"`: `solve_ivp` forward/backward integration, Legendre control
coefficients, and L-BFGS-B coefficient optimization. This is the fastest and
simplest implementation for routine experiments.

`"paper"`: Crank-Nicolson forward/backward integration, time-grid control
values, and Barzilai-Borwein updates. This is the closest implementation to the
paper's reduced-gradient algorithm.

`CoefficientOptimizer`: the backend that updates the control coefficients. The
initial cleaned implementation should support SciPy `minimize` and
Barzilai-Borwein as explicit options, not as separate solver architectures.
The solver config should call this field `coefficient_optimizer`, with explicit
values such as `"barzilai_borwein"` and `"l_bfgs_b"`. The default should be
`"barzilai_borwein"` for the VDP benchmark.

The solver should support two control parameterizations:

`"time_grid"`: optimize control values on the configured time grid. This is the
closest implementation to the paper's reduced-gradient BB algorithm. With
`solve_ivp`, the control is evaluated between grid points by interpolation.

`"legendre"`: optimize a low-dimensional Legendre coefficient vector. This is a
faster optional mode and matches the current forward-backward implementation,
but it restricts the admissible control shape.

The default VDP benchmark mode should be the `"paper"` profile:
Crank-Nicolson integration, `"time_grid"` control parameterization, and
`"barzilai_borwein"` optimization. The `"fast"` profile should be requested
explicitly when speed is more important than paper fidelity.

For the `"paper"` profile, the default time step should match the paper's VDP
experiment: `time_step=1e-4` on `T_final=3`.
The implicit Crank-Nicolson state and adjoint steps should initially solve
their small nonlinear equations with `scipy.optimize.root`.

For the `"fast"` profile, keep the current forward-backward defaults:
`num_control_basis=30`, `301` time-grid points on `T_final=3`, L-BFGS-B
optimization, and `solve_ivp` tolerances `rtol=1e-7`, `atol=1e-9`.

The forward-backward solver machinery belongs under `src/OpenLoop/vdp/`, not at
the root of `src/OpenLoop/`. Pendulum now has an independent infinite-horizon
PMP implementation, so leaving finite-horizon forward-backward code at the
OpenLoop root incorrectly presents it as shared infrastructure.

Only concepts genuinely shared by VDP and pendulum should live at the OpenLoop
root. Today that is the `ValueSamples` contract: final `(x, v, dv)` arrays,
PDAP dictionary conversion, and `.npz` save/load. Pendulum is now independent
and does not need the old `sample_sets.py` helpers, so rectangular VDP
initial-state sampling should not remain as shared OpenLoop infrastructure.

## Barzilai-Borwein

Barzilai-Borwein is a coefficient optimization method, not a different
mathematical VDP problem. The current BVP/BB implementation mixes coefficient
parameterization, state-costate solve, gradient projection, line search, and
dataset-generation concerns. In the cleaned design, BB should be preserved as
an optional backend behind the same solver interface used by SciPy optimizers.

The common interface should make the optimization target explicit:

`objective(coefficients) -> (value, coefficient_gradient, diagnostics)`.

Both BB and SciPy optimizers should consume this same evaluator. This prevents
the choice of optimizer from changing the problem definition, sample format, or
PDAP-facing output.

For the Barzilai-Borwein backend, convergence should use the time-domain
reduced-gradient norm that appears in the paper:

`sqrt(integral_0^T G(u)(t)^2 dt) <= 1e-5`.

For VDP, `G(u)(t) = p2(t) + 2 beta u(t)`. In `"time_grid"` mode, this gradient
directly updates the control values. In `"legendre"` mode, it is projected onto
the Legendre basis before updating coefficients. The projected
coefficient-gradient norm should still be stored as a diagnostic and used by
SciPy optimizers, but it is not the BB stopping criterion.

Both `"fast"` and `"paper"` profiles should report convergence using the
time-domain reduced-gradient norm. Optimizer-specific norms, such as the
projected Legendre coefficient-gradient norm, are diagnostics rather than the
official convergence criterion.

## Dataset Contract

The VDP solver output for PDAP is the same open-loop value sample shape used by
pendulum:

`x = initial state`

`v = V(x)`

`dv = dV(x) = p(0)`

The training-facing output should be a dictionary with NumPy arrays:

`{"x": x, "v": v, "dv": dv}`.

New VDP datasets should be saved as `.npz` files with `x`, `v`, and `dv` keys,
matching the pendulum dataset format and the PDAP loaders. The old structured
`.npy` dtype is legacy and should not be used for new VDP output.
The durable training artifact should be written under `rawdata/data/` so it can
be referenced directly by the PDAP data config.
Dataset filenames should omit `openloop` and include the generation date, for
example `VDP_paper_grid_30x30_20260605.npz`, with matching metadata and failed
sample sidecars.
Dataset generation should not automatically update `conf/data/vdp.yaml`; the
training config should be changed manually after inspecting the generated
dataset.

The legacy two-target phase-capture minimum should be removed. It is not needed
for the smooth VDP benchmark and can accidentally create an artificial
nonsmooth dataset.

The old `DataGenerator` class should be retired as a public interface. It mixes
problem definition, state sampling, optimization, progress output, and file
persistence. The official VDP entry point should be `VdpOptimalControlProblem`,
`VdpOpenLoopSolverConfig`, and `VdpOpenLoopSolver.solve(initial_states)`.

The solver should accept explicit `initial_states`. Rectangular grid/random
sampling can live in a VDP script or VDP-local helper, but it should not be a
responsibility of the solver itself.

The solver should return a rich solution object. `value_samples` contains only
converged samples for PDAP training, while per-sample results retain convergence
status, messages, final value/gradient, reduced-gradient norm, iterations, and
failed initial states for diagnosis.

The root-level `bvp_bb_optimizer.py` should be removed with `DataGenerator`.
Barzilai-Borwein remains supported, but only as a coefficient optimizer backend
using the same forward-backward objective/gradient evaluator as the SciPy
backend.
