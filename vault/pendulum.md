# Pendulum BB Open-Loop TPBVP

This is the finite-horizon open-loop pendulum problem used for BB data
generation. It follows the reduced-gradient PMP workflow used in the thesis:
for each initial state, solve an ODE optimal-control problem, then store the
optimal value and the initial adjoint `p(0)` as the value-gradient sample.

## State, Dynamics, And Cost

The state is two-dimensional:

`x = (theta, omega)`.

The controlled pendulum dynamics around the upright equilibrium are

`theta_dot = omega`

`omega_dot = -d omega + gamma sin(theta) + c u`

where

`c = 1 / (m l^2)`, `d = b / (m l^2)`, and `gamma = g / l`.

For horizon `T`, the finite-horizon open-loop objective is

`J(u; x0) = integral_0^T [q1(2 - 2 cos(theta)) + q2 omega^2 + R u^2] dt + Phi(x(T))`.

The terminal penalty `Phi` is the local LQR value around the upright state:

`Phi(theta, omega) = terminal_weight * e^T P e`,

with `e = (wrap(theta), omega)`. Setting `terminal_weight = 0` recovers the
plain thesis-style finite-horizon terminal condition `p(T)=0`.

## Hamiltonian

Use the same sign convention as the existing VDP open-loop generator:

`H = q1(2 - 2 cos(theta)) + q2 omega^2 + R u^2`

`    + p1 omega + p2(-d omega + gamma sin(theta) + c u)`.

The stationarity condition is

`G(u)(t) = c p2(t) + 2 R u(t) = 0`.

So the unconstrained optimal control implied by the value gradient is

`u*(theta, omega) = -c V_omega(theta, omega) / (2 R)`.

The stationary infinite-horizon HJB equation for the same running cost is

`0 = min_u {q1(2 - 2 cos(theta)) + q2 omega^2 + R u^2`

`           + V_theta omega + V_omega(-d omega + gamma sin(theta) + c u)}`.

After eliminating `u`, this becomes

`0 = q1(2 - 2 cos(theta)) + q2 omega^2`

`    + V_theta omega + V_omega(-d omega + gamma sin(theta))`

`    - c^2 V_omega^2 / (4 R)`.

## TPBVP For A Fixed Open-Loop Control

For a trial control function `u(t)`, the BVP solved inside the BB loop is

`theta_dot = omega`

`omega_dot = -d omega + gamma sin(theta) + c u(t)`

`p1_dot = -2 q1 sin(theta) - gamma cos(theta) p2`

`p2_dot = -2 q2 omega - p1 + d p2`

with boundary conditions

`theta(0) = theta0`

`omega(0) = omega0`

`p(T) = dPhi(x(T))`.

For the local LQR terminal penalty,

`dPhi = 2 terminal_weight P (wrap(theta(T)), omega(T))`.

When `terminal_weight = 0`, this reduces to `p1(T)=0`, `p2(T)=0`.

## Dataset Sample

After BB convergence, the training sample is

`x = x0`

`v = J(u*; x0)`

`dv = p*(0)`.

The value-gradient identity `dv = p*(0)` is the PMP/HJB link used in the
thesis. For the pendulum, multiple open-loop local minimizers can exist, so the
dataset generator supports multiple initial control seeds and keeps the lowest
value. That is necessary for exposing the switching geometry where the true
value gradient is discontinuous.

# pendulum: infinite horizon

The paper-backed pendulum problem is an autonomous infinite-horizon optimal
control problem. The value function maps a state to the remaining optimal cost:

`V(x) = cost from state x to the upright equilibrium`.

Because the problem is autonomous, `V(x)` does not depend on absolute time. If a
backward-PMP trajectory gives a point `x(tau)` with accumulated value `V(tau)`,
that value means the forward cost-to-go from `x(tau)` back to the upright
equilibrium.

The paper generates raw PMP trajectories by sampling terminal states on a small
local-LQR level-set boundary near the upright equilibrium, then integrating the
state and costate backward. Along this backward integration parameter `tau`, the
accumulated value increases as the trajectory moves away from the local LQR
region.

Boundary sampling should use the paper's adaptive idea rather than only uniform
angles. Uniform boundary angles can cluster backward-PMP trajectories and leave
important regions undercovered. In the cleaned solver, adaptive boundary
sampling is a separate stage/helper that chooses boundary angles before raw PMP
trajectory integration.

For a contour level `k * delta`, each raw trajectory contributes the state whose
accumulated value is closest to that level from below:

`jmax = max { j | trajectory(j).value <= k * delta }`.

This does not choose the earliest physical time across trajectories. Different
trajectories may reach the same value level at different backward integration
times. The contour construction matches equal cost-to-go value, not equal time.

The nonsmooth curve is found by forming the equal-value contour in
`(theta, omega)` space and intersecting it with a copy shifted by `2 pi` in the
angle direction. Such intersections represent the same physical pendulum state
being reached by two periodic swing-up branches with the same value. This
nonsmooth-curve handling is part of the solver contract for producing final
`ValueSamples`; raw trajectory points without branch restriction are diagnostics,
not the dataset. After the curve is identified, raw PMP trajectories are
restricted at their first crossing so the retained samples represent the
lower-value branch of the value function.

The solver output needed for training is therefore the open-loop value sample:

`x = state`

`v = V(x)`

`dv = dV(x) = costate`.

## Clean Module Shape

`PendulumSwingUpProblem` should be the self-contained mathematical problem. It
owns the physical parameters, cost weights, optional control saturation,
dynamics, running cost, Hamiltonian-minimizing control, costate RHS, and local
LQR value/gradient. A separate dynamics wrapper is unnecessary once the
finite-horizon and BB pendulum implementations are removed.

The numerical run configuration should live outside the problem object. Solver
tolerances, integration horizon, value limits, sampling density, contour
resolution, and output choices belong to solver or stage config objects.

The implementation should include concise comments around paper-specific
algorithm steps: backward-PMP integration, adaptive boundary sampling,
equal-value contour extraction, shifted contour intersections, and branch
restriction. Comments should explain why the step exists, not restate simple
assignments.

The cleaned pendulum module layout should be:

`problem.py`: `PendulumSwingUpProblem`, the mathematical OCP.

`solver.py`: `PendulumPmpSolver`, solver configuration, and the high-level
`solve()` pipeline.

`trajectories.py`: raw `PmpTrajectory` data and backward-PMP trajectory
integration helpers.

`nonsmooth.py`: `NonsmoothCurve` and equal-value contour/intersection logic.
This module owns the Shapely dependency and converts geometry results back to
plain NumPy arrays.

`samples.py`: `ValueSamples`, PDAP dictionary conversion, and `.npz` save/load
helpers.

The pendulum module should not depend on the generic `sample_sets.py` helpers.
Those helpers mix rectangular state-space sampling with file persistence and are
not central to the infinite-horizon PMP algorithm. Pendulum should expose its
own `ValueSamples` data object and pendulum-specific save/load helpers.

Internally, `ValueSamples` should store separate plain NumPy arrays:
`x`, `v`, and `dv`, all coerced to `float64`. Its PDAP-facing output should be a
dictionary with the same keys, `{"x": x, "v": v, "dv": dv}`, because the PDAP
preprocessing path expects that dictionary format before converting to
`torch.float64`.

On disk, pendulum `ValueSamples` should be saved as `.npz` files with `x`, `v`,
and `dv` arrays. This preserves the same key-based shape expected by PDAP
without making structured NumPy dtypes the internal representation.

The old pendulum-only finite-horizon and BB implementations should be removed
when the new solver is introduced. Keep VDP/shared utilities outside this
cleanup scope.

The first implementation milestone stops at reliable `ValueSamples` generation
for PDAP training. Interpolation of `V`/`dV` at arbitrary states and controller
synthesis are downstream validation tasks, not part of this cleanup.

After the nonsmooth curve is detected, branch restriction happens before
building final `ValueSamples`. For each raw trajectory, keep the samples before
its first crossing of the nonsmooth curve and discard samples after that point.
The first implementation may use an approximate cut at the nearest detected
sample index, but diagnostics must report how many points were discarded.
