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
