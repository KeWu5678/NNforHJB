# Power-Transformed Penalty (general activation power p)

Sub-level disclosure for `../CLAUDE.md`. Technical detail of the `ReLU^p`
penalty and the proximal dead-zone bug for `p != 1`.

When the activation is `ReLU^p` with `p != 1`, the regularization uses
`phi(|u|^q)` instead of `phi(|u|)`, where `q = 2/(p+1)`. For `p=1`, `q=1` and
everything reduces to the standard case.

**Files involved** (all changes backward-compatible for p=1):
- `SSN/prox.py` + `SSN/penalty.py`: `_compute_prox(v, mu, q)`, `_compute_dprox(v, mu, q, prox_result)`, `_phi_prox(sigma, g, th, gamma, q)` — accept optional `q`. For `q != 1`, use Newton's method instead of closed form. (Re-exported from `utils.py` for backward compat.)
- `SSN/optimizer.py`: stores `self.q = 2/(power+1)`. `_initialize_q`, `_initialize_G`, `_DG` use chain-rule derivatives with an `active` mask for the `|u|^{q-1}` singularity at `u=0`. `_prox`/`_dprox` pass `self.q` to the proximal operators.
- `models/signed.py`: `_compute_loss` applies the `|u|^q` transform before `_phi`. `_setup_optimizer` passes `power` to `SSN` (and maps `optimizer_type` to `method=`). `warm_start` passes `q=2/(p+1)` to `_phi_prox`.
- `models/semiconcave.py`: `warm_start` likewise passes `self.q` to `_phi_prox`.

**Key math** (derivatives w.r.t. `t = |u|`, for `t > 0`):
- Full penalty gradient: `q * t^{q-1} * dphi(t^q)`
- Correction 2nd derivative: `q(q-1) * t^{q-2} * (dphi(t^q)-1) + q^2 * t^{2q-2} * ddphi(t^q)`
- Proximal of `mu * |.|^q` for `q < 1`: Newton solve of `t + mu*q*t^{q-1} = |v|` with threshold `t* = [mu*q*(1-q)]^{1/(2-q)}`

## SSN Proximal Dead Zone Bug (p != 1, OPEN)

**Status**: Diagnosed but NOT fixed. SSN currently broken for `power != 1` (q < 1).

**Stashed fix attempt**: `git stash@{0}` contains a partial fix (DPc clamping +
active-set preservation + dead-zone zeroing). Stashed because it also broke the
regular `power=1` case. Apply with `git stash pop`, discard with `git stash drop`.

**Symptom**: SSN line search fails on every step when `power != 1` (e.g.
`power=2.1`, `q=0.645`). Train loss decreases only from the coordinate-descent
warm-start; SSN contributes nothing.

**Root cause 1 — proximal jump discontinuity (two-branch problem)**:
The proximal of `mu * |.|^q` for `q < 1` has a jump discontinuity. The
stationarity condition `t + mu*q*t^{q-1} = v` has **two roots** for
`v > v_thresh`: `t_large` (local min, SOC > 0) and `t_small` (local max,
SOC < 0). `_compute_prox` always returns `t_large`.
SSN's `_initialize_q` inverts the FOC by plugging `t = |u_i|` to get `q_var_i`.
For small weights (`|u_i| < t*`), `|u_i|` is the `t_small` root. Then
`_compute_prox(q_var)` returns `t_large != |u_i|`, so `prox(q) != params` — the
fundamental SSN consistency assumption breaks. For `q=1` the equation
`t + mu = v` has a unique root (no ambiguity).

**Root cause 2 — indefinite Jacobian DG**:
For `q < 1`, the proximal Jacobian `DPc_i = 1/SOC_i` can exceed 1 when
`SOC_i < 1` (weights near the dead-zone boundary), making the generalized
Jacobian DG indefinite:
```
DG_{ii} = c + (H_data_{ii} - c) * DPc_i
```
When `DPc_i > 1` and `H_data_{ii} < c`, `DG_{ii} < 0`. The Newton direction
becomes an ascent direction, so line search can never succeed. For `q=1`,
`DPc_i in {0, 1}` (a projection), so `DG` is always PSD.

**Root cause 3 — inactive weight activation via proximal jump**:
For inactive weights (`u_i = 0`), `_initialize_q` sets `q_i = -(1/c) * grad_flat_i`.
If `|grad_flat_i| / c > v_thresh`, `prox(q_i)` jumps to a nonzero value,
activating the neuron. For `q=1` this is smooth (continuous soft-thresholding);
for `q<1` it is a discontinuous jump from 0 to `~t*`.

**Key threshold**: `t* = [mu*q*(1-q)]^{1/(2-q)}` where `mu = alpha/c`,
`c = 1 + alpha*gamma`. For `alpha=1e-5, gamma=0, q=0.645`: `t* ~ 7e-5`. Weights
below this are in the dead zone.

**Stashed fix approach** (in `git stash@{0}`):
1. Dead-zone zeroing: zero weights where SOC <= 0 before SSN, force `q[dead_zone] = 0`.
2. DPc clamping: clamp `_compute_dprox` diagonal to `[0, 1]` in `_DG` to keep the Jacobian PSD.
3. Active-set preservation: force `unew[inactive] = 0` after prox to prevent jump activation.
4. Inactive q clamping: force `q[inactive] = 0` for all zero-weight entries.

**Why the stashed fix failed**: the combined changes also broke the `power=1`
(q=1) case — edits to `_initialize_q`, `_initilize_G`, `_DG` altered the code
paths used by q=1 too. Ensure backward compatibility before re-applying.

**Visualization**: `scripts/visualize_proximal_deadzone.py` produces a 4-panel
figure (FOC, SOC, proximal objective, jump discontinuity).
