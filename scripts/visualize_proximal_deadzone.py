"""
Visualize the proximal dead zone problem for mu * |.|^q with q < 1.

The proximal operator solves:
    prox(v) = argmin_{t >= 0}  mu * t^q + (1/2)(t - v)^2

For q < 1, this has a jump discontinuity: the FOC equation
    t + mu*q*t^{q-1} = v
has TWO roots for v > v_thresh, and the proximal always picks the large one.

The SSN bug: _initialize_q inverts the FOC assuming prox(q_var) = u,
but for small |u| < t*, |u| is on the LOCAL MAX branch (SOC < 0),
so prox(q_var) != u.
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters matching the training setup
alpha = 1e-5
gamma = 0.0
power = 2.1
q_exp = 2.0 / (power + 1.0)  # = 0.645
c = 1.0 + alpha * gamma      # = 1.0
mu = alpha / c                # = 1e-5

print(f"power = {power}, q = {q_exp:.4f}, mu = alpha/c = {mu:.2e}")

# Dead zone threshold
t_star = (mu * q_exp * (1.0 - q_exp)) ** (1.0 / (2.0 - q_exp))
v_thresh = t_star + mu * q_exp * t_star ** (q_exp - 1)
print(f"t* = {t_star:.6e}")
print(f"v_thresh = {v_thresh:.6e}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ──────────────────────────────────────────────────────────
# Panel 1: The FOC equation g(t) = t + mu*q*t^{q-1}
# This is the key plot. For q<1, g(t) has a U-shape:
#   g(0+) = +inf, g decreases to v_thresh at t*, then increases to +inf.
# For a given v > v_thresh, the horizontal line g(t) = v
# intersects at TWO points: t_small (local max) and t_large (local min).
# ──────────────────────────────────────────────────────────
ax = axes[0, 0]
t = np.linspace(1e-8, 5 * t_star, 2000)
g = t + mu * q_exp * t ** (q_exp - 1)

ax.plot(t, g, 'b-', linewidth=2, label=r'$g(t) = t + \mu q\, t^{q-1}$')
ax.axhline(y=v_thresh, color='r', linestyle='--', linewidth=1.5, label=f'v_thresh = {v_thresh:.2e}')
ax.plot(t_star, v_thresh, 'ro', markersize=10, zorder=5, label=f't* = {t_star:.2e}')

# Show an example v and its two roots
v_example = 2.5 * v_thresh
ax.axhline(y=v_example, color='green', linestyle=':', linewidth=1.5, label=f'example v = {v_example:.2e}')

# Find the two roots numerically
from scipy.optimize import brentq
# t_small: root in (0, t_star)
t_small = brentq(lambda t: t + mu * q_exp * t ** (q_exp - 1) - v_example, 1e-12, t_star)
# t_large: root in (t_star, big)
t_large = brentq(lambda t: t + mu * q_exp * t ** (q_exp - 1) - v_example, t_star, 10 * v_example)

ax.plot(t_small, v_example, 'rx', markersize=12, markeredgewidth=3, zorder=5)
ax.plot(t_large, v_example, 'g^', markersize=12, zorder=5)
ax.annotate(f't_small = {t_small:.2e}\n(LOCAL MAX, SOC < 0)',
            xy=(t_small, v_example), xytext=(t_small + t_star * 0.5, v_example * 1.3),
            fontsize=9, color='red', arrowprops=dict(arrowstyle='->', color='red'))
ax.annotate(f't_large = {t_large:.2e}\n(LOCAL MIN, SOC > 0)',
            xy=(t_large, v_example), xytext=(t_large, v_example * 1.4),
            fontsize=9, color='green', arrowprops=dict(arrowstyle='->', color='green'))

ax.set_xlabel('t')
ax.set_ylabel('g(t)')
ax.set_title('FOC: g(t) = v has TWO roots for v > v_thresh')
ax.legend(fontsize=8, loc='upper right')
ax.set_ylim(0, 5 * v_thresh)
ax.grid(True, alpha=0.3)

# ──────────────────────────────────────────────────────────
# Panel 2: The SOC = 1 + mu*q*(q-1)*t^{q-2}
# Positive = local min (valid proximal), Negative = local max (invalid)
# ──────────────────────────────────────────────────────────
ax = axes[0, 1]
t = np.linspace(1e-8, 5 * t_star, 2000)
soc = 1.0 + mu * q_exp * (q_exp - 1) * t ** (q_exp - 2)

ax.plot(t, soc, 'b-', linewidth=2)
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax.axvline(x=t_star, color='r', linestyle='--', linewidth=1.5, label=f't* = {t_star:.2e}')
ax.fill_between(t, soc, 0, where=(soc < 0), color='red', alpha=0.3, label='SOC < 0 (local MAX)')
ax.fill_between(t, soc, 0, where=(soc > 0), color='green', alpha=0.3, label='SOC > 0 (local MIN)')

ax.set_xlabel('t = |u_i|  (outer weight magnitude)')
ax.set_ylabel('SOC = 1 + mu*q*(q-1)*t^{q-2}')
ax.set_title('2nd Order Condition: which branch is the weight on?')
ax.legend(fontsize=9)
ax.set_ylim(-3, 2)
ax.grid(True, alpha=0.3)

# ──────────────────────────────────────────────────────────
# Panel 3: The proximal objective for several values of v
# Shows the two critical points (local max and local min)
# ──────────────────────────────────────────────────────────
ax = axes[1, 0]
t_plot = np.linspace(1e-9, 6 * t_star, 2000)

for v_val, color, label in [
    (0.5 * v_thresh, 'blue', f'v = 0.5*v_thresh (no root, prox=0)'),
    (v_thresh, 'orange', f'v = v_thresh (one root at t*)'),
    (2.5 * v_thresh, 'green', f'v = 2.5*v_thresh (two roots)'),
    (5.0 * v_thresh, 'red', f'v = 5*v_thresh (two roots)'),
]:
    obj = mu * t_plot ** q_exp + 0.5 * (t_plot - v_val) ** 2
    # Normalize for display: subtract minimum
    obj_shifted = obj - obj.min()
    ax.plot(t_plot, obj_shifted, color=color, linewidth=2, label=label)

ax.set_xlabel('t')
ax.set_ylabel('Proximal objective (shifted)')
ax.set_title('Proximal objective: mu*t^q + (1/2)(t-v)^2')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ──────────────────────────────────────────────────────────
# Panel 4: The proximal operator prox(v) as a function of v
# Shows the JUMP DISCONTINUITY at v = v_thresh
# ──────────────────────────────────────────────────────────
ax = axes[1, 1]
v_range = np.linspace(0, 8 * v_thresh, 1000)
prox_vals = np.zeros_like(v_range)

for i, v in enumerate(v_range):
    if v <= v_thresh:
        prox_vals[i] = 0.0
    else:
        # Newton iteration to find t_large (starting from t = v)
        t = v
        for _ in range(50):
            t = max(t, 1e-30)
            h = t + mu * q_exp * t ** (q_exp - 1) - v
            hp = 1.0 + mu * q_exp * (q_exp - 1) * t ** (q_exp - 2)
            t_new = t - h / hp
            t = max(t_new, 0.0)
        prox_vals[i] = t

# Also plot what happens on the t_small branch (for illustration)
t_small_branch = np.zeros_like(v_range)
for i, v in enumerate(v_range):
    if v <= v_thresh:
        t_small_branch[i] = np.nan
    else:
        # Newton iteration starting from SMALL t (near 0)
        t = 1e-10
        for _ in range(100):
            t = max(t, 1e-30)
            h = t + mu * q_exp * t ** (q_exp - 1) - v
            hp = 1.0 + mu * q_exp * (q_exp - 1) * t ** (q_exp - 2)
            if abs(hp) < 1e-30:
                break
            t_new = t - h / hp
            t = max(t_new, 1e-30)
        # Verify this is actually on the small branch
        soc_val = 1.0 + mu * q_exp * (q_exp - 1) * t ** (q_exp - 2)
        if soc_val < 0:  # local max branch
            t_small_branch[i] = t
        else:
            t_small_branch[i] = np.nan

ax.plot(v_range, prox_vals, 'g-', linewidth=2.5, label='prox(v) = t_large (VALID, what _compute_prox returns)')
ax.plot(v_range, t_small_branch, 'r--', linewidth=2, label='t_small branch (INVALID, local max)')
ax.plot(v_range, v_range, 'k:', linewidth=1, alpha=0.3, label='identity (prox if mu=0)')
ax.axvline(x=v_thresh, color='orange', linestyle='--', linewidth=1.5, label=f'v_thresh = {v_thresh:.2e}')

# Mark the jump
ax.annotate('JUMP\ndiscontinuity',
            xy=(v_thresh, 0), xytext=(v_thresh * 2.5, t_star * 1.5),
            fontsize=10, color='orange', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='orange', lw=2))

# Mark example: a small outer weight u_i on the wrong branch
u_small = t_star * 0.1  # a weight in the dead zone
v_for_u_small = u_small + mu * q_exp * u_small ** (q_exp - 1)
ax.plot(v_for_u_small, u_small, 'rx', markersize=15, markeredgewidth=3, zorder=5,
        label=f'SSN places u={u_small:.1e} HERE (wrong branch!)')

# What prox actually returns for that same v
prox_of_v = prox_vals[np.argmin(np.abs(v_range - v_for_u_small))]
ax.plot(v_for_u_small, prox_of_v, 'g^', markersize=12, zorder=5,
        label=f'But prox(q_var) returns {prox_of_v:.1e} (right branch)')

ax.set_xlabel('v (= q_var, the proximal preimage)')
ax.set_ylabel('prox(v)')
ax.set_title('Proximal operator: jump discontinuity at v_thresh')
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, alpha=0.3)

plt.suptitle(f'Proximal Dead Zone for mu*|.|^q,  q={q_exp:.3f}, mu={mu:.1e}\n'
             f'(power={power}, alpha={alpha}, gamma={gamma})',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/ruizhechao/Documents/NNforHJB/scripts/proximal_deadzone.png', dpi=150, bbox_inches='tight')
print("Figure saved to scripts/proximal_deadzone.png")

print("\n" + "="*70)
print("SUMMARY OF THE BUG:")
print("="*70)
print(f"""
SSN's _initialize_q computes q_var such that the FOC holds:
    |u_i| + mu*q*|u_i|^{{q-1}} = q_var_i

Then later, _compute_prox(q_var) is evaluated, which ALWAYS returns
the t_large root (the actual proximal = global minimizer).

For |u_i| > t* = {t_star:.2e}:
  |u_i| IS the t_large root => prox(q_var) = |u_i|  (CONSISTENT)

For |u_i| < t* = {t_star:.2e}:
  |u_i| is the t_small root (local MAX, SOC < 0)
  => prox(q_var) = t_large >> |u_i|  (INCONSISTENT!)
  => Newton direction is corrupted => line search fails

The 2nd order check:  SOC = 1 + mu*q*(q-1)*|u_i|^{{q-2}}
  SOC > 0  =>  weight is on valid branch (local min)
  SOC <= 0 =>  weight is on invalid branch (local max) => ZERO IT OUT
""")
