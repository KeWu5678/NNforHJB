import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# Define the grid and parameter u(t)
t = np.linspace(0, 1, 100)  # Grid points
u_values = np.sin(np.pi * t)  # u(t) = sin(pi * t)

# Define the ODE system, with u passed as an argument
def BVP_ODE(t, y, u):
    y1, y2, p1, p2 = y  # Unpack the variables
    dydt = np.zeros_like(y)
    dydt[0] = y2
    dydt[1] = -y1 + u(t)  # Use u(t) as a function
    dydt[2] = -p2
    dydt[3] = -p1
    return dydt

# Define the boundary conditions
def bc(ya, yb):
    return np.array([
        ya[0],       # y1(0) = 0
        yb[0] - 1,   # y1(1) = 1
        yb[2],       # p1(1) = 0
        ya[3]        # p2(0) = 0
    ])

# Interpolate u(t) for use in the ODE system
from scipy.interpolate import interp1d
u_interp = interp1d(t, u_values, kind='cubic', fill_value="extrapolate")

# Initial guess for the solution
guess = np.zeros((4, t.size))  # 4 variables (y1, y2, p1, p2)

# Solve the BVP, passing u_interp as an argument to BVP_ODE
sol = solve_bvp(lambda t, y: BVP_ODE(t, y, u_interp), bc, t, guess, tol=1e-6)

# Check if the solver was successful
if sol.success:
    print("Solver converged successfully.")
else:
    print("Solver failed to converge.")

# Plot the solution
plt.figure(figsize=(10, 6))
plt.plot(sol.x, sol.y[0], label="y1 (state)")
plt.plot(sol.x, sol.y[1], label="y2 (state)")
plt.plot(sol.x, sol.y[2], label="p1 (costate)")
plt.plot(sol.x, sol.y[3], label="p2 (costate)")
plt.xlabel("t")
plt.ylabel("Solution")
plt.title("Solution of the TPBVP with u(t) = sin(pi t)")
plt.legend()
plt.grid(True)
plt.show()
