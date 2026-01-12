import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

def vdp_bvp_with_terminal_zeros(u_value=1, t_span=(0, 3), num_points=100):
    """
    Solve the Van der Pol boundary value problem with terminal conditions p(T) = 0.
    
    Equations:
    ẏ₁ = y₂
    ẏ₂ = -y₁ + y₂(1 - y₁²) + u
    
    ṗ₁ = -p₂ - 2*y₁
    ṗ₂ = 2*y₁*y₂*p₁ + y₁²*p₂ + p₁ - p₂ - 2*y₂
    
    Boundary conditions:
    y₁(0) = y₁₀, y₂(0) = y₂₀
    p₁(T) = 0, p₂(T) = 0
    
    Args:
        u_value: Constant control value
        t_span: Time span (t0, tf)
        num_points: Number of points for solution
        
    Returns:
        t: Time points
        y: Solution array with shape (4, num_points)
    """
    # Initial mesh
    t = np.linspace(t_span[0], t_span[1], num_points)
    
    # Define the ODE system
    def fun(x, y):
        # y has shape (4, n_points): [y₁, y₂, p₁, p₂]
        # return array with shape (4, n_points)
        
        y1, y2, p1, p2 = y
        
        # State equations
        dy1_dx = y2
        dy2_dx = -y1 + y2 * (1 - y1**2) + u_value
        
        # Costate equations - correctly derived from test_VDP.py
        dp1_dx = -p2 - 2*y1
        dp2_dx = 2*y1*y2*p1 + y1**2*p2 + p1 - p2 - 2*y2
        
        return np.vstack([dy1_dx, dy2_dx, dp1_dx, dp2_dx])
    
    # Define the boundary conditions
    def bc(ya, yb):
        # ya is at x=0, yb is at x=T
        # For a well-posed problem, we need 4 boundary conditions (system of 4 ODEs)
        
        return np.array([
            ya[0] - 1.0,      # y₁(0) = 1.0
            ya[1] - 0.0,      # y₂(0) = 0.0
            yb[2],            # p₁(T) = 0
            yb[3]             # p₂(T) = 0
        ])
    
    # Initial guess for the solution
    y_guess = np.zeros((4, t.size))
    
    # Better initial guess based on understanding of the Van der Pol dynamics
    # For y1 and y2, use sine and cosine as better guesses
    y_guess[0] = np.cos(np.pi * t / t_span[1])         # y₁ guess: oscillatory, starts at 1.0
    y_guess[1] = -np.pi/t_span[1] * np.sin(np.pi * t / t_span[1])  # y₂ guess: derivative of y₁
    
    # For p1 and p2, use backward-decaying functions (since they must be 0 at T)
    time_reversed = t_span[1] - t
    decay_factor = 3.0
    y_guess[2] = np.exp(-decay_factor * time_reversed)  # p₁ guess: decays to 0 at t=T
    y_guess[3] = -decay_factor * np.exp(-decay_factor * time_reversed)  # p₂ guess: decays to 0 at t=T
    
    # Solve the BVP
    solution = solve_bvp(fun, bc, t, y_guess, tol=1e-3, max_nodes=10000)
    
    if not solution.success:
        print(f"Warning: BVP solver did not converge: {solution.message}")
        # Try with a more refined mesh and different initial guess
        print("Trying again with a refined mesh...")
        t_refined = np.linspace(t_span[0], t_span[1], num_points * 2)
        y_refined = np.zeros((4, t_refined.size))
        # Interpolate the previous guess to the new grid
        for i in range(4):
            y_refined[i] = np.interp(t_refined, t, y_guess[i])
        solution = solve_bvp(fun, bc, t_refined, y_refined, tol=1e-3, max_nodes=20000)
    
    if solution.success:
        print(f"BVP solved successfully: {solution.message}")
        print(f"Number of nodes used: {len(solution.x)}")
        print(f"Number of iterations: {solution.niter}")
    else:
        print(f"Warning: BVP solver still failed to converge: {solution.message}")
    
    # Get solution on a finer grid
    t_fine = np.linspace(t_span[0], t_span[1], num_points * 4)
    y_fine = solution.sol(t_fine)
    
    # Verify terminal conditions
    terminal_p1 = y_fine[2, -1]
    terminal_p2 = y_fine[3, -1]
    print(f"Terminal values: p₁(T) = {terminal_p1:.6e}, p₂(T) = {terminal_p2:.6e}")
    
    return t_fine, y_fine, u_value

def plot_solution(t, y, u):
    """Plot the BVP solution."""
    fig, ax = plt.subplots(3, 1, figsize=(12, 14))
    
    # Plot state variables (y1, y2)
    ax[0].plot(t, y[0], 'b-', label='$y_1(t)$')
    ax[0].plot(t, y[1], 'r-', label='$y_2(t)$')
    ax[0].set_title(f'Van der Pol BVP with constant u={u}')
    ax[0].set_ylabel('State')
    ax[0].grid(True)
    ax[0].legend()
    
    # Add phase portrait as inset
    axins = ax[0].inset_axes([0.65, 0.1, 0.3, 0.3])
    axins.plot(y[0], y[1], 'g-')
    axins.set_xlabel('$y_1$')
    axins.set_ylabel('$y_2$')
    axins.set_title('Phase Portrait')
    axins.grid(True)
    
    # Plot costate variables (p1, p2)
    ax[1].plot(t, y[2], 'g-', label='$p_1(t)$')
    ax[1].plot(t, y[3], 'm-', label='$p_2(t)$')
    ax[1].set_title('Costate Variables (with terminal zeros)')
    ax[1].set_ylabel('Costate')
    ax[1].grid(True)
    ax[1].legend()
    
    # Plot constant control
    ax[2].plot(t, np.full_like(t, u), 'k-', label='u(t)')
    ax[2].set_title('Control Input (constant)')
    ax[2].set_xlabel('Time')
    ax[2].set_ylabel('Control')
    ax[2].grid(True)
    ax[2].legend()
    
    plt.tight_layout()
    plt.savefig(f'vdp_bvp_terminal_zeros_u_{u}.png')
    plt.show()

def verify_equations(t, y, u_value):
    """Verify that the solution satisfies the differential equations."""
    y1, y2, p1, p2 = y
    
    # Compute derivatives using finite differences
    dt = t[1] - t[0]
    dy1_dt = np.gradient(y1, dt)
    dy2_dt = np.gradient(y2, dt)
    dp1_dt = np.gradient(p1, dt)
    dp2_dt = np.gradient(p2, dt)
    
    # Compute right-hand side of equations
    rhs_y1 = y2
    rhs_y2 = -y1 + y2 * (1 - y1**2) + u_value
    rhs_p1 = -p2 - 2*y1
    rhs_p2 = 2*y1*y2*p1 + y1**2*p2 + p1 - p2 - 2*y2
    
    # Compute errors
    error_y1 = np.abs(dy1_dt - rhs_y1)
    error_y2 = np.abs(dy2_dt - rhs_y2)
    error_p1 = np.abs(dp1_dt - rhs_p1)
    error_p2 = np.abs(dp2_dt - rhs_p2)
    
    # Print maximum errors
    print("Maximum errors in differential equations:")
    print(f"y₁ equation: {np.max(error_y1):.6e}")
    print(f"y₂ equation: {np.max(error_y2):.6e}")
    print(f"p₁ equation: {np.max(error_p1):.6e}")
    print(f"p₂ equation: {np.max(error_p2):.6e}")
    
    # Plot errors
    plt.figure(figsize=(12, 8))
    plt.semilogy(t, error_y1, 'b-', label='Error in $y_1$ equation')
    plt.semilogy(t, error_y2, 'r-', label='Error in $y_2$ equation')
    plt.semilogy(t, error_p1, 'g-', label='Error in $p_1$ equation')
    plt.semilogy(t, error_p2, 'm-', label='Error in $p_2$ equation')
    plt.title('Verification of Differential Equations')
    plt.xlabel('Time')
    plt.ylabel('Error (log scale)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'vdp_bvp_error_verification_u_{u_value}.png')
    plt.show()

def solve_for_multiple_controls():
    """Solve the BVP for multiple control values."""
    # Different constant control values
    control_values = [0, 1, 2]
    
    for u_value in control_values:
        print(f"\n=== Solving BVP with constant control u = {u_value} ===")
        
        try:
            # Solve the BVP
            t, y, u = vdp_bvp_with_terminal_zeros(u_value=u_value)
            
            # Plot solution
            plot_solution(t, y, u)
            
            # Verify the equations
            verify_equations(t, y, u)
            
            # Save solution to CSV
            data = np.vstack((t, y, np.full_like(t, u))).T
            header = 't,y1,y2,p1,p2,u'
            filename = f'vdp_bvp_terminal_zeros_u_{u}.csv'
            np.savetxt(filename, data, delimiter=',', header=header)
            print(f"Solution saved to {filename}")
            
        except Exception as e:
            print(f"Error solving BVP with u={u_value}: {e}")

def run_with_varying_parameters():
    """Investigate how changing u affects the solution."""
    # A range of control values
    u_values = np.linspace(0, 3, 7)  # [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    # Record terminal state values
    terminal_y1 = []
    terminal_y2 = []
    
    for u in u_values:
        print(f"Solving for u = {u:.1f}")
        try:
            t, y, _ = vdp_bvp_with_terminal_zeros(u_value=u)
            terminal_y1.append(y[0, -1])
            terminal_y2.append(y[1, -1])
        except Exception as e:
            print(f"Error: {e}")
            terminal_y1.append(np.nan)
            terminal_y2.append(np.nan)
    
    # Plot the effect of control on terminal state
    plt.figure(figsize=(10, 6))
    plt.plot(u_values, terminal_y1, 'bo-', label='Terminal $y_1(T)$')
    plt.plot(u_values, terminal_y2, 'ro-', label='Terminal $y_2(T)$')
    plt.title('Effect of Control Value on Terminal State')
    plt.xlabel('Control Value (u)')
    plt.ylabel('Terminal State')
    plt.grid(True)
    plt.legend()
    plt.savefig('vdp_bvp_control_effect.png')
    plt.show()

if __name__ == "__main__":
    # Solve for standard control values
    solve_for_multiple_controls()
    
    # Uncomment to analyze effect of control value
    # run_with_varying_parameters() 