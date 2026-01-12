import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def vdp(u_func, ini, t, max_iter=50, tol=1e-6):
    """
    Solve the Van der Pol boundary value problem with terminal conditions p(T) = 0
    using a direct implementation of the Crank-Nicolson method.
    
    This implementation uses a shooting approach with multiple iteration sweeps
    to satisfy both initial and terminal conditions.
    
    Equations:
    ẏ₁ = y₂
    ẏ₂ = -y₁ + y₂(1 - y₁²) + u(t)
    
    ṗ₁ = -p₂ - 2*y₁
    ṗ₂ = 2*y₁*y₂*p₁ + y₁²*p₂ + p₁ - p₂ - 2*y₂
    
    Boundary conditions:
    y₁(0), y₂(0) = ini
    p₁(T) = 0.0, p₂(T) = 0.0
    
    Args:
        u_func: Function that takes time t and returns the control value u(t)
        ini: Initial conditions (y₁(0), y₂(0))
        t_span: Time span (t0, tf)
        num_steps: Number of time steps
        max_iter: Maximum number of iterations for convergence in fsolve
        tol: Tolerance for convergence in fsolve
        
    Returns:
        t: Time points
        y: Solution array with shape (4, num_steps+1)
    """
    # Setup time grid
    t0, tf = t[0], t[-1]
    num_steps = len(t) - 1
    dt = (tf - t0) / num_steps
    
    # Initialize solution array
    y = np.zeros((4, num_steps + 1))
    
    # Set initial conditions for state variables
    y[0, 0], y[1, 0] = ini
    
    # Initial guess for costate variables at t=0
    p1_init = 0.1
    p2_init = 0.1
    
    # Shooting method for finding correct initial values
    def shooting_error(init_values):
        p1_0, p2_0 = init_values
        
        # Set initial costate variables
        y[2, 0] = p1_0
        y[3, 0] = p2_0
        
        # Forward integration using Crank-Nicolson
        for i in range(num_steps):
            # Current values
            y1_curr = y[0, i]
            y2_curr = y[1, i]
            p1_curr = y[2, i]
            p2_curr = y[3, i]
            t_curr = t[i]
            
            # Get current control value
            u_curr = u_func(t_curr)
            
            # Initial guess for next values (Forward Euler)
            f1_curr = y2_curr
            f2_curr = -y1_curr + y2_curr * (1 - y1_curr**2) + u_curr
            f3_curr = -p2_curr - 2*y1_curr
            f4_curr = 2*y1_curr*y2_curr*p1_curr + y1_curr**2*p2_curr + p1_curr - p2_curr - 2*y2_curr
            
            y1_next_guess = y1_curr + dt * f1_curr
            y2_next_guess = y2_curr + dt * f2_curr
            p1_next_guess = p1_curr + dt * f3_curr
            p2_next_guess = p2_curr + dt * f4_curr
            
            # Solve the Crank-Nicolson system for the next step
            def cn_system(vars_next):
                y1_next, y2_next, p1_next, p2_next = vars_next
                t_next = t[i+1]
                u_next = u_func(t_next)
                
                # Compute right-hand side values at the next step
                f1_next = y2_next
                f2_next = -y1_next + y2_next * (1 - y1_next**2) + u_next
                f3_next = -p2_next - 2*y1_next
                f4_next = 2*y1_next*y2_next*p1_next + y1_next**2*p2_next + p1_next - p2_next - 2*y2_next
                
                # Crank-Nicolson equations
                r1 = y1_next - y1_curr - 0.5 * dt * (f1_curr + f1_next)
                r2 = y2_next - y2_curr - 0.5 * dt * (f2_curr + f2_next)
                r3 = p1_next - p1_curr - 0.5 * dt * (f3_curr + f3_next)
                r4 = p2_next - p2_curr - 0.5 * dt * (f4_curr + f4_next)
                
                return [r1, r2, r3, r4]
            
            # Solve for next values using fsolve with specified tolerance and max iterations
            initial_guess = [y1_next_guess, y2_next_guess, p1_next_guess, p2_next_guess]
            next_values = fsolve(cn_system, initial_guess, 
                                xtol=tol,      # Relative error in solution acceptable for convergence
                                maxfev=max_iter)  # Maximum number of function evaluations
            
            # Update the solution
            y[0, i+1] = next_values[0]
            y[1, i+1] = next_values[1]
            y[2, i+1] = next_values[2]
            y[3, i+1] = next_values[3]
        
        # Return terminal error in costate variables (should be zero)
        return [y[2, -1], y[3, -1]]
    
    # Setup options for fsolve to control convergence
    from scipy.optimize import OptimizeResult
    
    # Use root finding to find the correct initial costate values
    # Apply tol and max_iter to fsolve
    result = fsolve(shooting_error, [p1_init, p2_init], 
                   xtol=tol,       # Relative error in solution acceptable for convergence
                   maxfev=max_iter*10,  # Maximum number of function evaluations (use more for outer loop)
                   full_output=True)
    
    # Extract solution and info
    p_initial = result[0]
    info = result[1]
    
    # Run the final integration with the correct initial values
    final_error = shooting_error(p_initial)
    print(f"Initial costate values: p₁(0) = {p_initial[0]:.6f}, p₂(0) = {p_initial[1]:.6f}")
    print(f"Terminal error: |p₁(T)| = {abs(final_error[0]):.6e}, |p₂(T)| = {abs(final_error[1]):.6e}")
    
    # Check the structure of info dictionary to safely access its fields
    if isinstance(info, dict):
        if 'nfev' in info:
            print(f"fsolve iterations: {info['nfev']}", end="")
            if 'mesg' in info:
                print(f", status: {info['mesg']}")
            else:
                print("")
    
    u_values = np.array([u_func(ti) for ti in t])
    
    return t, y

def plot_solution(t, y, u):
    """Plot the BVP solution."""
    fig, ax = plt.subplots(3, 1, figsize=(12, 14))
    
    # Plot state variables (y1, y2)
    ax[0].plot(t, y[0], 'b-', label='$y_1(t)$')
    ax[0].plot(t, y[1], 'r-', label='$y_2(t)$')
    ax[0].set_title(f'Van der Pol BVP with control u(t) (Crank-Nicolson Method)')
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
    
    # Plot control
    ax[2].plot(t, u, 'k-', label='u(t)')
    ax[2].set_title('Control Input')
    ax[2].set_xlabel('Time')
    ax[2].set_ylabel('Control')
    ax[2].grid(True)
    ax[2].legend()
    
    plt.tight_layout()
    plt.savefig(f'vdp_bvp_cn_u_{u[-1]}.png')
    plt.close()

def verify_equations(t, y, u_value):
    """Verify that the solution satisfies the differential equations."""
    y1, y2, p1, p2 = y
    
    # Compute derivatives using finite differences
    dt = t[1] - t[0]
    dy1_dt = np.gradient(y1, dt)
    dy2_dt = np.gradient(y2, dt)
    dp1_dt = np.gradient(p1, dt)
    dp2_dt = np.gradient(p2, dt)
    
    # Compute right-hand side of equations (correct VDP system)
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
    plt.savefig(f'vdp_bvp_cn_error_verification_u_{u_value}.png')
    plt.close()

def solve_for_multiple_controls():
    """Solve the BVP for multiple control values."""
    # Different control functions
    control_functions = [
        (lambda t: np.sin(t), "sinusoidal"),
        (lambda t: np.exp(-t), "exponential_decay")
    ]
    
    for u_func, name in control_functions:
        print(f"\n=== Solving BVP with control u = {name} (Crank-Nicolson) ===")
        
        try:
            # Solve the BVP using Crank-Nicolson
            t, y = vdp_bvp_crank_nicolson(u_func=u_func, ini=[1, 1], t=np.linspace(0, 3, 1001))
            
            # Plot solution
            plot_solution(t, y, u)
            
            # Verify the equations
            verify_equations(t, y, u[-1])
            
            # Save solution to CSV
            data = np.vstack((t, y, u)).T
            header = 't,y1,y2,p1,p2,u'
            filename = f'vdp_bvp_cn_u_{name}.csv'
            np.savetxt(filename, data, delimiter=',', header=header)
            print(f"Solution saved to {filename}")
            
        except Exception as e:
            print(f"Error solving BVP with u={name}: {e}")

if __name__ == "__main__":
    # Solve for standard control values
    solve_for_multiple_controls() 