import os
import numpy as np
from src import utils
from src import openloop_optimizer as op


class DateGenerator:
    """Generator for controlled VDP data and saving to disk."""

    def __init__(
        self,
        beta: float = 0.1,
        dt: float = 1e-4,
        T_final: float = 3.0,
    ) -> None:
        # Parameters
        self.beta = beta
        self.dt = dt
        self.T_final = T_final

    

        # State grid and dataset
        self.x0_values = None

    """=====================THE ODE======================"""

    def VDP(self, t, y, u):
        """Define the Boundary value problem with control parameter u"""
        return np.vstack([
            y[1],
            -y[0] + y[1] * (1 - y[0] ** 2) + u(t),
            -2 * y[0] + y[3] * (2 * y[0] * y[1] + 1),
            -2 * y[1] - y[2] - y[3] * (1 - y[0] ** 2),
        ])

    def gradient(self, u, p):
        """Gradient of the objective with respect to the control."""
        if len(p) != len(u):
            raise ValueError("p and u must have the same length")
        n = len(p)
        grad = np.zeros(n)
        for i in range(n):
            grad[i] = p[i] + 2 * self.beta * u[i]
        return grad

    def V(self, grid, u, y1, y2):
        """Objective functional."""
        return utils.L2(grid, u) * self.beta + 0.5 * (utils.L2(grid, y1) + utils.L2(grid, y2))

    def gen_bc(self, ini_val):
        """Generate boundary conditions (p and y) based on initial values."""

        def bc_func(ya, yb):
            return np.array([
                ya[0] - ini_val[0],
                ya[1] - ini_val[1],
                yb[2],
                yb[3],
            ])

        return bc_func

    def apply_initial_gridding(self, nx1, nx2, x1_range = [-3, 3], x2_range = [-3, 3]):
        """Create state grid, and allocate dataset."""

        # Define ranges and discrete grid in state space
        x1_values = np.linspace(x1_range[0], x1_range[1], nx1)
        x2_values = np.linspace(x2_range[0], x2_range[1], nx2)

        # Create 2D grid and list of all (x1, x2) pairs
        X1, X2 = np.meshgrid(x1_values, x2_values)
        self.x0_values = np.column_stack((X1.ravel(), X2.ravel()))

        print(f"Created a {len(x1_values)}x{len(x2_values)} grid with {len(self.x0_values)} points")

    def apply_randoom_initial_sampling(self, N, x1_range = [-3, 3], x2_range = [-3, 3]):
        """Randomly sample N initial points in the specified ranges."""
        
        # Randomly sample N points for x1 and x2 within their respective ranges
        x1_values = np.random.uniform(x1_range[0], x1_range[1], N)
        x2_values = np.random.uniform(x2_range[0], x2_range[1], N)
        
        # Create array of (x1, x2) pairs in the same format as gridding
        self.x0_values = np.column_stack((x1_values, x2_values))
        
        print(f"Randomly sampled {N} points in range x1: {x1_range}, x2: {x2_range}")
        

    def data_generation(self, tol = 1e-5, max_it = 500):
        """
        Run the open-loop optimizer for all initial conditions and fill the dataset.
        Args:
        tol: precision for the optimization w.r.t. the gradient of the open loop
        max_it: max iteration of the optimization w.r.t. the gradient of the open loop
        """
        if self.x0_values is None:
            raise RuntimeError("generate initial value before data_generation().")
        failed_ini = []

        # time diecretization
        grid = np.arange(0.0, self.T_final + self.dt, self.dt)
        guess = np.ones((4, grid.size))

        N = self.x0_values.shape[0]
        dtype = [('x', '2float64'), ('dv', '2float64'), ('v', 'float64')]
        dataset = np.zeros(N, dtype=dtype)

        print(f"N = {N}")
        for i in range(N):
            # Get initial condition from the grid
            ini = self.x0_values[i]

            # Generate boundary conditions
            bc_func = self.gen_bc(ini)

            print(f"i = {i}")
            print(f"ini = {ini}")
            dv, v = op.OpenLoopOptimizer(
                self.VDP,
                bc_func,
                self.V,
                self.gradient,
                grid,
                guess,
                tol,
                max_it,
            ).optimize()

            if dv is not None and not np.isnan(v):
                dataset[i] = (ini, dv, v)
            else:
                # Store failed initial condition
                failed_ini.append(ini)
                print(f"Failed to converge for ini = {ini}")
        
        # Return dataset and failed initial conditions
        return dataset, failed_ini

    def data_save(self, rawdata_subdir: str = "raw_data"):
        """Save dataset and failed initial conditions to .npy files under rawdata."""
        # Base rawdata directory (../rawdata relative to this file)
        base_dir = os.path.join(os.path.dirname(__file__), "..", "rawdata")
        save_dir = os.path.join(base_dir, rawdata_subdir)
        os.makedirs(save_dir, exist_ok=True)

        # Filenames that indicate beta and grid resolution
        output_file = os.path.join(
            save_dir,
            f"VDP_beta_{self.beta}_grid_{self.nx1}x{self.nx2}.npy",
        )
        failed_output_file = os.path.join(
            save_dir,
            f"VDP_beta_{self.beta}_failed_ini_{self.nx1}x{self.nx2}.npy",
        )

        # Save failed initial conditions if any
        if self.failed_ini:
            failed_ini_arr = np.array(self.failed_ini)
            print(f"Saving {len(failed_ini_arr)} failed initial conditions to {failed_output_file}")
            np.save(failed_output_file, failed_ini_arr)
        else:
            print("All initial conditions converged successfully")

        print(f"Saving results to {output_file}")
        np.save(output_file, self.dataset)

