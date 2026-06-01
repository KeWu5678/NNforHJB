from datetime import datetime
from pathlib import Path
import numpy as np
from src import utils
from src.OpenLoop.bvp_bb_optimizer import BvpBbOpenLoopOptimizer
from src.OpenLoop.sample_sets import (
    grid_initial_states,
    random_initial_states,
    save_dataset_bundle,
)


class DataGenerator:
    """Generator for controlled VDP data and saving to disk."""

    def __init__(
        self,
        beta: float = 0.1,
        dt: float = 1e-4,
        T_final: float = 3.0,
    ) -> None:
        self.beta = beta
        self.dt = dt
        self.T_final = T_final
        self.x0_values = None

    # --- ODE ---

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
        return p + 2.0 * self.beta * u

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

        self.x0_values = grid_initial_states(nx1, nx2, x1_range, x2_range)

        print(f"Created a {nx1}x{nx2} grid with {len(self.x0_values)} points")

    def apply_randoom_initial_sampling(self, N, x1_range = [-3, 3], x2_range = [-3, 3]):
        """Randomly sample N initial points in the specified ranges."""
        
        self.x0_values = random_initial_states(N, x1_range, x2_range)
        
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

        grid = np.arange(0.0, self.T_final + self.dt, self.dt)
        guess = np.ones((4, grid.size))
        dtype = [('x', '2float64'), ('dv', '2float64'), ('v', 'float64')]
        rows = []

        print(f"N = {self.x0_values.shape[0]}")
        for i, ini in enumerate(self.x0_values):
            bc_func = self.gen_bc(ini)
            print(f"i = {i}")
            print(f"ini = {ini}")
            result = BvpBbOpenLoopOptimizer(
                self.VDP,
                bc_func,
                self.V,
                self.gradient,
                grid,
                guess,
                tol,
                max_it,
            ).optimize_result()

            if (
                result.converged
                and result.gradient is not None
                and result.value is not None
                and not np.isnan(result.value)
            ):
                rows.append((ini, result.gradient, result.value))
            else:
                failed_ini.append(ini)
                print(f"Failed to converge for ini = {ini}")

        dataset = np.array(rows, dtype=dtype) if rows else np.zeros(0, dtype=dtype)
        return dataset, failed_ini

    def data_save(
        self,
        dataset,
        failed_ini,
        output_dir: str | Path,
        tag: str | None = None,
    ) -> tuple[Path, Path | None]:
        """
        Save dataset and failed initial conditions to an explicit directory.

        This function is intentionally stateless: it only saves the arrays that are
        passed in (typically the return values of `data_generation`) and does not
        rely on any other attributes of the class instance.
        """
        save_dir = Path(output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        date_tag = datetime.now().strftime("%Y%m%d") if tag is None else tag

        output_file = save_dir / f"VDP_dataset_{date_tag}.npy"
        failed_output_file = save_dir / f"VDP_failed_ini_{date_tag}.npy"

        arrays = {output_file.name: dataset}
        if failed_ini is not None and len(failed_ini) > 0:
            failed_ini_arr = np.array(failed_ini)
            print(f"Saving {len(failed_ini_arr)} failed initial conditions to {failed_output_file}")
            arrays[failed_output_file.name] = failed_ini_arr
        else:
            print("No failed initial conditions to save")

        print(f"Saving dataset to {output_file}")
        saved = save_dataset_bundle(save_dir, arrays=arrays)
        return saved[output_file.name], saved.get(failed_output_file.name)
