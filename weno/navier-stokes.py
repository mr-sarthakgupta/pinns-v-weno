import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.sparse import diags, lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d

class NavierStokesWENOSolver:
    def __init__(self, nx=100, ny=100, Re=100, max_iter=10000, tol=1e-6):
        """
        Initialize the solver for incompressible steady Navier-Stokes equations.
        
        Parameters:
        -----------
        nx, ny : int
            Number of grid points in x and y directions
        Re : float
            Reynolds number
        max_iter : int
            Maximum number of iterations for the solver
        tol : float
            Tolerance for convergence
        """
        self.nx = nx
        self.ny = ny
        self.Re = Re
        self.max_iter = max_iter
        self.tol = tol
        
        # Domain size [0,1]x[0,1]
        self.x = np.linspace(0, 1, nx)
        self.y = np.linspace(0, 1, ny)
        self.dx = 1.0 / (nx - 1)
        self.dy = 1.0 / (ny - 1)
        
        # Initialize velocity and pressure fields
        self.u = np.zeros((nx, ny))  # x-velocity
        self.v = np.zeros((nx, ny))  # y-velocity
        self.p = np.zeros((nx, ny))  # pressure
        
        # Boundary conditions for lid-driven cavity flow
        self.u[:, -1] = 1.0  # Top lid moves with u=1
        
        # For error tracking
        self.divergence_history = []
        
    def weno5_flux(self, v_m2, v_m1, v_p0, v_p1, v_p2):
        """
        Compute WENO5 reconstruction for fluxes.
        """
        # Smooth indicators
        beta0 = 13/12 * (v_m2 - 2*v_m1 + v_p0)**2 + 1/4 * (v_m2 - 4*v_m1 + 3*v_p0)**2
        beta1 = 13/12 * (v_m1 - 2*v_p0 + v_p1)**2 + 1/4 * (v_m1 - v_p1)**2
        beta2 = 13/12 * (v_p0 - 2*v_p1 + v_p2)**2 + 1/4 * (3*v_p0 - 4*v_p1 + v_p2)**2
        
        # Avoid division by zero
        epsilon = 1e-6
        
        # Linear weights
        d0, d1, d2 = 0.1, 0.6, 0.3
        
        # Nonlinear weights
        alpha0 = d0 / (epsilon + beta0)**2
        alpha1 = d1 / (epsilon + beta1)**2
        alpha2 = d2 / (epsilon + beta2)**2
        alpha_sum = alpha0 + alpha1 + alpha2
        
        omega0 = alpha0 / alpha_sum
        omega1 = alpha1 / alpha_sum
        omega2 = alpha2 / alpha_sum
        
        # WENO reconstructions for each stencil
        p0 = (1/3) * v_m2 - (7/6) * v_m1 + (11/6) * v_p0
        p1 = -(1/6) * v_m1 + (5/6) * v_p0 + (1/3) * v_p1
        p2 = (1/3) * v_p0 + (5/6) * v_p1 - (1/6) * v_p2
        
        # Final WENO reconstruction
        return omega0 * p0 + omega1 * p1 + omega2 * p2
    
    def apply_weno_to_field(self, field, direction='x'):
        """
        Apply WENO scheme to a field in a specified direction.
        
        Parameters:
        -----------
        field : ndarray
            2D array representing the field
        direction : str
            Direction for differentiation ('x' or 'y')
        
        Returns:
        --------
        ndarray
            Field after WENO reconstruction
        """
        result = np.zeros_like(field)
        
        if direction == 'x':
            for j in range(self.ny):
                for i in range(2, self.nx-2):
                    result[i, j] = self.weno5_flux(field[i-2, j], field[i-1, j], field[i, j],
                                                  field[i+1, j], field[i+2, j])
                    
                # Special treatment for boundaries
                result[0, j] = field[0, j]
                result[1, j] = field[1, j]
                result[-2, j] = field[-2, j]
                result[-1, j] = field[-1, j]
                
        elif direction == 'y':
            for i in range(self.nx):
                for j in range(2, self.ny-2):
                    result[i, j] = self.weno5_flux(field[i, j-2], field[i, j-1], field[i, j],
                                                  field[i, j+1], field[i, j+2])
                    
                # Special treatment for boundaries
                result[i, 0] = field[i, 0]
                result[i, 1] = field[i, 1]
                result[i, -2] = field[i, -2]
                result[i, -1] = field[i, -1]
                
        return result
    
    def solve_pressure_poisson(self):
        """
        Solve the pressure Poisson equation using a sparse matrix approach.
        """
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        
        # Compute divergence of the velocity field
        div = np.zeros((nx, ny))
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                div[i, j] = (self.u[i+1, j] - self.u[i-1, j]) / (2*dx) + \
                            (self.v[i, j+1] - self.v[i, j-1]) / (2*dy)
        
        # Track divergence as an error metric
        self.divergence_history.append(np.linalg.norm(div))
        
        # Set up the sparse matrix for the Poisson equation
        n = nx * ny
        A = lil_matrix((n, n))
        b = np.zeros(n)
        
        # Interior points
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                idx = i + j * nx
                
                A[idx, idx] = -2/dx**2 - 2/dy**2
                A[idx, idx+1] = 1/dx**2
                A[idx, idx-1] = 1/dx**2
                A[idx, idx+nx] = 1/dy**2
                A[idx, idx-nx] = 1/dy**2
                
                b[idx] = div[i, j] / (dx * dy)
        
        # Boundary conditions (Neumann)
        for i in range(nx):
            # Bottom boundary
            j = 0
            idx = i + j * nx
            A[idx, idx] = 1
            A[idx, idx+nx] = -1
            
            # Top boundary
            j = ny-1
            idx = i + j * nx
            A[idx, idx] = 1
            A[idx, idx-nx] = -1
            
        for j in range(ny):
            # Left boundary
            i = 0
            idx = i + j * nx
            A[idx, idx] = 1
            A[idx, idx+1] = -1
            
            # Right boundary
            i = nx-1
            idx = i + j * nx
            A[idx, idx] = 1
            A[idx, idx-1] = -1
        
        # Fix pressure at one point to remove the null space
        A[0, 0] = 1
        b[0] = 0
        
        # Convert to CSC format for efficient solving
        A = csc_matrix(A)
        
        # Solve the linear system
        p_flat = spsolve(A, b)
        
        # Reshape the solution
        self.p = p_flat.reshape((ny, nx)).T
        
    def solve(self):
        """
        Solve the incompressible steady Navier-Stokes equations using projection method with WENO scheme.
        """
        dx, dy = self.dx, self.dy
        dt = 0.005 * min(dx, dy)**2 * self.Re  # CFL condition
        
        # Arrays to store intermediate velocity fields
        u_star = np.zeros_like(self.u)
        v_star = np.zeros_like(self.v)
        
        # Convergence tracking
        residual_history = []
        
        for iter in range(self.max_iter):
            # Store previous velocities for convergence check
            u_prev = self.u.copy()
            v_prev = self.v.copy()
            
            # Step 1: Apply WENO scheme for advection terms
            u_weno_x = self.apply_weno_to_field(self.u, 'x')
            u_weno_y = self.apply_weno_to_field(self.u, 'y')
            v_weno_x = self.apply_weno_to_field(self.v, 'x')
            v_weno_y = self.apply_weno_to_field(self.v, 'y')
            
            # Step 2: Compute intermediate velocity field (explicit time stepping)
            for i in range(1, self.nx-1):
                for j in range(1, self.ny-1):
                    # Convective terms with WENO scheme
                    conv_u_x = self.u[i, j] * (u_weno_x[i+1, j] - u_weno_x[i-1, j]) / (2*dx)
                    conv_u_y = self.v[i, j] * (u_weno_y[i, j+1] - u_weno_y[i, j-1]) / (2*dy)
                    
                    conv_v_x = self.u[i, j] * (v_weno_x[i+1, j] - v_weno_x[i-1, j]) / (2*dx)
                    conv_v_y = self.v[i, j] * (v_weno_y[i, j+1] - v_weno_y[i, j-1]) / (2*dy)
                    
                    # Diffusive terms (central difference)
                    diff_u_x = (self.u[i+1, j] - 2*self.u[i, j] + self.u[i-1, j]) / dx**2
                    diff_u_y = (self.u[i, j+1] - 2*self.u[i, j] + self.u[i, j-1]) / dy**2
                    
                    diff_v_x = (self.v[i+1, j] - 2*self.v[i, j] + self.v[i-1, j]) / dx**2
                    diff_v_y = (self.v[i, j+1] - 2*self.v[i, j] + self.v[i, j-1]) / dy**2
                    
                    # Pressure gradient terms
                    press_grad_x = (self.p[i+1, j] - self.p[i-1, j]) / (2*dx)
                    press_grad_y = (self.p[i, j+1] - self.p[i, j-1]) / (2*dy)
                    
                    # Intermediate velocity fields
                    u_star[i, j] = self.u[i, j] + dt * (
                        -conv_u_x - conv_u_y - press_grad_x + (1/self.Re) * (diff_u_x + diff_u_y)
                    )
                    
                    v_star[i, j] = self.v[i, j] + dt * (
                        -conv_v_x - conv_v_y - press_grad_y + (1/self.Re) * (diff_v_x + diff_v_y)
                    )
            
            # Apply boundary conditions to intermediate velocity
            # Lid-driven cavity: u=1 at top, u=v=0 elsewhere on boundary
            u_star[0, :] = 0
            u_star[-1, :] = 0
            u_star[:, 0] = 0
            u_star[:, -1] = 1  # Moving lid
            
            v_star[0, :] = 0
            v_star[-1, :] = 0
            v_star[:, 0] = 0
            v_star[:, -1] = 0
            
            # Step 3: Solve pressure Poisson equation
            self.u = u_star.copy()
            self.v = v_star.copy()
            self.solve_pressure_poisson()
            
            # Step 4: Project velocity field to make it divergence-free
            for i in range(1, self.nx-1):
                for j in range(1, self.ny-1):
                    self.u[i, j] = u_star[i, j] - dt * (self.p[i+1, j] - self.p[i-1, j]) / (2*dx)
                    self.v[i, j] = v_star[i, j] - dt * (self.p[i, j+1] - self.p[i, j-1]) / (2*dy)
            
            # Reapply boundary conditions
            self.u[0, :] = 0
            self.u[-1, :] = 0
            self.u[:, 0] = 0
            self.u[:, -1] = 1  # Moving lid
            
            self.v[0, :] = 0
            self.v[-1, :] = 0
            self.v[:, 0] = 0
            self.v[:, -1] = 0
            
            # Check convergence
            u_diff = np.linalg.norm(self.u - u_prev)
            v_diff = np.linalg.norm(self.v - v_prev)
            residual = u_diff + v_diff
            residual_history.append(residual)
            
            if iter % 100 == 0:
                print(f"Iteration {iter}, Residual: {residual:.8f}")
            
            if residual < self.tol:
                print(f"Converged after {iter} iterations with residual {residual:.8f}")
                break
        
        return residual_history
    
    def calculate_divergence(self):
        """
        Calculate the divergence of the velocity field (should be close to zero for incompressible flow).
        """
        dx, dy = self.dx, self.dy
        div = np.zeros((self.nx, self.ny))
        
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                div[i, j] = (self.u[i+1, j] - self.u[i-1, j]) / (2*dx) + \
                            (self.v[i, j+1] - self.v[i, j-1]) / (2*dy)
        
        return div
    
    def compare_with_benchmark(self):
        """
        Compare with benchmark solution from Ghia et al. (1982) for lid-driven cavity.
        Returns the error metrics.
        """
        # Benchmark data for Re=100 from Ghia et al.
        # u-velocity along vertical centerline
        ghia_y = np.array([1.0, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 
                          0.5, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0])
        ghia_u = np.array([1.0, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, 
                          -0.13641, -0.20581, -0.21090, -0.15662, -0.10150, -0.06434, 
                          -0.04775, -0.04192, 0.0])
        
        # v-velocity along horizontal centerline
        ghia_x = np.array([0.0, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5, 
                          0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0])
        ghia_v = np.array([0.0, 0.09233, 0.10091, 0.16077, 0.17507, 0.17527, 0.05454, 
                          0.0, -0.08906, -0.24427, -0.34323, -0.41933, -0.42901, 
                          -0.43643, -0.42901, 0.0])
        
        # Interpolate our solution to match benchmark positions
        # u-velocity along vertical centerline (x = 0.5)
        mid_x = self.nx // 2
        u_center = self.u[mid_x, :]
        u_interp = interp1d(self.y, u_center, kind='cubic')
        u_benchmark = u_interp(ghia_y)
        
        # v-velocity along horizontal centerline (y = 0.5)
        mid_y = self.ny // 2
        v_center = self.v[:, mid_y]
        v_interp = interp1d(self.x, v_center, kind='cubic')
        v_benchmark = v_interp(ghia_x)
        
        # Calculate errors
        u_error = np.linalg.norm(u_benchmark - ghia_u) / np.linalg.norm(ghia_u)
        v_error = np.linalg.norm(v_benchmark - ghia_v) / np.linalg.norm(ghia_v)
        
        return {
            'ghia_y': ghia_y, 
            'ghia_u': ghia_u, 
            'u_benchmark': u_benchmark,
            'ghia_x': ghia_x, 
            'ghia_v': ghia_v, 
            'v_benchmark': v_benchmark,
            'u_error': u_error,
            'v_error': v_error
        }
    
    def plot_results(self):
        """
        Plot the results: velocity field, pressure, and streamlines.
        """
        X, Y = np.meshgrid(self.x, self.y)
        
        # Create figure with 2x2 subplots
        fig = plt.figure(figsize=(15, 12))
        
        # Plot u-velocity
        ax1 = fig.add_subplot(221)
        cf1 = ax1.contourf(X, Y, self.u.T, cmap=cm.viridis, levels=50)
        plt.colorbar(cf1, ax=ax1)
        ax1.set_title('u-velocity')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        
        # Plot v-velocity
        ax2 = fig.add_subplot(222)
        cf2 = ax2.contourf(X, Y, self.v.T, cmap=cm.viridis, levels=50)
        plt.colorbar(cf2, ax=ax2)
        ax2.set_title('v-velocity')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        
        # Plot pressure
        ax3 = fig.add_subplot(223)
        cf3 = ax3.contourf(X, Y, self.p.T, cmap=cm.viridis, levels=50)
        plt.colorbar(cf3, ax=ax3)
        ax3.set_title('Pressure')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        
        # Plot streamlines
        ax4 = fig.add_subplot(224)
        speed = np.sqrt(self.u.T**2 + self.v.T**2)
        lw = 5 * speed / speed.max()
        
        strm = ax4.streamplot(X, Y, self.u.T, self.v.T, density=1, color=speed, 
                             linewidth=lw, cmap=cm.viridis)
        plt.colorbar(strm.lines, ax=ax4)
        ax4.set_title('Streamlines and Speed')
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        
        plt.tight_layout()
        plt.show()
        # Save the plot
        plt.savefig('navier_stokes_results/results_plot.png')
    
    def plot_benchmark_comparison(self):
        """
        Plot comparison with benchmark solution.
        """
        benchmark_data = self.compare_with_benchmark()
        
        # Create figure for comparison plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot u-velocity along vertical centerline
        ax1.plot(benchmark_data['u_benchmark'], benchmark_data['ghia_y'], 'b-', label='WENO Solution')
        ax1.plot(benchmark_data['ghia_u'], benchmark_data['ghia_y'], 'ro', label='Ghia et al. (1982)')
        ax1.set_xlabel('u-velocity')
        ax1.set_ylabel('y')
        ax1.set_title(f'u-velocity along vertical centerline\nError: {benchmark_data["u_error"]:.4f}')
        ax1.legend()
        ax1.grid(True)
        
        # Plot v-velocity along horizontal centerline
        ax2.plot(benchmark_data['ghia_x'], benchmark_data['v_benchmark'], 'b-', label='WENO Solution')
        ax2.plot(benchmark_data['ghia_x'], benchmark_data['ghia_v'], 'ro', label='Ghia et al. (1982)')
        ax2.set_xlabel('x')
        ax2.set_ylabel('v-velocity')
        ax2.set_title(f'v-velocity along horizontal centerline\nError: {benchmark_data["v_error"]:.4f}')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        # Save the benchmark comparison plot
        plt.savefig('navier_stokes_results/benchmark_comparison_plot.png')
    
    def plot_divergence(self):
        """
        Plot the divergence of the velocity field.
        """
        X, Y = np.meshgrid(self.x, self.y)
        div = self.calculate_divergence()
        
        # Create figure
        plt.figure(figsize=(10, 8))
        cf = plt.contourf(X, Y, div.T, cmap=cm.coolwarm, levels=50)
        plt.colorbar(cf)
        plt.title(f'Divergence of Velocity Field\nMax abs value: {np.max(np.abs(div)):.8f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.show()
        # Save divergence plot
        plt.savefig('navier_stokes_results/divergence_plot.png')
        
        # Plot divergence history if available
        if self.divergence_history:
            plt.figure(figsize=(10, 6))
            plt.semilogy(self.divergence_history)
            plt.xlabel('Iteration')
            plt.ylabel('Divergence Norm')
            plt.title('Divergence History')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            # Save divergence plot
            plt.savefig('navier_stokes_results/divergence_plot.png')


# Example usage for a lid-driven cavity flow simulation with error analysis
if __name__ == "__main__":
    # Create solver with Reynolds number 100
    solver = NavierStokesWENOSolver(nx=50, ny=50, Re=100, max_iter=25000, tol=1e-6)
    
    # Solve the Navier-Stokes equations
    residual_history = solver.solve()
    
    # Plot results
    solver.plot_results()
    
    # Plot convergence history
    plt.figure(figsize=(10, 6))
    plt.semilogy(residual_history)
    plt.xlabel('Iteration')
    plt.ylabel('Residual')
    plt.title('Convergence History')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Save the convergence history plot
    plt.savefig('navier_stokes_results/convergence_history.png')
    
    # Plot divergence of velocity field (error in incompressibility constraint)
    solver.plot_divergence()
    
    # Compare with benchmark solution
    solver.plot_benchmark_comparison()
    
    # Print overall error metrics
    benchmark_data = solver.compare_with_benchmark()
    print(f"u-velocity error compared to benchmark: {benchmark_data['u_error']:.6f}")
    print(f"v-velocity error compared to benchmark: {benchmark_data['v_error']:.6f}")
    
    div = solver.calculate_divergence()
    print(f"Maximum absolute divergence: {np.max(np.abs(div)):.8f}")
    print(f"Mean absolute divergence: {np.mean(np.abs(div)):.8f}")