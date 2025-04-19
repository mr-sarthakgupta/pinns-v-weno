import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
from scipy.interpolate import interp1d

class BuckleyLeverettWENOSolver:
    def __init__(self, nx=200, T=0.5, cfl=0.4, M=2.0):
        """
        Initialize the solver for the Buckley-Leverett equation.
        
        Parameters:
        -----------
        nx : int
            Number of grid points
        T : float
            Final time
        cfl : float
            CFL number for stability
        M : float
            Mobility ratio (oil/water viscosity ratio)
        """
        self.nx = nx
        self.T = T
        self.cfl = cfl
        self.M = M
        
        # Domain [0,1]
        self.x = np.linspace(0, 1, nx)
        self.dx = 1.0 / (nx - 1)
        
        # Initial water saturation: step function
        self.S = np.zeros(nx)
        self.S[self.x < 0.1] = 1.0  # Water saturation = 1 at inlet
        
        # For tracking solutions at different time steps
        self.solution_history = []
        self.time_points = []
        
        # For tracking residuals
        self.residual_history = []
    
    def fractional_flow(self, S):
        """
        Compute the fractional flow function f(S) for the Buckley-Leverett equation.
        
        Parameters:
        -----------
        S : ndarray
            Water saturation
        
        Returns:
        --------
        ndarray
            Fractional flow of water
        """
        # Standard Buckley-Leverett fractional flow function
        # f(S) = S^2 / (S^2 + M*(1-S)^2)
        # where M is the mobility ratio (oil/water viscosity ratio)
        
        # Avoid division by zero
        epsilon = 1e-10
        S_safe = np.clip(S, epsilon, 1.0 - epsilon)
        
        numerator = S_safe**2
        denominator = S_safe**2 + self.M * (1 - S_safe)**2
        
        return numerator / denominator
    
    def weno5_reconstruction(self, v_m2, v_m1, v_p0, v_p1, v_p2):
        """
        WENO5 reconstruction for flux computation.
        
        Parameters:
        -----------
        v_m2, v_m1, v_p0, v_p1, v_p2 : float
            Stencil values
        
        Returns:
        --------
        float
            WENO5 reconstructed value
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
    
    def compute_numerical_flux(self, S):
        """
        Compute the numerical flux using the WENO5 scheme.
        
        Parameters:
        -----------
        S : ndarray
            Water saturation
        
        Returns:
        --------
        ndarray
            Numerical flux at cell interfaces
        """
        nx = self.nx
        flux = np.zeros(nx + 1)
        
        # Extended saturation array with ghost cells
        S_ext = np.zeros(nx + 4)
        S_ext[2:-2] = S
        
        # Ghost cells (transmissive boundary conditions)
        S_ext[0] = S[0]
        S_ext[1] = S[0]
        S_ext[-2] = S[-1]
        S_ext[-1] = S[-1]
        
        # Convert saturation to fractional flow
        f_ext = self.fractional_flow(S_ext)
        
        # Flux at interior interfaces using WENO5
        for i in range(2, nx + 2):
            # For Buckley-Leverett with positive velocity, upwind direction is from left
            # WENO reconstruction from left (for f^+)
            flux[i-2] = self.weno5_reconstruction(
                f_ext[i-2], f_ext[i-1], f_ext[i], f_ext[i+1], f_ext[i+2]
            )
        
        return flux
    
    def calculate_residual(self, S, S_new):
        """
        Calculate residual between consecutive time steps.
        
        Parameters:
        -----------
        S : ndarray
            Current solution
        S_new : ndarray
            New solution
            
        Returns:
        --------
        float
            L2 norm of the difference
        """
        print(f"S_new: {S_new.max():.4f}, {S_new.min():.4f}")
        print(f"S: {S.max():.4f}, {S.min():.4f}")
        return np.sqrt(np.mean((S_new - S)**2))
    
    def solve(self):
        """
        Solve the Buckley-Leverett equation using WENO5 scheme.
        
        Returns:
        --------
        tuple
            Time points, solution history, and residual history
        """
        nx = self.nx
        dx = self.dx
        S = self.S.copy()
        
        # Save initial condition
        self.solution_history.append(S.copy())
        self.time_points.append(0.0)
        
        t = 0.0
        step = 0
        
        while t < self.T:
            # Compute fractional flow at current saturation
            f = self.fractional_flow(S)
            
            # Maximum wave speed for CFL condition
            # df/dS approximated by central differences
            df_dS = np.zeros_like(S)
            df_dS[1:-1] = (f[2:] - f[:-2]) / (2 * dx)
            df_dS[0] = (f[1] - f[0]) / dx
            df_dS[-1] = (f[-1] - f[-2]) / dx
            
            max_speed = np.max(np.abs(df_dS))
            dt = self.cfl * dx / (max_speed + 1e-10)
            
            # Ensure we don't go beyond final time
            if t + dt > self.T:
                dt = self.T - t
            
            # Compute numerical flux
            flux = self.compute_numerical_flux(S)
            
            # Update solution (conservative form)
            S_new = S.copy()
            for i in range(1, nx - 1):
                S_new[i] = S[i] - dt / dx * (flux[i+1] - flux[i])
            
            # Apply boundary conditions
            S_new[0] = 1.0  # Injection of water at left boundary
            S_new[-1] = S_new[-2]  # Transmissive right boundary
            
            # Calculate residual
            residual = self.calculate_residual(S, S_new)
            self.residual_history.append(residual)
            
            # Update time and solution
            t += dt
            S = S_new.copy()
            
            # Save solution at specified intervals
            if step % 10 == 0 or abs(t - self.T) < 1e-10:
                self.solution_history.append(S.copy())
                self.time_points.append(t)
            
            step += 1
            if step % 100 == 0:
                print(f"Time: {t:.6f}, Step: {step}, Residual: {residual:.8f}")
                print(f"Max saturation: {np.max(S):.6f}, Min saturation: {np.min(S):.6f}")
        
        # Final reporting
        print("\nResidual Summary:")
        print(f"Final time: {t:.6f}")
        print(f"Number of time steps: {step}")
        print(f"Initial residual: {self.residual_history[0]:.8f}")
        print(f"Final residual: {self.residual_history[-1]:.8f}")
        print(f"Maximum residual: {np.max(self.residual_history):.8f}")
        print(f"Average residual: {np.mean(self.residual_history):.8f}")
        
        return self.time_points, self.solution_history, self.residual_history
    
    def plot_results(self):
        """
        Plot the results at different time steps.
        """
        times = self.time_points
        solutions = self.solution_history
        
        plt.figure(figsize=(12, 8))
        
        # Plot every nth solution to avoid overcrowding
        n = max(1, len(times) // 5)
        for i in range(0, len(times), n):
            plt.plot(self.x, solutions[i], label=f"t = {times[i]:.3f}")
        
        # Always plot the final solution
        if len(times) % n != 0:
            plt.plot(self.x, solutions[-1], label=f"t = {times[-1]:.3f}")
        
        plt.xlabel("Position (x)")
        plt.ylabel("Water Saturation (S)")
        plt.title("Buckley-Leverett Solution with WENO5 Scheme")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("buckley_leverett_results/buckley_leverett_results.png")
        plt.show()
    
    def plot_residual_history(self):
        """
        Plot the history of residuals.
        """
        plt.figure(figsize=(12, 6))
        
        # Plot in linear scale
        plt.subplot(1, 2, 1)
        plt.plot(self.residual_history)
        plt.xlabel("Time Step")
        plt.ylabel("Residual (L2 norm)")
        plt.title("Residual History")
        plt.grid(True)
        
        # Plot in log scale
        plt.subplot(1, 2, 2)
        plt.semilogy(self.residual_history)
        plt.xlabel("Time Step")
        plt.ylabel("Residual (L2 norm)")
        plt.title("Residual History (Log Scale)")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("buckley_leverett_results/buckley_leverett_residuals.png")
        plt.show()
    
    def exact_solution(self, t):
        """
        Compute the exact solution for the Buckley-Leverett equation with the Welge tangent method.
        This is only valid for a simple step initial condition.
        
        Parameters:
        -----------
        t : float
            Time at which to compute the exact solution
            
        Returns:
        --------
        tuple
            x and S arrays for the exact solution
        """
        # Generate a fine grid of saturation values
        S_values = np.linspace(0, 1, 1000)
        f_values = self.fractional_flow(S_values)
        
        # Calculate df/dS for the wave speed
        df_dS = np.zeros_like(S_values)
        df_dS[1:] = np.diff(f_values) / np.diff(S_values)
        
        # Calculate x positions for each saturation value
        x_values = df_dS * t
        
        # Filter valid positions (0 <= x <= 1)
        valid = (x_values >= 0) & (x_values <= 1)
        x_valid = x_values[valid]
        S_valid = S_values[valid]
        
        # Sort by position
        idx = np.argsort(x_valid)
        x_sorted = x_valid[idx]
        S_sorted = S_valid[idx]
        
        # Find the shock position and height
        # For the simple Buckley-Leverett, the shock connects S=0 to the point where df/dS = f/S
        
        # Simple approximation: find where df/dS crosses f/S
        f_over_S = np.zeros_like(S_values)
        f_over_S[1:] = f_values[1:] / S_values[1:]
        
        # Find crossover point (shock)
        idx_shock = np.argmin(np.abs(df_dS - f_over_S))
        S_shock = S_values[idx_shock]
        
        # Shock speed is df/dS at the shock
        shock_speed = df_dS[idx_shock]
        shock_pos = shock_speed * t
        
        # Add the shock to the solution
        if shock_pos <= 1:
            # Insert the shock
            x_final = np.concatenate([[shock_pos - 1e-6, shock_pos, shock_pos + 1e-6], 
                                      x_sorted[x_sorted > shock_pos]])
            S_final = np.concatenate([[S_sorted[0], 0, 0], 
                                     S_sorted[x_sorted > shock_pos]])
        else:
            x_final = x_sorted
            S_final = S_sorted
        
        return x_final, S_final
    
    def comprehensive_comparison(self):
        """
        Generate comprehensive comparison plots for multiple time steps.
        """
        # Select time points for comparison
        if len(self.time_points) <= 3:
            comparison_times = self.time_points
        else:
            # Choose a few representative times including the final time
            idx = np.linspace(0, len(self.time_points)-1, 4, dtype=int)
            comparison_times = [self.time_points[i] for i in idx]
        
        # Create a multi-panel figure
        fig, axes = plt.subplots(len(comparison_times), 1, figsize=(12, 4*len(comparison_times)))
        if len(comparison_times) == 1:
            axes = [axes]
        
        # Error metrics for each time
        error_metrics = []
        
        for i, t in enumerate(comparison_times):
            # Find closest time in solution history
            idx = np.argmin(np.abs(np.array(self.time_points) - t))
            t_actual = self.time_points[idx]
            numerical_S = self.solution_history[idx]
            
            # Get exact solution
            x_exact, S_exact = self.exact_solution(t_actual)
            
            # Interpolate exact solution to our grid for error calculation
            f_interp = interp1d(x_exact, S_exact, bounds_error=False, fill_value=(S_exact[0], S_exact[-1]))
            exact_S_interp = f_interp(self.x)
            
            # Calculate error
            error = np.abs(numerical_S - exact_S_interp)
            l1_error = np.mean(error)
            l2_error = np.sqrt(np.mean(error**2))
            linf_error = np.max(error)
            
            error_metrics.append({
                "time": t_actual,
                "l1_error": l1_error,
                "l2_error": l2_error,
                "linf_error": linf_error
            })
            
            # Plot on the appropriate axis
            ax = axes[i]
            ax.plot(self.x, numerical_S, 'b-', label="Numerical (WENO5)")
            ax.plot(x_exact, S_exact, 'r--', label="Exact Solution")
            
            # Plot the point-wise error
            ax2 = ax.twinx()
            ax2.plot(self.x, error, 'g-', alpha=0.5, label="Absolute Error")
            ax2.set_ylabel("Absolute Error", color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            
            # Set titles and labels
            ax.set_xlabel("Position (x)")
            ax.set_ylabel("Water Saturation (S)")
            ax.set_title(f"Comparison at t = {t_actual:.3f} (L2 Error: {l2_error:.6f})")
            
            # Create combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig("buckley_leverett_results/buckley_leverett_comprehensive_comparison.png")
        plt.show()
        
        # Print error metrics
        print("\nError Metrics Summary:")
        for metrics in error_metrics:
            print(f"Time: {metrics['time']:.3f}")
            print(f"  L1 Error: {metrics['l1_error']:.6f}")
            print(f"  L2 Error: {metrics['l2_error']:.6f}")
            print(f"  L∞ Error: {metrics['linf_error']:.6f}")
        
        return error_metrics
    
    def create_animation(self):
        """
        Create an animation of the solution evolution with comparison to exact solution.
        
        Returns:
        --------
        animation.FuncAnimation
            Animation object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        line_num, = ax.plot([], [], 'b-', lw=2, label='Numerical')
        line_exact, = ax.plot([], [], 'r--', lw=2, label='Exact')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Position (x)")
        ax.set_ylabel("Water Saturation (S)")
        ax.set_title("Buckley-Leverett Solution Evolution")
        ax.legend(loc='upper right')
        ax.grid(True)
        
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        error_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
        
        def init():
            line_num.set_data([], [])
            line_exact.set_data([], [])
            time_text.set_text('')
            error_text.set_text('')
            return line_num, line_exact, time_text, error_text
        
        def animate(i):
            t = self.time_points[i]
            numerical_S = self.solution_history[i]
            
            # Get exact solution
            x_exact, S_exact = self.exact_solution(t)
            
            # Interpolate exact solution to grid for error calculation
            f_interp = interp1d(x_exact, S_exact, bounds_error=False, fill_value=(S_exact[0], S_exact[-1]))
            exact_S_interp = f_interp(self.x)
            
            # Calculate error
            l2_error = np.sqrt(np.mean((numerical_S - exact_S_interp)**2))
            
            # Update plots
            line_num.set_data(self.x, numerical_S)
            line_exact.set_data(x_exact, S_exact)
            time_text.set_text(f"Time: {t:.3f}")
            error_text.set_text(f"L2 Error: {l2_error:.6f}")
            
            return line_num, line_exact, time_text, error_text
        
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                      frames=len(self.time_points), interval=100, blit=True)
        
        plt.tight_layout()
        return anim


# Example usage with different mobility ratios for comparison
if __name__ == "__main__":
    print("Solving Buckley-Leverett equation with WENO5 scheme...\n")
    
    # Solve with mobility ratio M=2 (standard case)
    print("Case 1: Mobility ratio M=2")
    solver1 = BuckleyLeverettWENOSolver(nx=200, T=0.4, cfl=0.4, M=2.0)
    times1, solutions1, residuals1 = solver1.solve()
    
    # Plot results for case 1
    solver1.plot_results()
    solver1.plot_residual_history()
    metrics1 = solver1.comprehensive_comparison()
    
    # Create animation for case 1
    anim1 = solver1.create_animation()
    # Uncomment to save animation
    # anim1.save('buckley_leverett_animation_M2.mp4', writer='ffmpeg', fps=15)
    
    # Solve with mobility ratio M=5 (higher viscosity contrast)
    print("\nCase 2: Mobility ratio M=5")
    solver2 = BuckleyLeverettWENOSolver(nx=200, T=0.4, cfl=0.4, M=5.0)
    times2, solutions2, residuals2 = solver2.solve()
    
    # Plot results for case 2
    solver2.plot_results()
    solver2.plot_residual_history()
    metrics2 = solver2.comprehensive_comparison()
    
    # Compare residuals between cases
    plt.figure(figsize=(12, 6))
    plt.semilogy(residuals1, 'b-', label='M=2')
    plt.semilogy(residuals2, 'r-', label='M=5')
    plt.xlabel("Time Step")
    plt.ylabel("Residual (L2 norm)")
    plt.title("Residual Comparison for Different Mobility Ratios")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("buckley_leverett_results/buckley_leverett_residual_comparison.png")
    plt.show()
    
    # Compare final solutions
    plt.figure(figsize=(12, 6))
    plt.plot(solver1.x, solver1.solution_history[-1], 'b-', label=f'M=2, t={times1[-1]:.3f}')
    plt.plot(solver2.x, solver2.solution_history[-1], 'r-', label=f'M=5, t={times2[-1]:.3f}')
    
    # Add exact solutions
    x_exact1, S_exact1 = solver1.exact_solution(times1[-1])
    x_exact2, S_exact2 = solver2.exact_solution(times2[-1])
    plt.plot(x_exact1, S_exact1, 'b--', label='Exact (M=2)')
    plt.plot(x_exact2, S_exact2, 'r--', label='Exact (M=5)')
    
    plt.xlabel("Position (x)")
    plt.ylabel("Water Saturation (S)")
    plt.title("Comparison of Solutions for Different Mobility Ratios")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("buckley_leverett_results/buckley_leverett_solution_comparison.png")
    plt.show()
    
    print("\nComparison Summary:")
    print(f"Case 1 (M=2) - Final L2 Error: {metrics1[-1]['l2_error']:.6f}")
    print(f"Case 2 (M=5) - Final L2 Error: {metrics2[-1]['l2_error']:.6f}")