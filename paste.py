import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import time
import os
from matplotlib import cm
from mpl_toolkits.axes_3d import Axes3D

# Set random seeds for reproducibility
np.random.seed(1234)
torch.manual_seed(1234)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create directory for saving results
os.makedirs('results', exist_ok=True)

# =============================================
# Define the exact solution for 2D lid-driven cavity flow
# =============================================

def exact_solution(x, y, Re=100):
    """
    Approximate analytical solution for lid-driven cavity flow
    This is based on a series solution that works well for moderate Reynolds numbers
    
    Args:
        x, y: Coordinates in [0,1]×[0,1]
        Re: Reynolds number
    
    Returns:
        u, v, p: Velocity components and pressure
    """
    # Number of terms in the series
    N = 20
    
    # Initialize velocity and pressure
    u = np.zeros_like(x)
    v = np.zeros_like(y)
    p = np.zeros_like(x)
    
    # Lid velocity
    U = 1.0
    
    for n in range(1, N+1):
        # Eigenvalue
        lambda_n = n * np.pi
        
        # Terms for u component
        u += ((-1)**n) * (2/(lambda_n**3)) * (1 - np.cosh(lambda_n * (y - 0.5)) / np.cosh(lambda_n/2)) * np.cos(lambda_n * (x - 0.5))
        
        # Terms for v component
        v += ((-1)**(n+1)) * (2/(lambda_n**2)) * (np.sinh(lambda_n * (y - 0.5)) / np.cosh(lambda_n/2)) * np.sin(lambda_n * (x - 0.5))
        
        # Terms for pressure
        p += ((-1)**n) * (4/lambda_n) * (np.sinh(lambda_n * (y - 0.5)) / np.cosh(lambda_n/2)) * np.cos(lambda_n * (x - 0.5))
    
    # Scale by lid velocity and Reynolds number
    u *= U
    v *= U
    p *= U / Re
    
    return u, v, p

# =============================================
# Physics-Informed Neural Network Implementation
# =============================================

class PINN(nn.Module):
    def __init__(self, num_hidden_layers=6, num_neurons=50):
        super(PINN, self).__init__()
        
        # Input layer (x, y)
        self.input_layer = nn.Linear(2, num_neurons)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(num_neurons, num_neurons))
        
        # Output layer (u, v, p)
        self.output_layer = nn.Linear(num_neurons, 3)
        
        # Activation function
        self.activation = nn.Tanh()
    
    def forward(self, x):
        """
        Forward pass of the network
        
        Args:
            x: Input tensor of shape [batch_size, 2] containing coordinates (x, y)
            
        Returns:
            Output tensor of shape [batch_size, 3] containing (u, v, p)
        """
        x = self.activation(self.input_layer(x))
        
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        
        return self.output_layer(x)
    
    def net_uvp(self, x, y):
        """
        Get velocity and pressure at coordinates (x, y)
        
        Args:
            x, y: Coordinate tensors
            
        Returns:
            u, v, p: Velocity and pressure predictions
        """
        coords = torch.cat([x, y], dim=1)
        output = self.forward(coords)
        
        u = output[:, 0:1]
        v = output[:, 1:2]
        p = output[:, 2:3]
        
        return u, v, p

def compute_pinn_residuals(model, x, y, Re=100):
    """
    Compute the Navier-Stokes equation residuals for the PINN model
    
    Args:
        model: PINN model
        x, y: Coordinate tensors
        Re: Reynolds number
        
    Returns:
        continuity, momentum_x, momentum_y: Residuals for each equation
    """
    x.requires_grad_(True)
    y.requires_grad_(True)
    
    # Get predictions
    u, v, p = model.net_uvp(x, y)
    
    # Compute derivatives
    u_x = grad(u.sum(), x, create_graph=True)[0]
    u_y = grad(u.sum(), y, create_graph=True)[0]
    v_x = grad(v.sum(), x, create_graph=True)[0]
    v_y = grad(v.sum(), y, create_graph=True)[0]
    
    p_x = grad(p.sum(), x, create_graph=True)[0]
    p_y = grad(p.sum(), y, create_graph=True)[0]
    
    u_xx = grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = grad(u_y.sum(), y, create_graph=True)[0]
    v_xx = grad(v_x.sum(), x, create_graph=True)[0]
    v_yy = grad(v_y.sum(), y, create_graph=True)[0]
    
    # Continuity equation: ∇·u = 0
    continuity = u_x + v_y
    
    # Momentum equation x: u·∇u = -∇p + (1/Re)·∇²u
    momentum_x = (u * u_x + v * u_y) + p_x - (1/Re) * (u_xx + u_yy)
    
    # Momentum equation y: u·∇v = -∇p + (1/Re)·∇²v
    momentum_y = (u * v_x + v * v_y) + p_y - (1/Re) * (v_xx + v_yy)
    
    return continuity, momentum_x, momentum_y

def train_pinn(domain_points=10000, boundary_points=1000, Re=100, num_epochs=20000, 
               learning_rate=1e-3, display_interval=1000):
    """
    Train the PINN model to solve the steady Navier-Stokes equations
    
    Args:
        domain_points: Number of collocation points in the domain
        boundary_points: Number of points on each boundary
        Re: Reynolds number
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        display_interval: Interval for displaying training progress
        
    Returns:
        model: Trained model
        loss_history: History of losses during training
    """
    # Create model and move to device
    model = PINN(num_hidden_layers=8, num_neurons=50).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    
    # Generate domain points
    x_domain = torch.rand(domain_points, 1, device=device)
    y_domain = torch.rand(domain_points, 1, device=device)
    
    # Generate boundary points
    # Bottom boundary (y=0)
    x_bottom = torch.rand(boundary_points, 1, device=device)
    y_bottom = torch.zeros(boundary_points, 1, device=device)
    
    # Top boundary (y=1)
    x_top = torch.rand(boundary_points, 1, device=device)
    y_top = torch.ones(boundary_points, 1, device=device)
    
    # Left boundary (x=0)
    x_left = torch.zeros(boundary_points, 1, device=device)
    y_left = torch.rand(boundary_points, 1, device=device)
    
    # Right boundary (x=1)
    x_right = torch.ones(boundary_points, 1, device=device)
    y_right = torch.rand(boundary_points, 1, device=device)
    
    # Lists to store loss history
    loss_history = {'total': [], 'pde': [], 'bc': []}
    
    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs + 1):
        optimizer.zero_grad()
        
        # PDE residuals
        continuity, momentum_x, momentum_y = compute_pinn_residuals(model, x_domain, y_domain, Re)
        
        pde_loss = torch.mean(continuity**2) + torch.mean(momentum_x**2) + torch.mean(momentum_y**2)
        
        # Boundary conditions
        # Bottom wall: u=v=0
        u_bottom, v_bottom, _ = model.net_uvp(x_bottom, y_bottom)
        
        # Top wall: u=1, v=0 (lid)
        u_top, v_top, _ = model.net_uvp(x_top, y_top)
        
        # Left wall: u=v=0
        u_left, v_left, _ = model.net_uvp(x_left, y_left)
        
        # Right wall: u=v=0
        u_right, v_right, _ = model.net_uvp(x_right, y_right)
        
        bc_loss = (torch.mean(u_bottom**2) + torch.mean(v_bottom**2) + 
                   torch.mean((u_top - 1.0)**2) + torch.mean(v_top**2) + 
                   torch.mean(u_left**2) + torch.mean(v_left**2) + 
                   torch.mean(u_right**2) + torch.mean(v_right**2))
        
        # Total loss
        loss = pde_loss + 10.0 * bc_loss
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Record losses
        loss_history['total'].append(loss.item())
        loss_history['pde'].append(pde_loss.item())
        loss_history['bc'].append(bc_loss.item())
        
        # Print progress
        if epoch % display_interval == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.6f}, PDE Loss: {pde_loss.item():.6f}, BC Loss: {bc_loss.item():.6f}, Time: {elapsed:.2f}s")
            start_time = time.time()
    
    return model, loss_history

def evaluate_pinn(model, nx=101, ny=101, Re=100):
    """
    Evaluate the trained PINN model on a grid
    
    Args:
        model: Trained PINN model
        nx, ny: Number of grid points in x and y directions
        Re: Reynolds number
        
    Returns:
        x_grid, y_grid: Mesh grid coordinates
        u_pred, v_pred, p_pred: Predicted solutions
    """
    # Create mesh grid
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Convert to PyTorch tensors
    x_tensor = torch.tensor(x_grid.flatten()[:, None], dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_grid.flatten()[:, None], dtype=torch.float32, device=device)
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        u_pred, v_pred, p_pred = model.net_uvp(x_tensor, y_tensor)
    
    # Convert back to numpy and reshape
    u_pred = u_pred.cpu().numpy().reshape(ny, nx)
    v_pred = v_pred.cpu().numpy().reshape(ny, nx)
    p_pred = p_pred.cpu().numpy().reshape(ny, nx)
    
    # Center the pressure field (it's defined up to a constant)
    p_pred = p_pred - np.mean(p_pred)
    
    return x_grid, y_grid, u_pred, v_pred, p_pred

# =============================================
# WENO Scheme Implementation for Navier-Stokes
# =============================================

class WENOSolver:
    def __init__(self, nx=101, ny=101, Re=100, dt=0.001, max_iter=10000, tol=1e-6):
        """
        Initialize the WENO solver for incompressible steady Navier-Stokes
        
        Args:
            nx, ny: Number of grid points in x and y directions
            Re: Reynolds number
            dt: Time step for pseudo-time stepping
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
        """
        self.nx = nx
        self.ny = ny
        self.Re = Re
        self.dt = dt
        self.max_iter = max_iter
        self.tol = tol
        
        # Grid spacing
        self.dx = 1.0 / (nx - 1)
        self.dy = 1.0 / (ny - 1)
        
        # Initialize grid
        self.x = np.linspace(0, 1, nx)
        self.y = np.linspace(0, 1, ny)
        self.x_grid, self.y_grid = np.meshgrid(self.x, self.y)
        
        # Initialize solution
        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))
        
        # Boundary conditions for the lid-driven cavity flow
        self.u[-1, :] = 1.0  # Top lid moves with u=1
    
    def weno5_flux(self, v, f):
        """
        WENO5 scheme for computing numerical flux
        
        Args:
            v: Velocity component (u or v)
            f: Flux values
            
        Returns:
            flux_left, flux_right: Left and right fluxes
        """
        epsilon = 1e-6
        
        # Left flux at i+1/2
        f_im2 = np.roll(f, 2)
        f_im1 = np.roll(f, 1)
        f_i = f
        f_ip1 = np.roll(f, -1)
        f_ip2 = np.roll(f, -2)
        
        # Smoothness indicators
        beta0 = 13/12 * (f_im2 - 2*f_im1 + f_i)**2 + 1/4 * (f_im2 - 4*f_im1 + 3*f_i)**2
        beta1 = 13/12 * (f_im1 - 2*f_i + f_ip1)**2 + 1/4 * (f_im1 - f_ip1)**2
        beta2 = 13/12 * (f_i - 2*f_ip1 + f_ip2)**2 + 1/4 * (3*f_i - 4*f_ip1 + f_ip2)**2
        
        # Nonlinear weights
        d0 = 0.1 / ((epsilon + beta0)**2)
        d1 = 0.6 / ((epsilon + beta1)**2)
        d2 = 0.3 / ((epsilon + beta2)**2)
        
        omega0 = d0 / (d0 + d1 + d2)
        omega1 = d1 / (d0 + d1 + d2)
        omega2 = d2 / (d0 + d1 + d2)
        
        # Candidate stencils
        p0 = 1/3 * f_im2 - 7/6 * f_im1 + 11/6 * f_i
        p1 = -1/6 * f_im1 + 5/6 * f_i + 1/3 * f_ip1
        p2 = 1/3 * f_i + 5/6 * f_ip1 - 1/6 * f_ip2
        
        # Left flux
        flux_left = omega0 * p0 + omega1 * p1 + omega2 * p2
        
        # Right flux (at i-1/2)
        flux_right = np.roll(flux_left, 1)
        
        return flux_left, flux_right
    
    def solve_pressure_poisson(self, u, v, p, max_iter=1000, tol=1e-6):
        """
        Solve the pressure Poisson equation to enforce incompressibility
        
        Args:
            u, v: Velocity components
            p: Current pressure
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            
        Returns:
            p: Updated pressure
        """
        dx2 = self.dx ** 2
        dy2 = self.dy ** 2
        
        rhs = np.zeros_like(p)
        
        # Compute divergence for the RHS
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                rhs[j, i] = (
                    (u[j, i+1] - u[j, i-1]) / (2 * self.dx) +
                    (v[j+1, i] - v[j-1, i]) / (2 * self.dy)
                ) / self.dt
        
        # Gauss-Seidel iteration with SOR
        omega = 1.5  # SOR relaxation parameter
        p_new = p.copy()
        
        for _ in range(max_iter):
            for i in range(1, self.nx-1):
                for j in range(1, self.ny-1):
                    p_new[j, i] = (1 - omega) * p[j, i] + omega * (
                        (p_new[j, i+1] + p_new[j, i-1]) / dx2 +
                        (p_new[j+1, i] + p_new[j-1, i]) / dy2 -
                        rhs[j, i]
                    ) / (2 / dx2 + 2 / dy2)
            
            # Check convergence
            residual = np.max(np.abs(p_new - p))
            p[:] = p_new[:]
            
            if residual < tol:
                break
        
        # Zero mean pressure
        p -= np.mean(p)
        
        return p
    
    def solve(self):
        """
        Solve the Navier-Stokes equations using WENO scheme and projection method
        
        Returns:
            u, v, p: Solution arrays
            convergence_history: History of residuals
        """
        # Arrays for old values
        u_old = np.zeros_like(self.u)
        v_old = np.zeros_like(self.v)
        
        # Initialize convergence history
        convergence_history = []
        
        # Main iteration loop
        for iter in range(self.max_iter):
            # Store old values
            u_old[:] = self.u[:]
            v_old[:] = self.v[:]
            
            # Apply boundary conditions
            # Top lid: u=1, v=0
            self.u[-1, :] = 1.0
            self.v[-1, :] = 0.0
            
            # Other walls: u=v=0 (no-slip)
            self.u[0, :] = 0.0  # Bottom
            self.v[0, :] = 0.0
            
            self.u[:, 0] = 0.0  # Left
            self.v[:, 0] = 0.0
            
            self.u[:, -1] = 0.0  # Right
            self.v[:, -1] = 0.0
            
            # Compute advection terms using WENO
            # x-momentum advection
            flux_u_x_left, flux_u_x_right = self.weno5_flux(self.u, self.u)
            flux_u_y_left, flux_u_y_right = self.weno5_flux(self.v, self.u)
            
            # y-momentum advection
            flux_v_x_left, flux_v_x_right = self.weno5_flux(self.u, self.v)
            flux_v_y_left, flux_v_y_right = self.weno5_flux(self.v, self.v)
            
            # Interior points update
            for i in range(1, self.nx-1):
                for j in range(1, self.ny-1):
                    # Advection terms
                    adv_u_x = (flux_u_x_left[j, i] - flux_u_x_right[j, i]) / self.dx
                    adv_u_y = (flux_u_y_left[j, i] - flux_u_y_right[j, i]) / self.dy
                    
                    adv_v_x = (flux_v_x_left[j, i] - flux_v_x_right[j, i]) / self.dx
                    adv_v_y = (flux_v_y_left[j, i] - flux_v_y_right[j, i]) / self.dy
                    
                    # Pressure gradient
                    p_x = (self.p[j, i+1] - self.p[j, i-1]) / (2 * self.dx)
                    p_y = (self.p[j+1, i] - self.p[j-1, i]) / (2 * self.dy)
                    
                    # Viscous terms (central difference for second derivatives)
                    visc_u_x = (self.u[j, i+1] - 2*self.u[j, i] + self.u[j, i-1]) / (self.dx**2)
                    visc_u_y = (self.u[j+1, i] - 2*self.u[j, i] + self.u[j-1, i]) / (self.dy**2)
                    
                    visc_v_x = (self.v[j, i+1] - 2*self.v[j, i] + self.v[j, i-1]) / (self.dx**2)
                    visc_v_y = (self.v[j+1, i] - 2*self.v[j, i] + self.v[j-1, i]) / (self.dy**2)
                    
                    # Update velocities
                    self.u[j, i] += self.dt * (
                        -adv_u_x - adv_u_y - p_x + (1/self.Re) * (visc_u_x + visc_u_y)
                    )
                    
                    self.v[j, i] += self.dt * (
                        -adv_v_x - adv_v_y - p_y + (1/self.Re) * (visc_v_x + visc_v_y)
                    )
            
            # Solve pressure Poisson equation
            self.p = self.solve_pressure_poisson(self.u, self.v, self.p)
            
            # Project velocity field to enforce incompressibility
            for i in range(1, self.nx-1):
                for j in range(1, self.ny-1):
                    self.u[j, i] -= self.dt * (self.p[j, i+1] - self.p[j, i-1]) / (2 * self.dx)
                    self.v[j, i] -= self.dt * (self.p[j+1, i] - self.p[j-1, i]) / (2 * self.dy)
            
            # Compute convergence metric
            u_diff = np.max(np.abs(self.u - u_old))
            v_diff = np.max(np.abs(self.v - v_old))
            residual = max(u_diff, v_diff)
            
            convergence_history.append(residual)
            
            # Check for convergence
            if (iter + 1) % 100 == 0:
                print(f"Iteration {iter+1}, Residual: {residual:.6e}")
            
            if residual < self.tol:
                print(f"WENO solver converged after {iter+1} iterations.")
                break
        
        # Apply final boundary conditions (just to be sure)
        self.u[-1, :] = 1.0
        self.v[-1, :] = 0.0
        
        self.u[0, :] = 0.0  
        self.v[0, :] = 0.0
        
        self.u[:, 0] = 0.0
        self.v[:, 0] = 0.0
        
        self.u[:, -1] = 0.0
        self.v[:, -1] = 0.0
        
        return self.u, self.v, self.p, convergence_history

# =============================================
# Main Program for Comparison
# =============================================

def main(Re=100, nx=65, ny=65):
    """
    Main function to run the comparison between PINN and WENO
    
    Args:
        Re: Reynolds number
        nx, ny: Number of grid points for evaluation
    """
    print("=" * 50)
    print(f"Comparing PINN and WENO for Re = {Re}")
    print("=" * 50)
    
    # ======================
    # PART 1: PINN Solution
    # ======================
    print("\n[1/4] Training PINN model...")
    model, loss_history = train_pinn(Re=Re, num_epochs=15000)
    
    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history['total'], label='Total Loss')
    plt.semilogy(loss_history['pde'], label='PDE Loss')
    plt.semilogy(loss_history['bc'], label='BC Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PINN Training Loss History')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/pinn_loss_history.png', dpi=300, bbox_inches='tight')
    
    print("\n[2/4] Evaluating PINN model...")
    x_grid, y_grid, u_pinn, v_pinn, p_pinn = evaluate_pinn(model, nx=nx, ny=ny, Re=Re)
    
    # ======================
    # PART 2: WENO Solution
    # ======================
    print("\n[3/4] Running WENO solver...")
    weno_solver = WENOSolver(nx=nx, ny=ny, Re=Re, dt=0.001, max_iter=10000, tol=1e-5)
    u_weno, v_weno, p_weno, weno_history = weno_solver.solve()
    
    # Plot convergence history
    plt.figure(figsize=(10, 6))
    plt.semilogy(weno_history)
    plt.xlabel('Iteration')
    plt.ylabel('Residual')
    plt.title('WENO Solver Convergence History')
    plt.grid(True)
    plt.savefig('results/weno_convergence_history.png', dpi=300, bbox_inches='tight')
    
    # ======================
    # PART 3: Exact Solution
    # ======================
    print("\n[4/4] Computing exact solution...")
    u_exact, v_exact, p_exact = exact_solution(x_grid, y_grid, Re=Re)
    
    # ======================
    # PART 4: Error Analysis and Visualization
    # ======================
    print("\nGenerating comparative visualizations...")
    
    # Compute errors
    u_error_pinn = np.abs(u_pinn - u_exact)
    v_error_pinn = np.abs(v_pinn - v_exact)
    p_error_pinn = np.abs(p_pinn - p_exact)
    
    u_error_weno = np.abs(u_weno - u_exact)
    v_error_weno = np.abs(v_weno - v_exact)
    p_error_weno = np.abs(p_weno - p_exact)
    
    # Relative L2 errors
    u_rel_l2_pinn = np.sqrt(np.sum(u_error_pinn**2)) / np.sqrt(np.sum(u_exact**2))
    v_rel_l2_pinn = np.sqrt(np.sum(v_error_pinn**2)) / np.sqrt(np.sum(v_exact**2))
    p_rel_l2_pinn = np.sqrt(np.sum(p_error_pinn**2)) / np.sqrt(np.sum(p_exact**2))
    
    u_rel_l2_weno = np.sqrt(np.sum(u_error_weno**2)) / np.sqrt(np.sum(u_exact**2))
    v_rel_l2_weno = np.sqrt(np.sum(v_error_weno**2)) / np.sqrt(np.sum(v_exact**2))
    p_rel_l2_weno = np.sqrt(np.sum(p_error_weno**2)) / np.sqrt(np.sum(p_exact**2))
    
    # Print error summary
    print("\nError Summary:")
    print(f"PINN Relative L2 Errors: u = {u_rel_l2_pinn:.6f}, v = {v_rel_l2_pinn:.6f}, p = {p_rel_l2_pinn:.6f}")
    print(f"WENO Relative L2 Errors: u = {u_rel_l2_weno:.6f}, v = {v_rel_l2_weno:.6f}, p = {p_rel_l2_weno:.6f}")
    
    # Plot velocity magnitude

    # Plot velocity magnitude
    def plot_velocity_magnitude(u, v, x_grid, y_grid, title, filename):
        """Plot velocity magnitude as a contour plot"""
        vel_mag = np.sqrt(u**2 + v**2)
        
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(x_grid, y_grid, vel_mag, 50, cmap='viridis')
        plt.colorbar(contour, label='Velocity Magnitude')
        plt.streamplot(x_grid.T, y_grid.T, u.T, v.T, color='white', density=1.5, linewidth=0.5)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    # Plot all three solutions (exact, PINN, WENO)
    plot_velocity_magnitude(u_exact, v_exact, x_grid, y_grid, 
                           'Exact Solution: Velocity Magnitude',
                           'results/exact_velocity.png')
    
    plot_velocity_magnitude(u_pinn, v_pinn, x_grid, y_grid, 
                           'PINN Solution: Velocity Magnitude',
                           'results/pinn_velocity.png')
    
    plot_velocity_magnitude(u_weno, v_weno, x_grid, y_grid, 
                           'WENO Solution: Velocity Magnitude',
                           'results/weno_velocity.png')
    
    # Plot error comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # PINN errors
    cax1 = axes[0, 0].contourf(x_grid, y_grid, u_error_pinn, 50, cmap='hot')
    plt.colorbar(cax1, ax=axes[0, 0], label='Error')
    axes[0, 0].set_title('PINN u-velocity Error')
    
    cax2 = axes[0, 1].contourf(x_grid, y_grid, v_error_pinn, 50, cmap='hot')
    plt.colorbar(cax2, ax=axes[0, 1], label='Error')
    axes[0, 1].set_title('PINN v-velocity Error')
    
    cax3 = axes[0, 2].contourf(x_grid, y_grid, p_error_pinn, 50, cmap='hot')
    plt.colorbar(cax3, ax=axes[0, 2], label='Error')
    axes[0, 2].set_title('PINN Pressure Error')
    
    # WENO errors
    cax4 = axes[1, 0].contourf(x_grid, y_grid, u_error_weno, 50, cmap='hot')
    plt.colorbar(cax4, ax=axes[1, 0], label='Error')
    axes[1, 0].set_title('WENO u-velocity Error')
    
    cax5 = axes[1, 1].contourf(x_grid, y_grid, v_error_weno, 50, cmap='hot')
    plt.colorbar(cax5, ax=axes[1, 1], label='Error')
    axes[1, 1].set_title('WENO v-velocity Error')
    
    cax6 = axes[1, 2].contourf(x_grid, y_grid, p_error_weno, 50, cmap='hot')
    plt.colorbar(cax6, ax=axes[1, 2], label='Error')
    axes[1, 2].set_title('WENO Pressure Error')
    
    plt.tight_layout()
    plt.savefig('results/error_comparison.png', dpi=300, bbox_inches='tight')
    
    # Compare center line profiles
    center_x = nx // 2
    center_y = ny // 2
    
    # Vertical centerline (x = 0.5)
    plt.figure(figsize=(10, 8))
    plt.plot(y_grid[:, center_x], u_exact[:, center_x], 'k-', linewidth=2, label='Exact')
    plt.plot(y_grid[:, center_x], u_pinn[:, center_x], 'r--', linewidth=2, label='PINN')
    plt.plot(y_grid[:, center_x], u_weno[:, center_x], 'b-.', linewidth=2, label='WENO')
    plt.xlabel('y')
    plt.ylabel('u-velocity')
    plt.title('Centerline u-velocity Profile (x = 0.5)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/centerline_u_comparison.png', dpi=300, bbox_inches='tight')
    
    # Horizontal centerline (y = 0.5)
    plt.figure(figsize=(10, 8))
    plt.plot(x_grid[center_y, :], v_exact[center_y, :], 'k-', linewidth=2, label='Exact')
    plt.plot(x_grid[center_y, :], v_pinn[center_y, :], 'r--', linewidth=2, label='PINN')
    plt.plot(x_grid[center_y, :], v_weno[center_y, :], 'b-.', linewidth=2, label='WENO')
    plt.xlabel('x')
    plt.ylabel('v-velocity')
    plt.title('Centerline v-velocity Profile (y = 0.5)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/centerline_v_comparison.png', dpi=300, bbox_inches='tight')
    
    # 3D visualization of solutions
    fig = plt.figure(figsize=(18, 6))
    
    # Exact solution
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(x_grid, y_grid, u_exact, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax1.set_title('Exact u-velocity')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
    # PINN solution
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(x_grid, y_grid, u_pinn, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax2.set_title('PINN u-velocity')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    # WENO solution
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(x_grid, y_grid, u_weno, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax3.set_title('WENO u-velocity')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('u')
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig('results/3d_u_comparison.png', dpi=300, bbox_inches='tight')
    
    # Compute and print performance metrics
    print("\nPerformance Comparison:")
    print("=" * 50)
    print(f"{'Method':<10} {'u-velocity L2':<15} {'v-velocity L2':<15} {'Pressure L2':<15}")
    print("-" * 50)
    print(f"{'PINN':<10} {u_rel_l2_pinn:<15.6f} {v_rel_l2_pinn:<15.6f} {p_rel_l2_pinn:<15.6f}")
    print(f"{'WENO':<10} {u_rel_l2_weno:<15.6f} {v_rel_l2_weno:<15.6f} {p_rel_l2_weno:<15.6f}")
    print("=" * 50)
    
    # Save performance metrics to file
    with open('results/performance_comparison.txt', 'w') as f:
        f.write("Performance Comparison:\n")
        f.write("=" * 50 + "\n")
        f.write(f"{'Method':<10} {'u-velocity L2':<15} {'v-velocity L2':<15} {'Pressure L2':<15}\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'PINN':<10} {u_rel_l2_pinn:<15.6f} {v_rel_l2_pinn:<15.6f} {p_rel_l2_pinn:<15.6f}\n")
        f.write(f"{'WENO':<10} {u_rel_l2_weno:<15.6f} {v_rel_l2_weno:<15.6f} {p_rel_l2_weno:<15.6f}\n")
        f.write("=" * 50 + "\n")
    
    print("\nComparison completed. Results saved to 'results/' directory.")

# =============================================
# Run the comparison
# =============================================

if __name__ == "__main__":
    # Run for Re=100 (standard benchmark)
    main(Re=100, nx=65, ny=65)
    
    # Optionally, run for other Reynolds numbers to extend the comparison
    # main(Re=400, nx=65, ny=65)
    # main(Re=1000, nx=101, ny=101)