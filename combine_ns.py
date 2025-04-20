import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import time
import os
from matplotlib import cm

# Set random seeds for reproducibility
np.random.seed(1234)
torch.manual_seed(1234)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create directory for saving results_ns
os.makedirs('results_ns_ns', exist_ok=True)

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
        
        # Output layer for stream function (ψ)
        self.output_layer = nn.Linear(num_neurons, 1)
        
        # Activation function
        self.activation = nn.Tanh()
    
    def forward(self, x):
        """
        Forward pass of the network
        
        Args:
            x: Input tensor of shape [batch_size, 2] containing coordinates (x, y)
            
        Returns:
            Output tensor of shape [batch_size, 1] containing stream function (ψ)
        """
        x = self.activation(self.input_layer(x))
        
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        
        # Output is the stream function ψ
        return self.output_layer(x)
    
    def net_uvp(self, x, y):
        """
        Get velocity and pressure at coordinates (x, y)
        
        Args:
            x, y: Coordinate tensors
            
        Returns:
            u, v, p: Velocity and pressure predictions
        """
        x.requires_grad_(True)
        y.requires_grad_(True)
        
        coords = torch.cat([x, y], dim=1)
        
        # Compute stream function ψ
        psi = self.forward(coords)
        
        # Compute u and v from stream function derivatives
        # u = ∂ψ/∂y, v = -∂ψ/∂x
        u = grad(psi.sum(), y, create_graph=True)[0]
        v = -grad(psi.sum(), x, create_graph=True)[0]
        
        # For pressure, we'll predict it directly through a separate network
        # This is implemented as a separate forward pass with additional output neurons
        # For simplicity, we'll compute pressure through appropriate derivatives
        u_x = grad(u.sum(), x, create_graph=True)[0]
        u_y = grad(u.sum(), y, create_graph=True)[0]
        v_x = grad(v.sum(), x, create_graph=True)[0]
        v_y = grad(v.sum(), y, create_graph=True)[0]
        
        # Laplacian of stream function (∇²ψ) relates to vorticity (ω = v_x - u_y)
        # We use the Poisson equation for pressure: ∇²p = -ρ(∂u_i/∂x_j)(∂u_j/∂x_i)
        # For incompressible flow: ∇²p = -ρ(u_x²+2u_y·v_x+v_y²)
        p_field = -(u_x**2 + 2*u_y*v_x + v_y**2)
        
        # Integrate to get pressure (ignoring constant of integration)
        # This is a simplified approach - in reality, we might need to solve another system
        p = torch.cumsum(p_field, dim=0) / p_field.shape[0]
        
        return u, v, p

def compute_pinn_residuals(model, x, y, Re=100):
    """
    Compute the Navier-Stokes equation residuals for the PINN model
    
    Args:
        model: PINN model
        x, y: Coordinate tensors
        Re: Reynolds number
        
    Returns:
        momentum_x, momentum_y: Residuals for momentum equations
    """
    x.requires_grad_(True)
    y.requires_grad_(True)
    
    # Get predictions using stream function formulation
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
    
    # The continuity equation is automatically satisfied due to the stream function formulation
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
    
    # Initialize Adam optimizer
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
    
    # Adam Training loop
    start_time = time.time()
    for epoch in range(num_epochs + 1):
        optimizer.zero_grad()
        
        # PDE residuals
        continuity, momentum_x, momentum_y = compute_pinn_residuals(model, x_domain, y_domain, Re)
        
        pde_loss = torch.mean(momentum_x**2) + torch.mean(momentum_y**2) + torch.mean(continuity**2)
        
        # Boundary conditions for stream function formulation
        # Bottom wall: u=v=0 (ψ=const, we set ψ=0)
        _, _, psi_bottom = model.net_uvp(x_bottom, y_bottom)
        
        # Top wall: u=1, v=0 (ψ = y at y=1)
        u_top, v_top, _ = model.net_uvp(x_top, y_top)
        
        # Left wall: u=v=0 (ψ=const, we set ψ=0)
        _, _, psi_left = model.net_uvp(x_left, y_left)
        
        # Right wall: u=v=0 (ψ=const, we set ψ=0)
        _, _, psi_right = model.net_uvp(x_right, y_right)
        
        bc_loss = (torch.mean(psi_bottom**2) + 
                   torch.mean((u_top - 1.0)**2) + torch.mean(v_top**2) + 
                   torch.mean(psi_left**2) + 
                   torch.mean(psi_right**2))
        
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
    
    # L-BFGS optimization after Adam
    print("Starting L-BFGS optimization...")
    
    # Define closure function for L-BFGS
    def closure():
        optimizer_lbfgs.zero_grad()
        
        # PDE residuals
        continuity, momentum_x, momentum_y = compute_pinn_residuals(model, x_domain, y_domain, Re)
        pde_loss = torch.mean(momentum_x**2) + torch.mean(momentum_y**2) + torch.mean(continuity**2)
        
        # Boundary conditions
        _, _, psi_bottom = model.net_uvp(x_bottom, y_bottom)
        u_top, v_top, _ = model.net_uvp(x_top, y_top)
        _, _, psi_left = model.net_uvp(x_left, y_left)
        _, _, psi_right = model.net_uvp(x_right, y_right)
        
        bc_loss = (torch.mean(psi_bottom**2) + 
                   torch.mean((u_top - 1.0)**2) + torch.mean(v_top**2) + 
                   torch.mean(psi_left**2) + 
                   torch.mean(psi_right**2))
        
        # Total loss
        loss = pde_loss + 10.0 * bc_loss
        
        loss.backward()
        
        return loss
    
    # Initialize L-BFGS optimizer
    optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), 
                                        max_iter=500, 
                                        max_eval=500, 
                                        tolerance_grad=1e-05, 
                                        tolerance_change=1e-09, 
                                        history_size=50, 
                                        line_search_fn="strong_wolfe")
    
    # Run L-BFGS optimization
    optimizer_lbfgs.step(closure)
    
    print("L-BFGS optimization completed")
    
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
        psi_pred: Predicted stream function
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
        # First get u, v, p
        u_pred, v_pred, p_pred = model.net_uvp(x_tensor, y_tensor)
        
        # Get stream function
        coords = torch.cat([x_tensor, y_tensor], dim=1)
        psi_pred = model(coords)
    
    # Convert back to numpy and reshape
    u_pred = u_pred.cpu().numpy().reshape(ny, nx)
    v_pred = v_pred.cpu().numpy().reshape(ny, nx)
    p_pred = p_pred.cpu().numpy().reshape(ny, nx)
    psi_pred = psi_pred.cpu().numpy().reshape(ny, nx)
    
    # Center the pressure field (it's defined up to a constant)
    p_pred = p_pred - np.mean(p_pred)
    
    return x_grid, y_grid, u_pred, v_pred, p_pred, psi_pred


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
    
    def weno5_reconstruction(self, q):
        """
        WENO5 reconstruction for cell interfaces
        
        Args:
            q: Cell-centered values
            
        Returns:
            q_minus, q_plus: Reconstructed values at interfaces
        """

        # Shift arrays for stencil
        qm2 = np.roll(q, 2, axis=1)
        qm1 = np.roll(q, 1, axis=1)
        qp1 = np.roll(q, -1, axis=1)
        qp2 = np.roll(q, -2, axis=1)
        
        # Constants
        epsilon = 1e-6
        c_1 = 1/10
        c_2 = 6/10
        c_3 = 3/10
        
        # Smoothness indicators
        beta1 = 13/12 * (qm2 - 2*qm1 + q)**2 + 1/4 * (qm2 - 4*qm1 + 3*q)**2
        beta2 = 13/12 * (qm1 - 2*q + qp1)**2 + 1/4 * (qm1 - qp1)**2
        beta3 = 13/12 * (q - 2*qp1 + qp2)**2 + 1/4 * (3*q - 4*qp1 + qp2)**2
        
        # Compute nonlinear weights for q_{i+1/2}^-
        alpha1 = c_1 / ((epsilon + beta1)**2)
        alpha2 = c_2 / ((epsilon + beta2)**2)
        alpha3 = c_3 / ((epsilon + beta3)**2)
        
        omega1 = alpha1 / (alpha1 + alpha2 + alpha3)
        omega2 = alpha2 / (alpha1 + alpha2 + alpha3)
        omega3 = alpha3 / (alpha1 + alpha2 + alpha3)
        
        # Candidate stencils for q_{i+1/2}^-
        p1 = (2*qm2 - 7*qm1 + 11*q) / 6
        p2 = (-qm1 + 5*q + 2*qp1) / 6
        p3 = (2*q + 5*qp1 - qp2) / 6
        
        # Reconstruct q_{i+1/2}^-
        q_minus = omega1 * p1 + omega2 * p2 + omega3 * p3
        
        # For q_{i+1/2}^+ we need to shift our reference point to i+1
        # But instead of shifting all arrays again, we'll redefine the stencils
        
        # Smoothness indicators for the "plus" side
        beta1_plus = 13/12 * (q - 2*qp1 + qp2)**2 + 1/4 * (q - 4*qp1 + 3*qp2)**2
        beta2_plus = 13/12 * (qm1 - 2*q + qp1)**2 + 1/4 * (qm1 - qp1)**2
        beta3_plus = 13/12 * (qm2 - 2*qm1 + q)**2 + 1/4 * (3*qm2 - 4*qm1 + q)**2
        
        # Compute nonlinear weights for q_{i-1/2}^+
        alpha1_plus = c_3 / ((epsilon + beta1_plus)**2)
        alpha2_plus = c_2 / ((epsilon + beta2_plus)**2)
        alpha3_plus = c_1 / ((epsilon + beta3_plus)**2)
        
        omega1_plus = alpha1_plus / (alpha1_plus + alpha2_plus + alpha3_plus)
        omega2_plus = alpha2_plus / (alpha1_plus + alpha2_plus + alpha3_plus)
        omega3_plus = alpha3_plus / (alpha1_plus + alpha2_plus + alpha3_plus)
        
        # Candidate stencils for q_{i-1/2}^+
        p1_plus = (2*qp2 - 7*qp1 + 11*q) / 6
        p2_plus = (-qp1 + 5*q + 2*qm1) / 6
        p3_plus = (2*q + 5*qm1 - qm2) / 6
        
        # Reconstruct q_{i-1/2}^+
        q_plus = omega1_plus * p1_plus + omega2_plus * p2_plus + omega3_plus * p3_plus
        
        # Handle boundaries by zeroing out appropriate cells
        q_minus[:, 0] = q[:, 0]
        q_minus[:, -1] = q[:, -1]
        
        q_plus[:, 0] = q[:, 0]
        q_plus[:, -1] = q[:, -1]
        
        return q_minus, q_plus
    
    def compute_fluxes(self, u, v):
        """
        Compute the numerical fluxes using WENO reconstruction
        
        Args:
            u, v: Velocity components
            
        Returns:
            u_flux_x, u_flux_y, v_flux_x, v_flux_y: Advection fluxes
        """
        # x-direction fluxes for u-momentum
        u_minus_x, u_plus_x = self.weno5_reconstruction(u)
        
        # Define upwind fluxes based on sign of advection velocity
        u_pos_mask = u >= 0
        u_neg_mask = u < 0
        
        # Flux at i+1/2 for u-momentum in x-direction
        u_flux_x = np.zeros_like(u)
        u_flux_x[u_pos_mask] = u[u_pos_mask] * u_minus_x[u_pos_mask]
        u_flux_x[u_neg_mask] = u[u_neg_mask] * u_plus_x[u_neg_mask]
        
        # y-direction fluxes for u-momentum
        u_minus_y, u_plus_y = self.weno5_reconstruction(u.T)
        u_minus_y, u_plus_y = u_minus_y.T, u_plus_y.T
        
        # Flux at j+1/2 for u-momentum in y-direction
        u_flux_y = np.zeros_like(u)
        v_pos_mask = v >= 0
        v_neg_mask = v < 0
        u_flux_y[v_pos_mask] = v[v_pos_mask] * u_minus_y[v_pos_mask]
        u_flux_y[v_neg_mask] = v[v_neg_mask] * u_plus_y[v_neg_mask]
        
        # x-direction fluxes for v-momentum
        v_minus_x, v_plus_x = self.weno5_reconstruction(v)
        
        # Flux at i+1/2 for v-momentum in x-direction
        v_flux_x = np.zeros_like(v)
        v_flux_x[u_pos_mask] = u[u_pos_mask] * v_minus_x[u_pos_mask]
        v_flux_x[u_neg_mask] = u[u_neg_mask] * v_plus_x[u_neg_mask]
        
        # y-direction fluxes for v-momentum
        v_minus_y, v_plus_y = self.weno5_reconstruction(v.T)
        v_minus_y, v_plus_y = v_minus_y.T, v_plus_y.T
        
        # Flux at j+1/2 for v-momentum in y-direction
        v_flux_y = np.zeros_like(v)
        v_flux_y[v_pos_mask] = v[v_pos_mask] * v_minus_y[v_pos_mask]
        v_flux_y[v_neg_mask] = v[v_neg_mask] * v_plus_y[v_neg_mask]
        
        return u_flux_x, u_flux_y, v_flux_x, v_flux_y
    
    def solve_pressure_poisson(self, u, v, max_iter=100, tol=1e-5):
        """
        Solve the pressure Poisson equation to enforce incompressibility
        
        Args:
            u, v: Velocity components
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            
        Returns:
            p: Updated pressure
        """
        # Create right-hand side: div(u)/dt
        div = np.zeros((self.ny, self.nx))
        
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                div[j, i] = ((u[j, i+1] - u[j, i-1]) / (2*self.dx) + 
                            (v[j+1, i] - v[j-1, i]) / (2*self.dy)) / self.dt
        
        # Initialize pressure
        p = self.p.copy()
        
        # SOR coefficient
        omega = 1.5
        
        # Iterative solution
        for _ in range(max_iter):
            p_old = p.copy()
            
            for i in range(1, self.nx-1):
                for j in range(1, self.ny-1):
                    p[j, i] = (1-omega) * p[j, i] + omega/4 * (
                        p[j, i+1] + p[j, i-1] + p[j+1, i] + p[j-1, i] - 
                        div[j, i] * (self.dx**2)
                    )
            
            # Enforce Neumann boundary conditions
            p[0, :] = p[1, :]    # Bottom
            p[-1, :] = p[-2, :]  # Top
            p[:, 0] = p[:, 1]    # Left
            p[:, -1] = p[:, -2]  # Right
            
            # Check convergence
            if np.max(np.abs(p - p_old)) < tol:
                break
        
        # Normalize pressure to zero mean
        p = p - np.mean(p)
        
        return p
    
    def solve(self):
        """
        Solve the Navier-Stokes equations using WENO scheme and projection method
        
        Returns:
            u, v, p: Solution arrays
            convergence_history: History of residuals
        """
        # CFL condition for stability
        dt_cfl = 0.5 * min(self.dx, self.dy)**2 * self.Re
        if self.dt > dt_cfl:
            print(f"Warning: Time step {self.dt} is larger than CFL limit {dt_cfl}")
            print(f"Adjusting time step to {dt_cfl}")
            self.dt = dt_cfl
        
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
            
            # Compute advection fluxes using WENO
            u_flux_x, u_flux_y, v_flux_x, v_flux_y = self.compute_fluxes(self.u, self.v)
            
            # Temporary arrays for velocity update
            u_temp = np.copy(self.u)
            v_temp = np.copy(self.v)
            
            # Update interior points
            for i in range(1, self.nx-1):
                for j in range(1, self.ny-1):
                    # Advection terms (central differences for flux derivatives)
                    adv_u_x = (u_flux_x[j, i+1] - u_flux_x[j, i-1]) / (2*self.dx)
                    adv_u_y = (u_flux_y[j+1, i] - u_flux_y[j-1, i]) / (2*self.dy)
                    
                    adv_v_x = (v_flux_x[j, i+1] - v_flux_x[j, i-1]) / (2*self.dx)
                    adv_v_y = (v_flux_y[j+1, i] - v_flux_y[j-1, i]) / (2*self.dy)
                    
                    # Viscous terms (central difference for second derivatives)
                    visc_u_x = (self.u[j, i+1] - 2*self.u[j, i] + self.u[j, i-1]) / (self.dx**2)
                    visc_u_y = (self.u[j+1, i] - 2*self.u[j, i] + self.u[j-1, i]) / (self.dy**2)
                    
                    visc_v_x = (self.v[j, i+1] - 2*self.v[j, i] + self.v[j, i-1]) / (self.dx**2)
                    visc_v_y = (self.v[j+1, i] - 2*self.v[j, i] + self.v[j-1, i]) / (self.dy**2)
                    
                    # Update velocities with advection and viscous terms
                    u_temp[j, i] = self.u[j, i] + self.dt * (
                        -adv_u_x - adv_u_y + (1/self.Re) * (visc_u_x + visc_u_y)
                    )
                    
                    v_temp[j, i] = self.v[j, i] + self.dt * (
                        -adv_v_x - adv_v_y + (1/self.Re) * (visc_v_x + visc_v_y)
                    )
            
            # Update velocities
            self.u = u_temp.copy()
            self.v = v_temp.copy()
            
            # Enforce boundary conditions
            self.u[-1, :] = 1.0  # Top lid
            self.v[-1, :] = 0.0
            
            self.u[0, :] = 0.0  # Bottom
            self.v[0, :] = 0.0
            
            self.u[:, 0] = 0.0  # Left
            self.v[:, 0] = 0.0
            
            self.u[:, -1] = 0.0  # Right
            self.v[:, -1] = 0.0
            
            # Solve pressure Poisson equation
            self.p = self.solve_pressure_poisson(self.u, self.v)
            
            # Project velocity field to enforce incompressibility
            for i in range(1, self.nx-1):
                for j in range(1, self.ny-1):
                    self.u[j, i] -= self.dt * (self.p[j, i+1] - self.p[j, i-1]) / (2*self.dx)
                    self.v[j, i] -= self.dt * (self.p[j+1, i] - self.p[j-1, i]) / (2*self.dy)
            
            # Re-enforce boundary conditions
            self.u[-1, :] = 1.0  # Top lid
            self.v[-1, :] = 0.0
            
            self.u[0, :] = 0.0  # Bottom
            self.v[0, :] = 0.0
            
            self.u[:, 0] = 0.0  # Left
            self.v[:, 0] = 0.0
            
            self.u[:, -1] = 0.0  # Right
            self.v[:, -1] = 0.0
            
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
        
        return self.u, self.v, self.p, convergence_history

def compute_pinn_evaluation_metrics(model, nx=101, ny=101, Re=100):
    """
    Compute detailed evaluation metrics for PINN model including residuals and BC errors
    
    Args:
        model: Trained PINN model
        nx, ny: Number of grid points in x and y directions
        Re: Reynolds number
        
    Returns:
        Dict containing all evaluation metrics
    """
    # Create mesh grid
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Convert to PyTorch tensors
    x_tensor = torch.tensor(x_grid.flatten()[:, None], dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_grid.flatten()[:, None], dtype=torch.float32, device=device)
    
    # Compute residuals
    continuity, momentum_x, momentum_y = compute_pinn_residuals(model, x_tensor, y_tensor, Re)
    
    # Extract boundary points
    # Bottom boundary (y=0)
    bottom_indices = [i for i in range(len(y_tensor)) if y_tensor[i].item() < 1e-10]
    x_bottom = x_tensor[bottom_indices]
    y_bottom = y_tensor[bottom_indices]
    
    # Top boundary (y=1)
    top_indices = [i for i in range(len(y_tensor)) if abs(y_tensor[i].item() - 1.0) < 1e-10]
    x_top = x_tensor[top_indices]
    y_top = y_tensor[top_indices]
    
    # Left boundary (x=0)
    left_indices = [i for i in range(len(x_tensor)) if x_tensor[i].item() < 1e-10]
    x_left = x_tensor[left_indices]
    y_left = y_tensor[left_indices]
    
    # Right boundary (x=1)
    right_indices = [i for i in range(len(x_tensor)) if abs(x_tensor[i].item() - 1.0) < 1e-10]
    x_right = x_tensor[right_indices]
    y_right = y_tensor[right_indices]
    
    # Boundary conditions
    # Bottom wall: u=v=0
    u_bottom, v_bottom, _ = model.net_uvp(x_bottom, y_bottom)
    
    # Top wall: u=1, v=0 (lid)
    u_top, v_top, _ = model.net_uvp(x_top, y_top)
    
    # Left wall: u=v=0
    u_left, v_left, _ = model.net_uvp(x_left, y_left)
    
    # Right wall: u=v=0
    u_right, v_right, _ = model.net_uvp(x_right, y_right)
    
    # Calculate BC errors
    bc_error_bottom = torch.mean(u_bottom**2 + v_bottom**2).cpu().item()
    bc_error_top = torch.mean((u_top - 1.0)**2 + v_top**2).cpu().item()
    bc_error_left = torch.mean(u_left**2 + v_left**2).cpu().item()
    bc_error_right = torch.mean(u_right**2 + v_right**2).cpu().item()
    
    # Get full solution for comparison
    model.eval()
    # with torch.no_grad():
    u_pred, v_pred, p_pred = model.net_uvp(x_tensor, y_tensor)
    
    # Convert back to numpy and reshape
    u_pred = u_pred.detach().cpu().numpy().reshape(ny, nx)
    v_pred = v_pred.detach().cpu().numpy().reshape(ny, nx)
    p_pred = p_pred.detach().cpu().numpy().reshape(ny, nx)
    
    # Center the pressure field (it's defined up to a constant)
    p_pred = p_pred - np.mean(p_pred)
    
    # Compute mean squared residuals
    mean_continuity = torch.mean(continuity**2).cpu().item()
    mean_momentum_x = torch.mean(momentum_x**2).cpu().item()
    mean_momentum_y = torch.mean(momentum_y**2).cpu().item()
    
    metrics = {
        'u': u_pred,
        'v': v_pred,
        'p': p_pred,
        'x_grid': x_grid,
        'y_grid': y_grid,
        'residuals': {
            'continuity': mean_continuity,
            'momentum_x': mean_momentum_x,
            'momentum_y': mean_momentum_y,
            'total': mean_continuity + mean_momentum_x + mean_momentum_y
        },
        'bc_errors': {
            'bottom': bc_error_bottom,
            'top': bc_error_top,
            'left': bc_error_left,
            'right': bc_error_right,
            'total': bc_error_bottom + bc_error_top + bc_error_left + bc_error_right
        }
    }
    
    return metrics

def compute_weno_residuals(u, v, p, dx, dy, Re=100):
    """
    Compute residuals of WENO solution for Navier-Stokes equations
    
    Args:
        u, v: Velocity components
        p: Pressure
        dx, dy: Grid spacing
        Re: Reynolds number
        
    Returns:
        Dict containing residuals
    """
    ny, nx = u.shape
    
    # Initialize residuals
    continuity = np.zeros((ny, nx))
    momentum_x = np.zeros((ny, nx))
    momentum_y = np.zeros((ny, nx))
    
    # Compute residuals at interior points
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # Continuity equation: ∇·u = 0
            continuity[j, i] = (u[j, i+1] - u[j, i-1]) / (2*dx) + (v[j+1, i] - v[j-1, i]) / (2*dy)
            
            # Velocity derivatives for advection terms (central differences)
            u_x = (u[j, i+1] - u[j, i-1]) / (2*dx)
            u_y = (u[j+1, i] - u[j-1, i]) / (2*dy)
            v_x = (v[j, i+1] - v[j, i-1]) / (2*dx)
            v_y = (v[j+1, i] - v[j-1, i]) / (2*dy)
            
            # Second derivatives for viscous terms
            u_xx = (u[j, i+1] - 2*u[j, i] + u[j, i-1]) / (dx**2)
            u_yy = (u[j+1, i] - 2*u[j, i] + u[j-1, i]) / (dy**2)
            v_xx = (v[j, i+1] - 2*v[j, i] + v[j, i-1]) / (dx**2)
            v_yy = (v[j+1, i] - 2*v[j, i] + v[j-1, i]) / (dy**2)
            
            # Pressure derivatives
            p_x = (p[j, i+1] - p[j, i-1]) / (2*dx)
            p_y = (p[j+1, i] - p[j-1, i]) / (2*dy)
            
            # Momentum equations
            momentum_x[j, i] = u[j, i] * u_x + v[j, i] * u_y + p_x - (1/Re) * (u_xx + u_yy)
            momentum_y[j, i] = u[j, i] * v_x + v[j, i] * v_y + p_y - (1/Re) * (v_xx + v_yy)
    
    # Mean squared residuals
    mean_continuity = np.mean(continuity[1:-1, 1:-1]**2)
    mean_momentum_x = np.mean(momentum_x[1:-1, 1:-1]**2)
    mean_momentum_y = np.mean(momentum_y[1:-1, 1:-1]**2)
    
    # Check boundary conditions
    # Bottom wall (y=0): u=v=0
    bc_error_bottom = np.mean(u[0, :]**2 + v[0, :]**2)
    
    # Top wall (y=nx-1): u=1, v=0 (lid)
    bc_error_top = np.mean((u[-1, :] - 1.0)**2 + v[-1, :]**2)
    
    # Left wall (x=0): u=v=0
    bc_error_left = np.mean(u[:, 0]**2 + v[:, 0]**2)
    
    # Right wall (x=nx-1): u=v=0
    bc_error_right = np.mean(u[:, -1]**2 + v[:, -1]**2)
    
    return {
        'residuals': {
            'continuity': mean_continuity,
            'momentum_x': mean_momentum_x,
            'momentum_y': mean_momentum_y, 
            'total': mean_continuity + mean_momentum_x + mean_momentum_y
        },
        'bc_errors': {
            'bottom': bc_error_bottom,
            'top': bc_error_top,
            'left': bc_error_left,
            'right': bc_error_right,
            'total': bc_error_bottom + bc_error_top + bc_error_left + bc_error_right
        }
    }

# Update the main function to include the residual and BC error comparisons
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

    train_time_init = time.time()

    model, loss_history = train_pinn(Re=Re, num_epochs=5000)  # You can adjust based on computational resources

    train_time = time.time() - train_time_init
    print(f"Training time for PINN: {train_time:.2f} seconds")
    
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
    plt.savefig('results_ns/pinn_loss_history.png', dpi=300, bbox_inches='tight')
    
    print("\n[2/4] Evaluating PINN model...")
    pinn_metrics = compute_pinn_evaluation_metrics(model, nx=nx, ny=ny, Re=Re)
    x_grid, y_grid = pinn_metrics['x_grid'], pinn_metrics['y_grid']
    u_pinn, v_pinn, p_pinn = pinn_metrics['u'], pinn_metrics['v'], pinn_metrics['p']
    
    # ======================
    # PART 2: WENO Solution
    # ======================
    print("\n[3/4] Running WENO solver...")

    solve_time_init = time.time()

    weno_solver = WENOSolver(nx=nx, ny=ny, Re=Re, dt=0.001, max_iter=50000, tol=1e-5)
    u_weno, v_weno, p_weno, weno_history = weno_solver.solve()

    solve_time = time.time() - solve_time_init
    print(f"Solving time for WENO: {solve_time:.2f} seconds")
    
    # Compute WENO residuals
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    weno_metrics = compute_weno_residuals(u_weno, v_weno, p_weno, dx, dy, Re)
    
    # Plot convergence history
    plt.figure(figsize=(10, 6))
    plt.semilogy(weno_history)
    plt.xlabel('Iteration')
    plt.ylabel('Residual')
    plt.title('WENO Solver Convergence History')
    plt.grid(True)
    plt.savefig('results_ns/weno_convergence_history.png', dpi=300, bbox_inches='tight')
    
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
    plt.savefig('results_ns/error_comparison.png', dpi=300, bbox_inches='tight')

    # =====================================
    # PART 5: Residual and BC Error Comparison (New)
    # =====================================
    print("\nResidual Comparison:")
    print("=" * 70)
    print(f"{'Method':<10} {'Continuity':<15} {'Momentum-X':<15} {'Momentum-Y':<15} {'Total':<15}")
    print("-" * 70)
    print(f"{'PINN':<10} {pinn_metrics['residuals']['continuity']:<15.6e} "
          f"{pinn_metrics['residuals']['momentum_x']:<15.6e} "
          f"{pinn_metrics['residuals']['momentum_y']:<15.6e} "
          f"{pinn_metrics['residuals']['total']:<15.6e}")
    print(f"{'WENO':<10} {weno_metrics['residuals']['continuity']:<15.6e} "
          f"{weno_metrics['residuals']['momentum_x']:<15.6e} "
          f"{weno_metrics['residuals']['momentum_y']:<15.6e} "
          f"{weno_metrics['residuals']['total']:<15.6e}")
    print("=" * 70)
    
    print("\nBoundary Condition Error Comparison:")
    print("=" * 80)
    print(f"{'Method':<10} {'Bottom Wall':<15} {'Top Wall (Lid)':<15} {'Left Wall':<15} {'Right Wall':<15} {'Total':<15}")
    print("-" * 80)
    print(f"{'PINN':<10} {pinn_metrics['bc_errors']['bottom']:<15.6e} "
          f"{pinn_metrics['bc_errors']['top']:<15.6e} "
          f"{pinn_metrics['bc_errors']['left']:<15.6e} "
          f"{pinn_metrics['bc_errors']['right']:<15.6e} "
          f"{pinn_metrics['bc_errors']['total']:<15.6e}")
    print(f"{'WENO':<10} {weno_metrics['bc_errors']['bottom']:<15.6e} "
          f"{weno_metrics['bc_errors']['top']:<15.6e} "
          f"{weno_metrics['bc_errors']['left']:<15.6e} "
          f"{weno_metrics['bc_errors']['right']:<15.6e} "
          f"{weno_metrics['bc_errors']['total']:<15.6e}")
    print("=" * 80)
    
    # =====================================
    # PART 6: Visualize Residuals (New)
    # =====================================
    # Create bar plots for residuals comparison
    labels = ['Continuity', 'Momentum-X', 'Momentum-Y', 'Total']
    pinn_resid = [pinn_metrics['residuals']['continuity'], 
                 pinn_metrics['residuals']['momentum_x'],
                 pinn_metrics['residuals']['momentum_y'],
                 pinn_metrics['residuals']['total']]
    weno_resid = [weno_metrics['residuals']['continuity'], 
                 weno_metrics['residuals']['momentum_x'],
                 weno_metrics['residuals']['momentum_y'],
                 weno_metrics['residuals']['total']]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(labels))
    width = 0.35
    
    # Plot in log scale
    rects1 = ax.bar(x - width/2, pinn_resid, width, label='PINN')
    rects2 = ax.bar(x + width/2, weno_resid, width, label='WENO')
    
    ax.set_ylabel('Mean Squared Residual')
    ax.set_title('Comparison of Residuals Between PINN and WENO')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_yscale('log')
    
    fig.tight_layout()
    plt.savefig('results_ns/residual_comparison.png', dpi=300, bbox_inches='tight')
    
    # Bar plots for BC errors
    labels = ['Bottom Wall', 'Top Wall (Lid)', 'Left Wall', 'Right Wall', 'Total']
    pinn_bc = [pinn_metrics['bc_errors']['bottom'], 
              pinn_metrics['bc_errors']['top'],
              pinn_metrics['bc_errors']['left'],
              pinn_metrics['bc_errors']['right'],
              pinn_metrics['bc_errors']['total']]
    weno_bc = [weno_metrics['bc_errors']['bottom'], 
              weno_metrics['bc_errors']['top'],
              weno_metrics['bc_errors']['left'],
              weno_metrics['bc_errors']['right'],
              weno_metrics['bc_errors']['total']]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(labels))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, pinn_bc, width, label='PINN')
    rects2 = ax.bar(x + width/2, weno_bc, width, label='WENO')
    
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('Comparison of Boundary Condition Errors Between PINN and WENO')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_yscale('log')
    
    fig.tight_layout()
    plt.savefig('results_ns/bc_error_comparison.png', dpi=300, bbox_inches='tight')
    
    # Save performance metrics to file with residual and BC comparison
    with open('results_ns/comprehensive_comparison.txt', 'w') as f:
        f.write("Performance Comparison:\n")
        f.write("=" * 50 + "\n")
        f.write(f"{'Method':<10} {'u-velocity L2':<15} {'v-velocity L2':<15} {'Pressure L2':<15}\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'PINN':<10} {u_rel_l2_pinn:<15.6f} {v_rel_l2_pinn:<15.6f} {p_rel_l2_pinn:<15.6f}\n")
        f.write(f"{'WENO':<10} {u_rel_l2_weno:<15.6f} {v_rel_l2_weno:<15.6f} {p_rel_l2_weno:<15.6f}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Residual Comparison:\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Method':<10} {'Continuity':<15} {'Momentum-X':<15} {'Momentum-Y':<15} {'Total':<15}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'PINN':<10} {pinn_metrics['residuals']['continuity']:<15.6e} "
              f"{pinn_metrics['residuals']['momentum_x']:<15.6e} "
              f"{pinn_metrics['residuals']['momentum_y']:<15.6e} "
              f"{pinn_metrics['residuals']['total']:<15.6e}\n")
        f.write(f"{'WENO':<10} {weno_metrics['residuals']['continuity']:<15.6e} "
              f"{weno_metrics['residuals']['momentum_x']:<15.6e} "
              f"{weno_metrics['residuals']['momentum_y']:<15.6e} "
              f"{weno_metrics['residuals']['total']:<15.6e}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Boundary Condition Error Comparison:\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Method':<10} {'Bottom Wall':<15} {'Top Wall (Lid)':<15} {'Left Wall':<15} {'Right Wall':<15} {'Total':<15}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'PINN':<10} {pinn_metrics['bc_errors']['bottom']:<15.6e} "
              f"{pinn_metrics['bc_errors']['top']:<15.6e} "
              f"{pinn_metrics['bc_errors']['left']:<15.6e} "
              f"{pinn_metrics['bc_errors']['right']:<15.6e} "
              f"{pinn_metrics['bc_errors']['total']:<15.6e}\n")
        f.write(f"{'WENO':<10} {weno_metrics['bc_errors']['bottom']:<15.6e} "
              f"{weno_metrics['bc_errors']['top']:<15.6e} "
              f"{weno_metrics['bc_errors']['left']:<15.6e} "
              f"{weno_metrics['bc_errors']['right']:<15.6e} "
              f"{weno_metrics['bc_errors']['total']:<15.6e}\n")
        f.write("=" * 80 + "\n")

        
    
    print("\nComprehensive comparison completed. results saved to 'results_ns/' directory.")
    print(f"\nTraining time for PINN: {train_time:.2f} seconds")
    print(f"Solving time for WENO: {solve_time:.2f} seconds")

    # Measure inference time for PINN
    print("\nMeasuring inference time for PINN...")
    inference_time_pinn_start = time.time()
    pinn_metrics = compute_pinn_evaluation_metrics(model, nx=nx, ny=ny, Re=Re)
    inference_time_pinn = time.time() - inference_time_pinn_start
    print(f"Inference time for PINN: {inference_time_pinn:.6f} seconds")
    
    # Measure inference time for WENO
    print("\nMeasuring inference time for WENO...")
    inference_time_weno_start = time.time()
    u_weno, v_weno, p_weno, _ = weno_solver.solve()
    inference_time_weno = time.time() - inference_time_weno_start
    print(f"Inference time for WENO: {inference_time_weno:.6f} seconds")
    
    # Save inference times to file
    with open('results_ns/inference_times.txt', 'w') as f:
        f.write("Inference Times:\n")
        f.write("=" * 30 + "\n")
        f.write(f"PINN Inference Time: {inference_time_pinn:.6f} seconds\n")
        f.write(f"WENO Inference Time: {inference_time_weno:.6f} seconds\n")
        f.write("=" * 30 + "\n")
    
    print("\nInference times saved to 'results_ns/inference_times.txt'")

# Run the modified comparison if this script is executed directly
if __name__ == "__main__":
    # Run for Re=100 (standard benchmark)
    main(Re=100, nx=65, ny=65)