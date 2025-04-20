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

# Create directory for saving results_bl
os.makedirs('results_bl', exist_ok=True)


# =============================================
# Define the exact solution for Buckley-Leverett
# =============================================

def fractional_flow(s, M=2.0):
    """
    Fractional flow function for Buckley-Leverett equation
    
    Args:
        s: Water saturation
        M: Mobility ratio (oil/water viscosity ratio)
    
    Returns:
        f: Fractional flow of water
    """
    return s**2 / (s**2 + M * (1-s)**2)

def df_ds(s, M=2.0):
    """
    Derivative of fractional flow function
    """
    numerator = 2 * s * M * (1-s)**2
    denominator = (s**2 + M * (1-s)**2)**2
    return numerator / denominator

def exact_solution(x, t, M=2.0, s_l=1.0, s_r=0.0):
    """
    Exact solution for Buckley-Leverett equation with Riemann initial data
    
    Args:
        x: Spatial coordinates
        t: Time
        M: Mobility ratio
        s_l: Left saturation state
        s_r: Right saturation state
    
    Returns:
        s: Water saturation
    """
    # Initialize saturation
    s = np.zeros_like(x)
    
    # Shock location
    s_shock = 0.5  # Approximate for M=2
    f_shock = fractional_flow(s_shock, M)
    x_shock = t * df_ds(s_shock, M)
    
    # Rarefaction wave
    for i in range(len(x)):
        if x[i] <= 0:
            s[i] = s_l
        elif x[i] >= x_shock:
            s[i] = s_r
        else:
            # Find saturation value by inverting x/t = f'(s)
            # Using simple search method
            best_s = s_r
            min_diff = float('inf')
            for s_test in np.linspace(s_r, s_shock, 1000):
                wave_speed = df_ds(s_test, M)
                if abs(x[i]/t - wave_speed) < min_diff:
                    min_diff = abs(x[i]/t - wave_speed)
                    best_s = s_test
            s[i] = best_s
    
    return s

class PINN(nn.Module):
    def __init__(self, num_hidden_layers=6, num_neurons=50, M=2.0):
        super(PINN, self).__init__()
        
        self.M = M  # Mobility ratio
        
        # Input layer (x, t)
        self.input_layer = nn.Linear(2, num_neurons)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(num_neurons, num_neurons))
        
        # Output layer (s - saturation)
        self.output_layer = nn.Linear(num_neurons, 1)
        
        # Activation function
        self.activation = nn.Tanh()
    
    def forward(self, x):
        """
        Forward pass of the network
        
        Args:
            x: Input tensor of shape [batch_size, 2] containing coordinates (x, t)
            
        Returns:
            Output tensor of shape [batch_size, 1] containing saturation (s)
        """
        x = self.activation(self.input_layer(x))
        
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        
        return self.output_layer(x)
    
    def net_s(self, x, t):
        """
        Get saturation at coordinates (x, t)
        
        Args:
            x, t: Coordinate tensors
            
        Returns:
            s: Saturation prediction
        """
        coords = torch.cat([x, t], dim=1)
        s = self.forward(coords)
        return s
    
    def fractional_flow(self, s):
        """
        Fractional flow function
        """
        return s**2 / (s**2 + self.M * (1-s)**2)
    
    def df_ds(self, s):
        """
        Derivative of fractional flow function
        """
        numerator = 2 * s * self.M * (1-s)**2
        denominator = (s**2 + self.M * (1-s)**2)**2
        return numerator / denominator

def compute_pinn_residuals(model, x, t):
    """
    Compute the Buckley-Leverett equation residuals for the PINN model
    
    Args:
        model: PINN model
        x, t: Coordinate tensors
        
    Returns:
        residual: PDE residual
    """
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    # Get predictions
    s = model.net_s(x, t)
    
    # Compute derivatives
    s_x = grad(s.sum(), x, create_graph=True)[0]
    s_t = grad(s.sum(), t, create_graph=True)[0]
    
    # Compute flux derivative
    f = model.fractional_flow(s)
    df = model.df_ds(s)
    
    # Buckley-Leverett equation: ∂s/∂t + ∂f(s)/∂x = 0
    # ∂f(s)/∂x = df/ds * ∂s/∂x
    residual = s_t + df * s_x
    
    return residual

def train_pinn(domain_points=10000, boundary_points=1000, M=2.0, num_epochs=20000, 
               learning_rate=1e-3, display_interval=1000, x_max=1.0, t_max=0.5):
    """
    Train the PINN model to solve the Buckley-Leverett equation
    
    Args:
        domain_points: Number of collocation points in the domain
        boundary_points: Number of points on each boundary
        M: Mobility ratio
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        display_interval: Interval for displaying training progress
        x_max: Maximum x coordinate
        t_max: Maximum time
        
    Returns:
        model: Trained model
        loss_history: History of losses during training
    """
    # Create model and move to device
    model = PINN(num_hidden_layers=8, num_neurons=50, M=M).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    
    # Generate domain points
    x_domain = torch.rand(domain_points, 1, device=device) * x_max
    t_domain = torch.rand(domain_points, 1, device=device) * t_max
    
    # Generate initial condition points (t=0)
    x_initial = torch.rand(boundary_points, 1, device=device) * x_max
    t_initial = torch.zeros(boundary_points, 1, device=device)
    
    # Generate left boundary points (x=0)
    x_left = torch.zeros(boundary_points, 1, device=device)
    t_left = torch.rand(boundary_points, 1, device=device) * t_max
    
    # Lists to store loss history
    loss_history = {'total': [], 'pde': [], 'ic': [], 'bc': []}
    
    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs + 1):
        optimizer.zero_grad()
        
        # PDE residuals
        residual = compute_pinn_residuals(model, x_domain, t_domain)
        pde_loss = torch.mean(residual**2)
        
        # Initial condition: s(x,0) = 0 for x > 0, s(x,0) = 1 for x <= 0
        s_initial = model.net_s(x_initial, t_initial)
        ic_target = torch.zeros_like(s_initial)
        ic_target[x_initial <= 0] = 1.0
        ic_loss = torch.mean((s_initial - ic_target)**2)
        
        # Boundary condition: s(0,t) = 1 (constant injection)
        s_left = model.net_s(x_left, t_left)
        bc_loss = torch.mean((s_left - 1.0)**2)
        
        # Total loss
        loss = pde_loss + 10.0 * ic_loss + 10.0 * bc_loss
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Record losses
        loss_history['total'].append(loss.item())
        loss_history['pde'].append(pde_loss.item())
        loss_history['ic'].append(ic_loss.item())
        loss_history['bc'].append(bc_loss.item())
        
        # Print progress
        if epoch % display_interval == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.6f}, PDE: {pde_loss.item():.6f}, IC: {ic_loss.item():.6f}, BC: {bc_loss.item():.6f}, Time: {elapsed:.2f}s")
            start_time = time.time()
    
    return model, loss_history


def evaluate_pinn(model, x_grid, t_grid):
    """
    Evaluate the trained PINN model on a grid
    
    Args:
        model: Trained PINN model
        x_grid, t_grid: Mesh grid coordinates
        
    Returns:
        s_pred: Predicted saturation
    """
    # Create mesh grid
    x_mesh, t_mesh = np.meshgrid(x_grid, t_grid)
    
    # Convert to PyTorch tensors
    x_tensor = torch.tensor(x_mesh.flatten()[:, None], dtype=torch.float32, device=device)
    t_tensor = torch.tensor(t_mesh.flatten()[:, None], dtype=torch.float32, device=device)
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        s_pred = model.net_s(x_tensor, t_tensor)
    
    # Convert back to numpy and reshape
    s_pred = s_pred.cpu().numpy().reshape(len(t_grid), len(x_grid))
    
    return s_pred

# =============================================
# WENO Scheme Implementation for Navier-Stokes
# =============================================

class WENOSolver:
    def __init__(self, nx=101, nt=101, x_max=1.0, t_max=0.5, M=2.0, cfl=0.4):
        """
        Initialize the WENO solver for Buckley-Leverett equation
        
        Args:
            nx: Number of grid points in x direction
            nt: Number of time steps
            x_max: Maximum x coordinate
            t_max: Maximum time
            M: Mobility ratio
            cfl: CFL number
        """
        self.nx = nx
        self.nt = nt
        self.x_max = x_max
        self.t_max = t_max
        self.M = M
        self.cfl = cfl
        
        # Grid spacing
        self.dx = x_max / (nx - 1)
        
        # Initialize grid
        self.x = np.linspace(0, x_max, nx)
        self.t = np.linspace(0, t_max, nt)
        self.dt = self.t[1] - self.t[0]
        
        # Check CFL condition
        max_wave_speed = self.df_ds(0.5)  # Approximate maximum wave speed
        dt_cfl = cfl * self.dx / max_wave_speed
        if self.dt > dt_cfl:
            print(f"Warning: Time step {self.dt} is larger than CFL limit {dt_cfl}")
            self.dt = dt_cfl
            self.t = np.linspace(0, t_max, int(t_max/self.dt) + 1)
            self.nt = len(self.t)
        
        # Initialize solution
        self.s = np.zeros((self.nt, self.nx))
        
        # Initial condition: s(x,0) = 0 for x > 0, s(x,0) = 1 for x <= 0
        self.s[0, :] = np.where(self.x <= 0, 1.0, 0.0)
    
    def fractional_flow(self, s):
        """
        Fractional flow function
        """
        return s**2 / (s**2 + self.M * (1-s)**2)
    
    def df_ds(self, s):
        """
        Derivative of fractional flow function
        """
        numerator = 2 * s * self.M * (1-s)**2
        denominator = (s**2 + self.M * (1-s)**2)**2
        return numerator / denominator
    
    def weno5_reconstruction(self, q):
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
    
    def compute_flux(self, s):
        """
        Compute numerical flux using WENO reconstruction
        
        Args:
            s: Saturation array
            
        Returns:
            flux: Numerical flux
        """
        # Compute physical flux
        f = self.fractional_flow(s)
        
        # WENO reconstruction
        f_minus, f_plus = self.weno5_reconstruction(f)
        
        # Numerical flux (upwind based on wave speed sign)
        wave_speed = self.df_ds(s)
        pos_mask = wave_speed >= 0
        neg_mask = wave_speed < 0
        
        flux = np.zeros_like(f)
        flux[pos_mask] = f_minus[pos_mask]
        flux[neg_mask] = f_plus[neg_mask]
        
        return flux
    
    def solve(self):
        """
        Solve the Buckley-Leverett equation using WENO scheme
        
        Returns:
            s: Solution array for all time steps
        """
        # Time stepping
        for n in range(self.nt-1):
            # Compute flux
            flux = self.compute_flux(self.s[n, :])
            
            # Boundary conditions
            # Left boundary: s(0,t) = 1 (constant injection)
            self.s[n, 0] = 1.0
            
            # Update solution with conservative form
            # s_t + f(s)_x = 0
            for i in range(1, self.nx-1):
                self.s[n+1, i] = self.s[n, i] - self.dt/self.dx * (flux[i+1] - flux[i-1])/2
            
            # Enforce boundary conditions
            self.s[n+1, 0] = 1.0
            self.s[n+1, -1] = self.s[n+1, -2]  # Outflow condition
        
        return self.s

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

# Update the main function to include the residual and BC error comparisons
def main(M=2.0, nx=101, nt=101, x_max=1.0, t_max=0.5):
    """
    Main function to run the comparison between PINN and WENO
    
    Args:
        M: Mobility ratio
        nx: Number of grid points in x direction
        nt: Number of time steps
        x_max: Maximum x coordinate
        t_max: Maximum time
    """
    print("=" * 50)
    print(f"Comparing PINN and WENO for Buckley-Leverett equation with M = {M}")
    print("=" * 50)
    
    # Create grid
    x_grid = np.linspace(0, x_max, nx)
    t_grid = np.linspace(0, t_max, nt)
    
    # ======================
    # PART 1: PINN Solution
    # ======================
    print("\n[1/4] Training PINN model...")
    model, loss_history = train_pinn(M=M, num_epochs=100, x_max=x_max, t_max=t_max)
    
    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history['total'], label='Total Loss')
    plt.semilogy(loss_history['pde'], label='PDE Loss')
    plt.semilogy(loss_history['ic'], label='IC Loss')
    plt.semilogy(loss_history['bc'], label='BC Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PINN Training Loss History')
    plt.legend()
    plt.grid(True)
    plt.savefig('results_bl/bl_pinn_loss_history.png', dpi=300, bbox_inches='tight')
    
    print("\n[2/4] Evaluating PINN model...")
    s_pinn = evaluate_pinn(model, x_grid, t_grid)
    
    # ======================
    # PART 2: WENO Solution
    # ======================
    print("\n[3/4] Running WENO solver...")
    weno_solver = WENOSolver(nx=nx, nt=nt, x_max=x_max, t_max=t_max, M=M)
    s_weno = weno_solver.solve()
    
    # ======================
    # PART 3: Exact Solution
    # ======================
    print("\n[4/4] Computing exact solution...")
    s_exact = np.zeros((nt, nx))
    for i, t in enumerate(t_grid):
        if t > 0:  # Skip t=0 to avoid division by zero
            s_exact[i, :] = exact_solution(x_grid, t, M=M)
        else:
            s_exact[i, :] = np.where(x_grid <= 0, 1.0, 0.0)
    
    # ======================
    # PART 4: Visualization
    # ======================
    # Plot saturation profiles at different times
    times_to_plot = [0.1, 0.2, 0.3, 0.4]
    time_indices = [np.abs(t_grid - t).argmin() for t in times_to_plot]
    
    plt.figure(figsize=(15, 10))
    for i, t_idx in enumerate(time_indices):
        plt.subplot(2, 2, i+1)
        plt.plot(x_grid, s_exact[t_idx, :], 'k-', linewidth=2, label='Exact')
        plt.plot(x_grid, s_pinn[t_idx, :], 'r--', linewidth=2, label='PINN')
        plt.plot(x_grid, s_weno[t_idx, :], 'b-.', linewidth=2, label='WENO')
        plt.xlabel('x')
        plt.ylabel('Saturation')
        plt.title(f'Time t = {t_grid[t_idx]:.2f}')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results_bl/bl_saturation_profiles.png', dpi=300, bbox_inches='tight')
    
    # Compute errors
    s_error_pinn = np.abs(s_pinn - s_exact)
    s_error_weno = np.abs(s_weno - s_exact)
    
    # L2 relative error
    s_rel_l2_pinn = np.sqrt(np.sum(s_error_pinn**2)) / np.sqrt(np.sum(s_exact**2))
    s_rel_l2_weno = np.sqrt(np.sum(s_error_weno**2)) / np.sqrt(np.sum(s_exact**2))
    
    # Print error summary
    print("\nError Summary:")
    print(f"PINN Relative L2 Error: s = {s_rel_l2_pinn:.6f}")
    print(f"WENO Relative L2 Error: s = {s_rel_l2_weno:.6f}")
    
    # Plot error comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.contourf(x_grid, t_grid, s_error_pinn, 50, cmap='hot')
    plt.colorbar(label='Error')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('PINN Error')
    
    plt.subplot(1, 2, 2)
    plt.contourf(x_grid, t_grid, s_error_weno, 50, cmap='hot')
    plt.colorbar(label='Error')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('WENO Error')
    
    plt.tight_layout()
    plt.savefig('results_bl/bl_error_comparison.png', dpi=300, bbox_inches='tight')
    
    # Save performance metrics
    with open('results_bl/bl_performance_comparison.txt', 'w') as f:
        f.write("Performance Comparison for Buckley-Leverett:\n")
        f.write("=" * 50 + "\n")
        f.write(f"{'Method':<10} {'Saturation L2':<15}\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'PINN':<10} {s_rel_l2_pinn:<15.6f}\n")
        f.write(f"{'WENO':<10} {s_rel_l2_weno:<15.6f}\n")
        f.write("=" * 50 + "\n")
    
    print("\nComparison completed. Results saved to 'results_bl/' directory.")


# Run the modified comparison if this script is executed directly
if __name__ == "__main__":
    # Run for Re=100 (standard benchmark)
    main(M=2.0, nx=101, nt=101, x_max=1.0, t_max=0.5)