# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d
# from tqdm import tqdm

# # Set seeds for reproducibility
# np.random.seed(1234)
# torch.manual_seed(1234)

# # Enable GPU if available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# # Parameters for Buckley-Leverett equation
# class BuckleyLeverettParams:
#     def __init__(self, M=2.0):
#         self.M = M  # Mobility ratio (oil/water)
    
#     def flux_function(self, s):
#         """Fractional flow function f(s)"""
#         return s**2 / (s**2 + self.M * (1 - s)**2)
    
#     def flux_derivative(self, s):
#         """Derivative of fractional flow function f'(s)"""
#         num = 2 * s * (s**2 + self.M * (1 - s)**2) - s**2 * (2*s - 2*self.M*(1-s))
#         denom = (s**2 + self.M * (1 - s)**2)**2
#         return num / denom

# # Exact solution of Buckley-Leverett equation
# def exact_solution(x, t, params, s_l=0.0, s_r=1.0):
#     """
#     Compute the exact solution for the Riemann problem of Buckley-Leverett equation
#     using the method of characteristics and shock conditions
#     """
#     if t <= 0:
#         return np.where(x < 0, s_l, s_r)
    
#     # Find the shock location using equal-area rule
#     s_values = np.linspace(s_l + 1e-6, s_r - 1e-6, 1000)
#     f_values = np.array([params.flux_function(s) for s in s_values])
#     fp_values = np.array([params.flux_derivative(s) for s in s_values])
    
#     # Find s* where the characteristic speed equals the shock speed
#     def find_s_star():
#         for i in range(len(s_values)-1):
#             s1, s2 = s_values[i], s_values[i+1]
#             f1, f2 = f_values[i], f_values[i+1]
            
#             # Check if flux is concave at this point
#             if fp_values[i] < fp_values[i+1]:
#                 # Compute shock speed
#                 shock_speed = (f2 - f1) / (s2 - s1)
                
#                 # If characteristic speed at s1 > shock speed > characteristic speed at s2
#                 # This is our s*
#                 if fp_values[i] > shock_speed > fp_values[i+1]:
#                     return s1, shock_speed
#         return None, None
    
#     s_star, shock_speed = find_s_star()
    
#     if s_star is None:
#         # Fall back to simpler solution: pure rarefaction wave
#         s_star = s_r
#         shock_speed = params.flux_derivative(s_r)
    
#     # Compute solution
#     s_out = np.zeros_like(x, dtype=float)
    
#     # Behind the shock
#     mask_left = x <= shock_speed * t
#     s_out[mask_left] = s_l
    
#     # Rarefaction wave
#     wave_region = (x > 0) & (x <= params.flux_derivative(s_star) * t)
    
#     # Compute s values in rarefaction wave by inverting the characteristic relation
#     def invert_characteristic(x_over_t):
#         """Find s such that f'(s) = x/t"""
#         diff = np.abs(fp_values - x_over_t)
#         idx = np.argmin(diff)
#         return s_values[idx]
    
#     for i in range(len(x)):
#         if wave_region[i]:
#             x_over_t = x[i] / t
#             s_out[i] = invert_characteristic(x_over_t)
    
#     # After the shock
#     mask_right = x > shock_speed * t
#     s_out[mask_right] = s_r
    
#     return s_out

# # ==================== PINN Implementation ====================
# class PINN(nn.Module):
#     def __init__(self, hidden_layers=4, neurons=20):
#         super(PINN, self).__init__()
        
#         # Neural network architecture
#         layers = []
#         layers.append(nn.Linear(2, neurons))  # Input layer (x, t)
#         layers.append(nn.Tanh())
        
#         for _ in range(hidden_layers):
#             layers.append(nn.Linear(neurons, neurons))
#             layers.append(nn.Tanh())
        
#         layers.append(nn.Linear(neurons, 1))  # Output layer (s)
#         layers.append(nn.Sigmoid())  # Ensure output is between 0 and 1
        
#         self.net = nn.Sequential(*layers)
    
#     def forward(self, x, t):
#         inputs = torch.cat([x, t], dim=1)
#         return self.net(inputs)

# def train_pinn(x_domain, t_final, params, n_points=1000, n_epochs=100000, s_l=0.0, s_r=1.0):
#     """Train a PINN to solve the Buckley-Leverett equation"""
#     model = PINN().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=500, factor=0.5, verbose=True)
    
#     # Generate collocation points for PDE residual
#     x_collocation = torch.linspace(x_domain[0], x_domain[1], n_points, device=device).view(-1, 1)
#     t_collocation = torch.linspace(0, t_final, n_points, device=device).view(-1, 1)
    
#     X, T = torch.meshgrid(x_collocation.view(-1), t_collocation.view(-1), indexing='ij')
#     x_pde = X.reshape(-1, 1)
#     t_pde = T.reshape(-1, 1)
    
#     # Generate points for initial condition
#     x_ic = torch.linspace(x_domain[0], x_domain[1], n_points, device=device).view(-1, 1)
#     t_ic = torch.zeros_like(x_ic, device=device)
#     s_ic = torch.zeros_like(x_ic, device=device)
#     s_ic[x_ic <= 0] = s_l
#     s_ic[x_ic > 0] = s_r
    
#     # Generate points for boundary conditions
#     t_bc = torch.linspace(0, t_final, n_points, device=device).view(-1, 1)
#     x_bc_left = torch.full_like(t_bc, x_domain[0], device=device)
#     x_bc_right = torch.full_like(t_bc, x_domain[1], device=device)
#     s_bc_left = torch.full_like(t_bc, s_l, device=device)
#     s_bc_right = torch.full_like(t_bc, s_r, device=device)
    
#     # Training loop
#     pbar = tqdm(range(n_epochs))
#     losses = []
    
#     def flux_torch(s):
#         """PyTorch version of flux function"""
#         return s**2 / (s**2 + params.M * (1 - s)**2)
    
#     for epoch in pbar:
#         optimizer.zero_grad()
        
#         # Compute PDE residual
#         x_pde.requires_grad = True
#         t_pde.requires_grad = True
        
#         # Forward pass
#         s_pred = model(x_pde, t_pde)
        
#         # Compute derivatives
#         s_t = torch.autograd.grad(
#             s_pred, t_pde, 
#             grad_outputs=torch.ones_like(s_pred), 
#             create_graph=True
#         )[0]
        
#         s_x = torch.autograd.grad(
#             s_pred, x_pde, 
#             grad_outputs=torch.ones_like(s_pred), 
#             create_graph=True
#         )[0]
        
#         # Buckley-Leverett PDE: s_t + f(s)_x = 0
#         # Computing f(s)_x = f'(s) * s_x
#         f_pred = flux_torch(s_pred)
#         f_x = torch.autograd.grad(
#             f_pred, x_pde, 
#             grad_outputs=torch.ones_like(f_pred), 
#             create_graph=True
#         )[0]
        
#         residual = s_t + f_x
#         loss_pde = torch.mean(residual**2)
        
#         # Initial condition loss
#         s_pred_ic = model(x_ic, t_ic)
#         loss_ic = torch.mean((s_pred_ic - s_ic)**2)
        
#         # Boundary condition loss
#         s_pred_bc_left = model(x_bc_left, t_bc)
#         s_pred_bc_right = model(x_bc_right, t_bc)
#         loss_bc = torch.mean((s_pred_bc_left - s_bc_left)**2) + torch.mean((s_pred_bc_right - s_bc_right)**2)
        
#         # Total loss
#         loss = loss_pde + 10.0 * loss_ic + 10.0 * loss_bc
#         losses.append(loss.item())
        
#         # Backpropagation
#         loss.backward()
#         optimizer.step()
#         scheduler.step(loss)
        
#         # Update progress bar
#         if epoch % 100 == 0:
#             pbar.set_description(f"Loss: {loss.item():.6f}")
    
#     plt.figure(figsize=(10, 4))
#     plt.plot(losses)
#     plt.yscale('log')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('PINN Training Loss')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig('pinn_training_loss.png')
    
#     return model

# # ==================== WENO Implementation ====================
# def weno5_reconstruction(v_minus2, v_minus1, v, v_plus1, v_plus2):
#     """Fifth-order WENO reconstruction for flux computation"""
#     epsilon = 1e-6
    
#     # Compute smoothness indicators
#     beta0 = 13/12 * (v_minus2 - 2*v_minus1 + v)**2 + 1/4 * (v_minus2 - 4*v_minus1 + 3*v)**2
#     beta1 = 13/12 * (v_minus1 - 2*v + v_plus1)**2 + 1/4 * (v_minus1 - v_plus1)**2
#     beta2 = 13/12 * (v - 2*v_plus1 + v_plus2)**2 + 1/4 * (3*v - 4*v_plus1 + v_plus2)**2
    
#     # Compute nonlinear weights
#     d0 = 1/10
#     d1 = 6/10
#     d2 = 3/10
    
#     alpha0 = d0 / (epsilon + beta0)**2
#     alpha1 = d1 / (epsilon + beta1)**2
#     alpha2 = d2 / (epsilon + beta2)**2
    
#     omega0 = alpha0 / (alpha0 + alpha1 + alpha2)
#     omega1 = alpha1 / (alpha0 + alpha1 + alpha2)
#     omega2 = alpha2 / (alpha0 + alpha1 + alpha2)
    
#     # Candidate stencils
#     p0 = 1/3 * v_minus2 - 7/6 * v_minus1 + 11/6 * v
#     p1 = -1/6 * v_minus1 + 5/6 * v + 1/3 * v_plus1
#     p2 = 1/3 * v + 5/6 * v_plus1 - 1/6 * v_plus2
    
#     # Final reconstruction
#     return omega0 * p0 + omega1 * p1 + omega2 * p2

# def solve_weno(x_domain, t_final, params, nx=400, nt=800, s_l=0.0, s_r=1.0):
#     """Solve the Buckley-Leverett equation using WENO scheme"""
#     # Set up grid
#     dx = (x_domain[1] - x_domain[0]) / nx
#     dt = t_final / nt
    
#     # Check CFL condition
#     max_slope = params.flux_derivative(0.5)  # Approximate maximum slope
#     cfl = max_slope * dt / dx
#     print(f"CFL number: {cfl:.4f} (should be < 1)")
    
#     if cfl >= 1.0:
#         nt = int(nt * 1.2)  # Increase time steps if CFL condition not satisfied
#         dt = t_final / nt
#         print(f"Adjusted time steps to {nt}, new dt = {dt}, new CFL = {max_slope * dt / dx:.4f}")
    
#     # Initialize solution array
#     x = np.linspace(x_domain[0], x_domain[1], nx+1)
#     s = np.zeros_like(x)
#     s[x <= 0] = s_l
#     s[x > 0] = s_r
    
#     # Use ghost cells for boundary conditions
#     def apply_boundary(s):
#         s_with_ghost = np.zeros(len(s) + 4)
#         s_with_ghost[2:-2] = s
#         s_with_ghost[0:2] = s_l  # Left ghost cells
#         s_with_ghost[-2:] = s_r  # Right ghost cells
#         return s_with_ghost
    
#     # Time stepping loop with progress bar
#     pbar = tqdm(total=nt)
#     convergence_history = []
    
#     for n in range(nt):
#         s_old = s.copy()
        
#         # Apply boundary conditions
#         s_bc = apply_boundary(s)
        
#         # Compute numerical fluxes using WENO reconstruction
#         flux = np.zeros(len(s) + 1)
        
#         for j in range(len(flux) - 1):
#             idx = j + 2  # Offset for ghost cells
            
#             # WENO reconstruction for left and right states
#             s_minus = weno5_reconstruction(
#                 s_bc[idx-3], s_bc[idx-2], s_bc[idx-1], s_bc[idx], s_bc[idx+1]
#             )
#             s_plus = weno5_reconstruction(
#                 s_bc[idx+2], s_bc[idx+1], s_bc[idx], s_bc[idx-1], s_bc[idx-2]
#             )
            
#             # Compute numerical flux (Lax-Friedrichs flux)
#             f_minus = params.flux_function(s_minus)
#             f_plus = params.flux_function(s_plus)
#             alpha = max(abs(params.flux_derivative(s_minus)), abs(params.flux_derivative(s_plus)))
            
#             flux[j] = 0.5 * (f_minus + f_plus - alpha * (s_plus - s_minus))
        
#         # Update solution using conservative form
#         for j in range(len(s)):
#             s[j] = s[j] - dt/dx * (flux[j+1] - flux[j])
        
#         # Check convergence
#         change = np.max(np.abs(s - s_old))
#         convergence_history.append(change)
#         pbar.set_description(f"Change: {change:.6f}")
#         pbar.update(1)
    
#     pbar.close()
    
#     # Plot convergence history
#     plt.figure(figsize=(10, 4))
#     plt.plot(convergence_history)
#     plt.yscale('log')
#     plt.xlabel('Time Steps')
#     plt.ylabel('Max Change')
#     plt.title('WENO Convergence History')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig('weno_convergence.png')
    
#     return x, s

# # ==================== Main Execution ====================
# def main():
#     # Set problem parameters
#     x_domain = [-2.0, 2.0]
#     t_final = 0.5
#     params = BuckleyLeverettParams(M=2.0)
#     s_l = 0.0  # Left boundary condition
#     s_r = 1.0  # Right boundary condition
    
#     # Solve using WENO
#     print("Solving using WENO scheme...")
#     x_weno, s_weno = solve_weno(x_domain, t_final, params, nx=400, nt=100000, s_l=s_l, s_r=s_r)
    
#     # Compute exact solution
#     print("Computing exact solution...")
#     x_exact = np.linspace(x_domain[0], x_domain[1], 1000)
#     s_exact = exact_solution(x_exact, t_final, params, s_l=s_l, s_r=s_r)
    
#     # Train PINN
#     print("Training PINN model...")
#     pinn_model = train_pinn(x_domain, t_final, params, n_points=100, n_epochs=100000, s_l=s_l, s_r=s_r)
    
#     # Evaluate PINN solution
#     print("Evaluating PINN solution...")
#     x_pinn = torch.linspace(x_domain[0], x_domain[1], 500, device=device).view(-1, 1)
#     t_pinn = torch.full_like(x_pinn, t_final, device=device)
    
#     with torch.no_grad():
#         s_pinn = pinn_model(x_pinn, t_pinn).cpu().numpy()
    
#     # Calculate errors
#     # Interpolate WENO and PINN solutions to the exact solution grid for error calculation
#     weno_interp = interp1d(x_weno, s_weno, kind='linear', bounds_error=False, fill_value=(s_l, s_r))
#     s_weno_interp = weno_interp(x_exact)
    
#     pinn_interp = interp1d(x_pinn.cpu().numpy().flatten(), s_pinn.flatten(), kind='linear', bounds_error=False, fill_value=(s_l, s_r))
#     s_pinn_interp = pinn_interp(x_exact)
    
#     l2_error_weno = np.sqrt(np.mean((s_weno_interp - s_exact)**2))
#     l2_error_pinn = np.sqrt(np.mean((s_pinn_interp - s_exact)**2))
#     l_inf_error_weno = np.max(np.abs(s_weno_interp - s_exact))
#     l_inf_error_pinn = np.max(np.abs(s_pinn_interp - s_exact))
    
#     print(f"WENO L2 Error: {l2_error_weno:.6f}")
#     print(f"PINN L2 Error: {l2_error_pinn:.6f}")
#     print(f"WENO L∞ Error: {l_inf_error_weno:.6f}")
#     print(f"PINN L∞ Error: {l_inf_error_pinn:.6f}")
    
#     # Plot results
#     plt.figure(figsize=(12, 8))
    
#     plt.subplot(2, 1, 1)
#     plt.plot(x_exact, s_exact, 'k-', linewidth=2, label='Exact Solution')
#     plt.plot(x_weno, s_weno, 'r--', linewidth=1.5, label=f'WENO (L2: {l2_error_weno:.4f})')
#     plt.plot(x_pinn.cpu().numpy(), s_pinn, 'b-.', linewidth=1.5, label=f'PINN (L2: {l2_error_pinn:.4f})')
#     plt.xlabel('x')
#     plt.ylabel('Saturation (s)')
#     plt.title(f'Buckley-Leverett Equation Solutions at t = {t_final}')
#     plt.legend()
#     plt.grid(True)
    
#     plt.subplot(2, 1, 2)
#     plt.plot(x_exact, np.zeros_like(x_exact), 'k-', linewidth=1)  # Zero line
#     plt.plot(x_exact, s_weno_interp - s_exact, 'r--', linewidth=1.5, label=f'WENO Error (L∞: {l_inf_error_weno:.4f})')
#     plt.plot(x_exact, s_pinn_interp - s_exact, 'b-.', linewidth=1.5, label=f'PINN Error (L∞: {l_inf_error_pinn:.4f})')
#     plt.xlabel('x')
#     plt.ylabel('Error (s - s_exact)')
#     plt.title('Error Analysis')
#     plt.legend()
#     plt.grid(True)
    
#     plt.tight_layout()
#     plt.savefig('buckley_leverett_comparison.png')
#     plt.show()

# if __name__ == "__main__":
#     main()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm

# Set seeds for reproducibility
np.random.seed(1234)
torch.manual_seed(1234)

# Enable GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Parameters for Buckley-Leverett equation
class BuckleyLeverettParams:
    def __init__(self, M=2.0):
        self.M = M  # Mobility ratio (oil/water)
    
    def flux_function(self, s):
        """Fractional flow function f(s)"""
        return s**2 / (s**2 + self.M * (1 - s)**2)
    
    def flux_derivative(self, s):
        """Derivative of fractional flow function f'(s)"""
        num = 2 * s * (s**2 + self.M * (1 - s)**2) - s**2 * (2*s - 2*self.M*(1-s))
        denom = (s**2 + self.M * (1 - s)**2)**2
        return num / denom

# Exact solution of Buckley-Leverett equation
def exact_solution(x, t, params, s_l=0.0, s_r=1.0):
    """
    Compute the exact solution for the Riemann problem of Buckley-Leverett equation
    using the method of characteristics and shock conditions
    """
    if t <= 0:
        return np.where(x < 0, s_l, s_r)
    
    # Find the shock location using equal-area rule
    s_values = np.linspace(s_l + 1e-6, s_r - 1e-6, 1000)
    f_values = np.array([params.flux_function(s) for s in s_values])
    fp_values = np.array([params.flux_derivative(s) for s in s_values])
    
    # Find s* where the characteristic speed equals the shock speed
    def find_s_star():
        for i in range(len(s_values)-1):
            s1, s2 = s_values[i], s_values[i+1]
            f1, f2 = f_values[i], f_values[i+1]
            
            # Check if flux is concave at this point
            if fp_values[i] < fp_values[i+1]:
                # Compute shock speed
                shock_speed = (f2 - f1) / (s2 - s1)
                
                # If characteristic speed at s1 > shock speed > characteristic speed at s2
                # This is our s*
                if fp_values[i] > shock_speed > fp_values[i+1]:
                    return s1, shock_speed
        return None, None
    
    s_star, shock_speed = find_s_star()
    
    if s_star is None:
        # Fall back to simpler solution: pure rarefaction wave
        s_star = s_r
        shock_speed = params.flux_derivative(s_r)
    
    # Compute solution
    s_out = np.zeros_like(x, dtype=float)
    
    # Behind the shock
    mask_left = x <= shock_speed * t
    s_out[mask_left] = s_l
    
    # Rarefaction wave
    wave_region = (x > 0) & (x <= params.flux_derivative(s_star) * t)
    
    # Compute s values in rarefaction wave by inverting the characteristic relation
    def invert_characteristic(x_over_t):
        """Find s such that f'(s) = x/t"""
        diff = np.abs(fp_values - x_over_t)
        idx = np.argmin(diff)
        return s_values[idx]
    
    for i in range(len(x)):
        if wave_region[i]:
            x_over_t = x[i] / t
            s_out[i] = invert_characteristic(x_over_t)
    
    # After the shock
    mask_right = x > shock_speed * t
    s_out[mask_right] = s_r
    
    return s_out

# ==================== PINN Implementation ====================
class PINN(nn.Module):
    def __init__(self, hidden_layers=4, neurons=20):
        super(PINN, self).__init__()
        
        # Neural network architecture
        layers = []
        layers.append(nn.Linear(2, neurons))  # Input layer (x, t)
        layers.append(nn.Tanh())
        
        for _ in range(hidden_layers):
            layers.append(nn.Linear(neurons, neurons))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(neurons, 1))  # Output layer (s)
        layers.append(nn.Sigmoid())  # Ensure output is between 0 and 1
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

def train_pinn(x_domain, t_final, params, n_points=1000, n_epochs=100000, s_l=0.0, s_r=1.0):
    """Train a PINN to solve the Buckley-Leverett equation"""
    model = PINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=500, factor=0.5, verbose=True)
    
    # Generate collocation points for PDE residual
    x_collocation = torch.linspace(x_domain[0], x_domain[1], n_points, device=device).view(-1, 1)
    t_collocation = torch.linspace(0, t_final, n_points, device=device).view(-1, 1)
    
    X, T = torch.meshgrid(x_collocation.view(-1), t_collocation.view(-1), indexing='ij')
    x_pde = X.reshape(-1, 1)
    t_pde = T.reshape(-1, 1)
    
    # Generate points for initial condition
    x_ic = torch.linspace(x_domain[0], x_domain[1], n_points, device=device).view(-1, 1)
    t_ic = torch.zeros_like(x_ic, device=device)
    s_ic = torch.zeros_like(x_ic, device=device)
    s_ic[x_ic <= 0] = s_l
    s_ic[x_ic > 0] = s_r
    
    # Generate points for boundary conditions
    t_bc = torch.linspace(0, t_final, n_points, device=device).view(-1, 1)
    x_bc_left = torch.full_like(t_bc, x_domain[0], device=device)
    x_bc_right = torch.full_like(t_bc, x_domain[1], device=device)
    s_bc_left = torch.full_like(t_bc, s_l, device=device)
    s_bc_right = torch.full_like(t_bc, s_r, device=device)
    
    # Training loop
    pbar = tqdm(range(n_epochs))
    losses = []
    pde_losses = []
    ic_losses = []
    bc_losses = []
    
    def flux_torch(s):
        """PyTorch version of flux function"""
        return s**2 / (s**2 + params.M * (1 - s)**2)
    
    for epoch in pbar:
        optimizer.zero_grad()
        
        # Compute PDE residual
        x_pde.requires_grad = True
        t_pde.requires_grad = True
        
        # Forward pass
        s_pred = model(x_pde, t_pde)
        
        # Compute derivatives
        s_t = torch.autograd.grad(
            s_pred, t_pde, 
            grad_outputs=torch.ones_like(s_pred), 
            create_graph=True
        )[0]
        
        s_x = torch.autograd.grad(
            s_pred, x_pde, 
            grad_outputs=torch.ones_like(s_pred), 
            create_graph=True
        )[0]
        
        # Buckley-Leverett PDE: s_t + f(s)_x = 0
        # Computing f(s)_x = f'(s) * s_x
        f_pred = flux_torch(s_pred)
        f_x = torch.autograd.grad(
            f_pred, x_pde, 
            grad_outputs=torch.ones_like(f_pred), 
            create_graph=True
        )[0]
        
        residual = s_t + f_x
        loss_pde = torch.mean(residual**2)
        
        # Initial condition loss
        s_pred_ic = model(x_ic, t_ic)
        loss_ic = torch.mean((s_pred_ic - s_ic)**2)
        
        # Boundary condition loss
        s_pred_bc_left = model(x_bc_left, t_bc)
        s_pred_bc_right = model(x_bc_right, t_bc)
        loss_bc = torch.mean((s_pred_bc_left - s_bc_left)**2) + torch.mean((s_pred_bc_right - s_bc_right)**2)
        
        # Total loss
        loss = loss_pde + 10.0 * loss_ic + 10.0 * loss_bc
        losses.append(loss.item())
        pde_losses.append(loss_pde.item())
        ic_losses.append(loss_ic.item())
        bc_losses.append(loss_bc.item())
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        # Update progress bar
        if epoch % 100 == 0:
            pbar.set_description(f"Loss: {loss.item():.6f}, PDE: {loss_pde.item():.6f}, IC: {loss_ic.item():.6f}, BC: {loss_bc.item():.6f}")
    
    # Plot training losses
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Total Loss')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('PINN Training Total Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(pde_losses, label='PDE Residual')
    plt.plot(ic_losses, label='Initial Condition')
    plt.plot(bc_losses, label='Boundary Condition')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Components')
    plt.title('PINN Training Loss Components')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('pinn_training_loss.png')
    
    return model, {"pde_residuals": pde_losses, "ic_losses": ic_losses, "bc_losses": bc_losses}

# ==================== WENO Implementation ====================
def weno5_reconstruction(v_minus2, v_minus1, v, v_plus1, v_plus2):
    """Fifth-order WENO reconstruction for flux computation"""
    epsilon = 1e-6
    
    # Compute smoothness indicators
    beta0 = 13/12 * (v_minus2 - 2*v_minus1 + v)**2 + 1/4 * (v_minus2 - 4*v_minus1 + 3*v)**2
    beta1 = 13/12 * (v_minus1 - 2*v + v_plus1)**2 + 1/4 * (v_minus1 - v_plus1)**2
    beta2 = 13/12 * (v - 2*v_plus1 + v_plus2)**2 + 1/4 * (3*v - 4*v_plus1 + v_plus2)**2
    
    # Compute nonlinear weights
    d0 = 1/10
    d1 = 6/10
    d2 = 3/10
    
    alpha0 = d0 / (epsilon + beta0)**2
    alpha1 = d1 / (epsilon + beta1)**2
    alpha2 = d2 / (epsilon + beta2)**2
    
    omega0 = alpha0 / (alpha0 + alpha1 + alpha2)
    omega1 = alpha1 / (alpha0 + alpha1 + alpha2)
    omega2 = alpha2 / (alpha0 + alpha1 + alpha2)
    
    # Candidate stencils
    p0 = 1/3 * v_minus2 - 7/6 * v_minus1 + 11/6 * v
    p1 = -1/6 * v_minus1 + 5/6 * v + 1/3 * v_plus1
    p2 = 1/3 * v + 5/6 * v_plus1 - 1/6 * v_plus2
    
    # Final reconstruction
    return omega0 * p0 + omega1 * p1 + omega2 * p2

def solve_weno(x_domain, t_final, params, nx=400, nt=800, s_l=0.0, s_r=1.0):
    """Solve the Buckley-Leverett equation using WENO scheme"""
    # Set up grid
    dx = (x_domain[1] - x_domain[0]) / nx
    dt = t_final / nt
    
    # Check CFL condition
    max_slope = params.flux_derivative(0.5)  # Approximate maximum slope
    cfl = max_slope * dt / dx
    print(f"CFL number: {cfl:.4f} (should be < 1)")
    
    if cfl >= 1.0:
        nt = int(nt * 1.2)  # Increase time steps if CFL condition not satisfied
        dt = t_final / nt
        print(f"Adjusted time steps to {nt}, new dt = {dt}, new CFL = {max_slope * dt / dx:.4f}")
    
    # Initialize solution array
    x = np.linspace(x_domain[0], x_domain[1], nx+1)
    s = np.zeros_like(x)
    s[x <= 0] = s_l
    s[x > 0] = s_r
    
    # Use ghost cells for boundary conditions
    def apply_boundary(s):
        s_with_ghost = np.zeros(len(s) + 4)
        s_with_ghost[2:-2] = s
        s_with_ghost[0:2] = s_l  # Left ghost cells
        s_with_ghost[-2:] = s_r  # Right ghost cells
        return s_with_ghost
    
    # Time stepping loop with progress bar
    pbar = tqdm(total=nt)
    convergence_history = []
    
    # New: track residuals and boundary condition errors
    weno_residuals = []
    weno_bc_errors = []
    
    for n in range(nt):
        s_old = s.copy()
        
        # Apply boundary conditions
        s_bc = apply_boundary(s)
        
        # Compute numerical fluxes using WENO reconstruction
        flux = np.zeros(len(s) + 1)
        
        for j in range(len(flux) - 1):
            idx = j + 2  # Offset for ghost cells
            
            # WENO reconstruction for left and right states
            s_minus = weno5_reconstruction(
                s_bc[idx-3], s_bc[idx-2], s_bc[idx-1], s_bc[idx], s_bc[idx+1]
            )
            s_plus = weno5_reconstruction(
                s_bc[idx+2], s_bc[idx+1], s_bc[idx], s_bc[idx-1], s_bc[idx-2]
            )
            
            # Compute numerical flux (Lax-Friedrichs flux)
            f_minus = params.flux_function(s_minus)
            f_plus = params.flux_function(s_plus)
            alpha = max(abs(params.flux_derivative(s_minus)), abs(params.flux_derivative(s_plus)))
            
            flux[j] = 0.5 * (f_minus + f_plus - alpha * (s_plus - s_minus))
        
        # Update solution using conservative form
        s_new = s.copy()
        for j in range(len(s)):
            s_new[j] = s[j] - dt/dx * (flux[j+1] - flux[j])
        
        # Calculate residual: approximation of s_t + f(s)_x
        residual = np.zeros_like(s)
        for j in range(1, len(s)-1):
            # Finite difference approximation of f(s)_x
            f_j_minus_half = params.flux_function(s[j-1])
            f_j_plus_half = params.flux_function(s[j+1])
            f_x = (f_j_plus_half - f_j_minus_half) / (2*dx)
            
            # s_t approximation
            s_t = (s_new[j] - s[j]) / dt
            
            # Residual of PDE: s_t + f(s)_x = 0
            residual[j] = s_t + f_x
        
        # Mean squared residual
        mean_residual = np.mean(residual**2)
        weno_residuals.append(mean_residual)
        
        # Calculate boundary condition error
        bc_error_left = np.abs(s[0] - s_l)
        bc_error_right = np.abs(s[-1] - s_r)
        weno_bc_errors.append(bc_error_left + bc_error_right)
        
        # Update solution
        s = s_new
        
        # Check convergence
        change = np.max(np.abs(s - s_old))
        convergence_history.append(change)
        pbar.set_description(f"Change: {change:.6f}, Residual: {mean_residual:.6f}")
        pbar.update(1)
    
    pbar.close()
    
    # Plot convergence and residual histories
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(convergence_history, label='Solution Change')
    plt.yscale('log')
    plt.xlabel('Time Steps')
    plt.ylabel('Max Change')
    plt.title('WENO Convergence History')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(weno_residuals, label='PDE Residual')
    plt.plot(weno_bc_errors, label='BC Error')
    plt.yscale('log')
    plt.xlabel('Time Steps')
    plt.ylabel('Error Measures')
    plt.title('WENO Error Metrics')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('weno_convergence.png')
    
    return x, s, {"residuals": weno_residuals, "bc_errors": weno_bc_errors}

# ==================== Main Execution with Enhanced Analysis ====================
def main():
    # Set problem parameters
    x_domain = [-2.0, 2.0]
    t_final = 0.5
    params = BuckleyLeverettParams(M=2.0)
    s_l = 0.0  # Left boundary condition
    s_r = 1.0  # Right boundary condition
    
    # Solve using WENO
    print("Solving using WENO scheme...")
    x_weno, s_weno, weno_metrics = solve_weno(x_domain, t_final, params, nx=4000, nt=10000, s_l=s_l, s_r=s_r)
    
    # Compute exact solution
    print("Computing exact solution...")
    x_exact = np.linspace(x_domain[0], x_domain[1], 1000)
    s_exact = exact_solution(x_exact, t_final, params, s_l=s_l, s_r=s_r)
    
    # Train PINN
    print("Training PINN model...")
    pinn_model, pinn_metrics = train_pinn(x_domain, t_final, params, n_points=100, n_epochs=10000, s_l=s_l, s_r=s_r)
    
    # Evaluate PINN solution
    print("Evaluating PINN solution...")
    x_pinn = torch.linspace(x_domain[0], x_domain[1], 500, device=device).view(-1, 1)
    t_pinn = torch.full_like(x_pinn, t_final, device=device)
    
    with torch.no_grad():
        s_pinn = pinn_model(x_pinn, t_pinn).cpu().numpy()
    
    # Calculate errors
    # Interpolate WENO and PINN solutions to the exact solution grid for error calculation
    weno_interp = interp1d(x_weno, s_weno, kind='linear', bounds_error=False, fill_value=(s_l, s_r))
    s_weno_interp = weno_interp(x_exact)
    
    pinn_interp = interp1d(x_pinn.cpu().numpy().flatten(), s_pinn.flatten(), kind='linear', bounds_error=False, fill_value=(s_l, s_r))
    s_pinn_interp = pinn_interp(x_exact)
    
    l2_error_weno = np.sqrt(np.mean((s_weno_interp - s_exact)**2))
    l2_error_pinn = np.sqrt(np.mean((s_pinn_interp - s_exact)**2))
    l_inf_error_weno = np.max(np.abs(s_weno_interp - s_exact))
    l_inf_error_pinn = np.max(np.abs(s_pinn_interp - s_exact))
    
    print(f"WENO L2 Error: {l2_error_weno:.6f}")
    print(f"PINN L2 Error: {l2_error_pinn:.6f}")
    print(f"WENO L∞ Error: {l_inf_error_weno:.6f}")
    print(f"PINN L∞ Error: {l_inf_error_pinn:.6f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(x_exact, s_exact, 'k-', linewidth=2, label='Exact Solution')
    plt.plot(x_weno, s_weno, 'r--', linewidth=1.5, label=f'WENO (L2: {l2_error_weno:.4f})')
    plt.plot(x_pinn.cpu().numpy(), s_pinn, 'b-.', linewidth=1.5, label=f'PINN (L2: {l2_error_pinn:.4f})')
    plt.xlabel('x')
    plt.ylabel('Saturation (s)')
    plt.title(f'Buckley-Leverett Equation Solutions at t = {t_final}')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(x_exact, np.zeros_like(x_exact), 'k-', linewidth=1)  # Zero line
    plt.plot(x_exact, s_weno_interp - s_exact, 'r--', linewidth=1.5, label=f'WENO Error (L∞: {l_inf_error_weno:.4f})')
    plt.plot(x_exact, s_pinn_interp - s_exact, 'b-.', linewidth=1.5, label=f'PINN Error (L∞: {l_inf_error_pinn:.4f})')
    plt.xlabel('x')
    plt.ylabel('Error (s - s_exact)')
    plt.title('Error Analysis')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('buckley_leverett_comparison.png')
    
    # New: Compare residuals and boundary condition losses
    # Need to make the arrays same length for comparison
    min_len = min(len(weno_metrics["residuals"]), len(pinn_metrics["pde_residuals"]))
    
    # Normalize for comparison (percentages of max value)
    weno_residuals_norm = np.array(weno_metrics["residuals"][:min_len]) / max(weno_metrics["residuals"][:min_len])
    pinn_residuals_norm = np.array(pinn_metrics["pde_residuals"][:min_len]) / max(pinn_metrics["pde_residuals"][:min_len])
    
    weno_bc_norm = np.array(weno_metrics["bc_errors"][:min_len]) / max(weno_metrics["bc_errors"][:min_len])
    pinn_bc_norm = np.array(pinn_metrics["bc_losses"][:min_len]) / max(pinn_metrics["bc_losses"][:min_len])
    
    # Create a comparative plot
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(weno_metrics["residuals"], 'r-', label='WENO')
    plt.plot(pinn_metrics["pde_residuals"], 'b-', label='PINN')
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('PDE Residual (log scale)')
    plt.title('Raw PDE Residual Comparison')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(weno_metrics["bc_errors"], 'r-', label='WENO')
    plt.plot(pinn_metrics["bc_losses"], 'b-', label='PINN')
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('BC Error (log scale)')
    plt.title('Raw Boundary Condition Error Comparison')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(weno_residuals_norm[:min_len], 'r-', label='WENO')
    plt.plot(pinn_residuals_norm[:min_len], 'b-', label='PINN')
    plt.xlabel('Iterations')
    plt.ylabel('Normalized PDE Residual')
    plt.title('Normalized PDE Residual Comparison')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(weno_bc_norm[:min_len], 'r-', label='WENO')
    plt.plot(pinn_bc_norm[:min_len], 'b-', label='PINN')
    plt.xlabel('Iterations')
    plt.ylabel('Normalized BC Error')
    plt.title('Normalized Boundary Condition Error Comparison')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('residual_comparison.png')
    plt.show()
    
    # Print final metrics for comparison
    print("\nFinal Metrics Comparison:")
    print(f"WENO Final PDE Residual: {weno_metrics['residuals'][-1]:.6e}")
    print(f"PINN Final PDE Residual: {pinn_metrics['pde_residuals'][-1]:.6e}")
    print(f"WENO Final BC Error: {weno_metrics['bc_errors'][-1]:.6e}")
    print(f"PINN Final BC Error: {pinn_metrics['bc_losses'][-1]:.6e}")
    print(f"PINN Final IC Error: {pinn_metrics['ic_losses'][-1]:.6e}")

if __name__ == "__main__":
    main()