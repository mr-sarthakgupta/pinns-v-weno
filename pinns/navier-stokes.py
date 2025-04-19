import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

# Enable GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.activation = nn.Tanh()  # As specified in Table 1
        self.loss_function = nn.MSELoss(reduction='mean')
        
        # Initialize network layers
        layer_list = []
        for i in range(len(layers)-1):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
            
            # Apply Xavier initialization as mentioned in the paper
            if i < len(layers)-2:
                nn.init.xavier_normal_(layer_list[i].weight)
                nn.init.zeros_(layer_list[i].bias)
                
        self.layers = nn.ModuleList(layer_list)
    
    def forward(self, x):
        if not torch.is_tensor(x):
            intermediate_x = torch.tensor(x, dtype=torch.float32, device=device, requires_grad=True)
        else:
            init_x = x.to(device)
            init_x.requires_grad = True
            intermediate_x = init_x

        
        # Forward pass through the network
        for i in range(len(self.layers)-1):
            intermediate_x = self.activation(self.layers[i](intermediate_x))
        
        # Final layer without activation (raw outputs)
        intermediate_x = self.layers[-1](intermediate_x)
        
        # Extract outputs: stream function, pressure and temperature
        psi = intermediate_x[:, 0:1]
        p = intermediate_x[:, 1:2]
        theta = intermediate_x[:, 2:3]


        
        # Computing velocity components from stream function using automatic differentiation
        u = grad(psi.sum(), x, create_graph=True)[0][:, 1:2]  # u = ∂ψ/∂y
        v = -grad(psi.sum(), x, create_graph=True)[0][:, 0:1]  # v = -∂ψ/∂x

        return u, v, p, theta, psi

class PINNSolver:
    def __init__(self, re=20, pr=0.71):
        # Reynolds number and Prandtl number
        self.re = re
        self.pr = pr
        
        # Network architecture from Table 1 (6×100 for wavy channel)
        self.network = PINN([2, 100, 100, 100, 100, 100, 100, 3]).to(device)
        
        # Initialize with Adam optimizer followed by L-BFGS as per the paper
        self.adam_optimizer = optim.Adam(self.network.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.adam_optimizer, 'min', patience=5, factor=0.8, verbose=True
        )
    
    def initialize_lbfgs(self):
        """Initialize L-BFGS optimizer after Adam pre-training"""
        self.lbfgs_optimizer = optim.LBFGS(
            self.network.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )
    
    def compute_pde_residual(self, x_interior):
        """Compute PDE residuals at interior collocation points"""
        # x_interior.requires_grad_(True)
        x_interior.requires_grad = True

        
        # Get network outputs
        with torch.inference_mode(mode=False):
            u, v, p, theta, psi = self.network(x_interior)
        
        # Compute gradients for PDE residuals
        # Continuity equation: ∂u/∂x + ∂v/∂y = 0 (Eq. 8)
        u_x = grad(u.sum(), x_interior, create_graph=True)[0][:, 0:1]
        v_y = grad(v.sum(), x_interior, create_graph=True)[0][:, 1:2]
        continuity = u_x + v_y
        
        # x-momentum: u*∂u/∂x + v*∂u/∂y = -∂p/∂x + (∂²u/∂x² + ∂²u/∂y²)/Re (Eq. 9)
        u_y = grad(u.sum(), x_interior, create_graph=True)[0][:, 1:2]
        u_xx = grad(u_x.sum(), x_interior, create_graph=True)[0][:, 0:1]
        u_yy = grad(u_y.sum(), x_interior, create_graph=True)[0][:, 1:2]
        p_x = grad(p.sum(), x_interior, create_graph=True)[0][:, 0:1]
        x_momentum = u * u_x + v * u_y + p_x - (u_xx + u_yy) / self.re
        
        # y-momentum: u*∂v/∂x + v*∂v/∂y = -∂p/∂y + (∂²v/∂x² + ∂²v/∂y²)/Re (.Eq. 10)
        v_x = grad(v.sum(), x_interior, create_graph=True)[0][:, 0:1]
        v_xx = grad(v_x.sum(), x_interior, create_graph=True)[0][:, 0:1]
        v_yy = grad(v_y.sum(), x_interior, create_graph=True)[0][:, 1:2]
        p_y = grad(p.sum(), x_interior, create_graph=True)[0][:, 1:2]
        y_momentum = u * v_x + v * v_y + p_y - (v_xx + v_yy) / self.re
        
        # Energy equation: u*∂θ/∂x + v*∂θ/∂y = (∂²θ/∂x² + ∂²θ/∂y²)/(Re*Pr) (Eq. 11)
        theta_x = grad(theta.sum(), x_interior, create_graph=True)[0][:, 0:1]
        theta_y = grad(theta.sum(), x_interior, create_graph=True)[0][:, 1:2]
        theta_xx = grad(theta_x.sum(), x_interior, create_graph=True)[0][:, 0:1]
        theta_yy = grad(theta_y.sum(), x_interior, create_graph=True)[0][:, 1:2]
        energy = u * theta_x + v * theta_y - (theta_xx + theta_yy) / (self.re * self.pr)
        
        return continuity, x_momentum, y_momentum, energy
    
    def compute_boundary_residual(self, x_wall, u_wall, v_wall, theta_wall,
                                  x_inlet, u_inlet, v_inlet, p_inlet, theta_inlet,
                                  x_outlet, p_outlet):
        """Compute residuals at boundary points"""
        # Wall boundary conditions: u=0, v=0, θ=1 (no-slip & const. temperature)
        u_wall_pred, v_wall_pred, _, theta_wall_pred, _ = self.network(x_wall)
        wall_residual_u = u_wall_pred - u_wall
        wall_residual_v = v_wall_pred - v_wall
        wall_residual_theta = theta_wall_pred - theta_wall
        
        # Inlet boundary conditions: u=1, v=0, p=0, θ=0
        u_inlet_pred, v_inlet_pred, p_inlet_pred, theta_inlet_pred, _ = self.network(x_inlet)
        inlet_residual_u = u_inlet_pred - u_inlet
        inlet_residual_v = v_inlet_pred - v_inlet
        inlet_residual_p = p_inlet_pred - p_inlet
        inlet_residual_theta = theta_inlet_pred - theta_inlet
        
        # Outlet boundary condition: p=0, ∂u/∂x=0, ∂v/∂x=0
        _, _, p_outlet_pred, _, _ = self.network(x_outlet)
        x_outlet.requires_grad_(True)
        u_outlet_pred, v_outlet_pred, _, _, _ = self.network(x_outlet)
        u_x_outlet = grad(u_outlet_pred.sum(), x_outlet, create_graph=True)[0][:, 0:1]
        v_x_outlet = grad(v_outlet_pred.sum(), x_outlet, create_graph=True)[0][:, 0:1]
        outlet_residual_p = p_outlet_pred - p_outlet
        outlet_residual_u_x = u_x_outlet
        outlet_residual_v_x = v_x_outlet
        
        return (wall_residual_u, wall_residual_v, wall_residual_theta,
                inlet_residual_u, inlet_residual_v, inlet_residual_p, inlet_residual_theta,
                outlet_residual_p, outlet_residual_u_x, outlet_residual_v_x)
    
    def compute_loss(self, x_interior, x_wall, u_wall, v_wall, theta_wall,
                    x_inlet, u_inlet, v_inlet, p_inlet, theta_inlet,
                    x_outlet, p_outlet, beta=1.0):
        """Compute total loss as per equations 19-25"""
        # PDE residuals
        continuity, x_momentum, y_momentum, energy = self.compute_pde_residual(x_interior)
        
        # Boundary residuals
        (wall_residual_u, wall_residual_v, wall_residual_theta,
         inlet_residual_u, inlet_residual_v, inlet_residual_p, inlet_residual_theta,
         outlet_residual_p, outlet_residual_u_x, outlet_residual_v_x) = self.compute_boundary_residual(
            x_wall, u_wall, v_wall, theta_wall,
            x_inlet, u_inlet, v_inlet, p_inlet, theta_inlet,
            x_outlet, p_outlet
        )
        
        # PDE losses L_g1, L_g2, L_g3, L_g4, L_g5 (Eqs. 14-18)
        loss_g1 = torch.mean(continuity**2)
        loss_g2 = torch.mean(x_momentum**2)
        loss_g3 = torch.mean(y_momentum**2)
        loss_g4 = torch.mean(energy**2)
        loss_g5 = torch.mean((continuity + x_momentum + y_momentum)**2)
        
        # Total PDE loss (Eq. 19)
        loss_g = loss_g1 + loss_g2 + loss_g3 + loss_g4 + loss_g5
        
        # Boundary losses (Eqs. 20-22)
        loss_bc_wall = torch.mean(wall_residual_u**2) + torch.mean(wall_residual_v**2) + torch.mean(wall_residual_theta**2)
        loss_bc_inlet = torch.mean(inlet_residual_u**2) + torch.mean(inlet_residual_v**2) + \
                        torch.mean(inlet_residual_p**2) + torch.mean(inlet_residual_theta**2)
        loss_bc_outlet = torch.mean(outlet_residual_p**2) + torch.mean(outlet_residual_u_x**2) + torch.mean(outlet_residual_v_x**2)
        
        # Total boundary loss (Eq. 24)
        loss_bc = loss_bc_wall + loss_bc_inlet + loss_bc_outlet
        
        # Total loss (Eq. 25)
        total_loss = loss_g + beta * loss_bc
        
        return total_loss, loss_g, loss_bc
    
    def generate_collocation_points(self, n_interior=62000, n_boundary=8000, symmetric=True):
        """Generate collocation points using Latin Hypercube Sampling as per section 2.4"""
        if symmetric:
            # For symmetric wavy channel as per Figure 3(c)
            # Implementation of equations 27-28 for wavy channel geometry
            L = 8.0  # Module length
            B = 2 * np.pi / L  # Wave number
            a = 1.0  # Amplitude
            phi = 0.0  # Phase shift
            H_min = 1.0  # Minimum channel height
            H_max = 3.0  # Maximum channel height
            
            # Generate random interior points
            x_interior_raw = np.random.uniform(0, L, size=(n_interior, 1))
            y_interior_raw = np.random.uniform(-1.5, 1.5, size=(n_interior, 1))
            
            # Upper and lower wall functions
            def y_upper(x):
                return a * np.sin(B * x + np.pi/2) + (H_max/2 + a)
                
            def y_lower(x):
                return a * np.sin(B * x - np.pi/2 + phi) - (H_max/2 + a)
            
            # Filter points to be inside the channel
            upper_bounds = y_upper(x_interior_raw)
            lower_bounds = y_lower(x_interior_raw)
            valid_indices = np.logical_and(y_interior_raw < upper_bounds, y_interior_raw > lower_bounds)
            valid_indices = valid_indices.flatten()
            
            x_interior = np.column_stack((x_interior_raw[valid_indices], y_interior_raw[valid_indices]))
            
            # Ensure we have enough interior points after filtering
            while x_interior.shape[0] < n_interior:
                n_additional = n_interior - x_interior.shape[0]
                x_additional_raw = np.random.uniform(0, L, size=(n_additional*2, 1))
                y_additional_raw = np.random.uniform(-1.5, 1.5, size=(n_additional*2, 1))
                
                upper_bounds = y_upper(x_additional_raw)
                lower_bounds = y_lower(x_additional_raw)
                valid_indices = np.logical_and(y_additional_raw < upper_bounds, y_additional_raw > lower_bounds)
                valid_indices = valid_indices.flatten()
                
                new_points = np.column_stack((x_additional_raw[valid_indices], y_additional_raw[valid_indices]))
                x_interior = np.vstack((x_interior, new_points[:min(n_additional, new_points.shape[0])]))
            
            x_interior = x_interior[:n_interior]
            
            # Generate wall boundary points (more concentrated near walls)
            n_wall = n_boundary // 2
            x_wall_raw = np.random.uniform(0, L, size=n_wall)
            
            # Upper wall points
            x_upper_wall = x_wall_raw
            y_upper_wall = y_upper(x_upper_wall.reshape(-1, 1)).flatten()
            
            # Lower wall points
            x_lower_wall = x_wall_raw
            y_lower_wall = y_lower(x_lower_wall.reshape(-1, 1)).flatten()
            
            x_wall = np.column_stack((
                np.concatenate((x_upper_wall, x_lower_wall)),
                np.concatenate((y_upper_wall, y_lower_wall))
            ))
            
            # Generate inlet/outlet boundary points
            n_inout = n_boundary - n_wall
            n_inlet = n_inout // 2
            n_outlet = n_inout - n_inlet
            
            # Inlet points (x=0)
            y_inlet_raw = np.random.uniform(y_lower(np.array([[0]]))[0][0], y_upper(np.array([[0]]))[0][0], size=n_inlet)
            x_inlet = np.zeros_like(y_inlet_raw)
            x_inlet = np.column_stack((x_inlet, y_inlet_raw))
            
            # Outlet points (x=L)
            y_outlet_raw = np.random.uniform(y_lower(np.array([[L]]))[0][0], y_upper(np.array([[L]]))[0][0], size=n_outlet)
            x_outlet = np.ones_like(y_outlet_raw) * L
            x_outlet = np.column_stack((x_outlet, y_outlet_raw))
            
        else:
            # For asymmetric wavy channel as per Figure 3(d)
            # Similar implementation but with asymmetric geometry
            # This would need the phi parameter to be non-zero
            pass
        
        # Convert to PyTorch tensors
        x_interior_tensor = torch.tensor(x_interior, dtype=torch.float32, device=device)
        x_wall_tensor = torch.tensor(x_wall, dtype=torch.float32, device=device)
        x_inlet_tensor = torch.tensor(x_inlet, dtype=torch.float32, device=device)
        x_outlet_tensor = torch.tensor(x_outlet, dtype=torch.float32, device=device)
        
        # Boundary conditions
        u_wall = torch.zeros(x_wall_tensor.shape[0], 1, device=device)
        v_wall = torch.zeros(x_wall_tensor.shape[0], 1, device=device)
        theta_wall = torch.ones(x_wall_tensor.shape[0], 1, device=device)
        
        u_inlet = torch.ones(x_inlet_tensor.shape[0], 1, device=device)
        v_inlet = torch.zeros(x_inlet_tensor.shape[0], 1, device=device)
        p_inlet = torch.zeros(x_inlet_tensor.shape[0], 1, device=device)
        theta_inlet = torch.zeros(x_inlet_tensor.shape[0], 1, device=device)
        
        p_outlet = torch.zeros(x_outlet_tensor.shape[0], 1, device=device)
        
        return (x_interior_tensor, x_wall_tensor, u_wall, v_wall, theta_wall, 
                x_inlet_tensor, u_inlet, v_inlet, p_inlet, theta_inlet, 
                x_outlet_tensor, p_outlet)
    
    def train_adam(self, n_epochs=10000, beta=1.0):
        """Train with Adam optimizer"""
        # Generate collocation points
        (x_interior, x_wall, u_wall, v_wall, theta_wall,
         x_inlet, u_inlet, v_inlet, p_inlet, theta_inlet,
         x_outlet, p_outlet) = self.generate_collocation_points()
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        max_patience = 20  # Stop early if no improvement for 20 consecutive epochs
        
        for epoch in range(n_epochs):
            self.adam_optimizer.zero_grad()
            
            loss, loss_g, loss_bc = self.compute_loss(
                x_interior, x_wall, u_wall, v_wall, theta_wall,
                x_inlet, u_inlet, v_inlet, p_inlet, theta_inlet,
                x_outlet, p_outlet, beta
            )
            
            loss.backward()
            self.adam_optimizer.step()
            self.scheduler.step(loss)
            
            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.6e}, PDE Loss = {loss_g.item():.6e}, BC Loss = {loss_bc.item():.6e}")
            
            # Check for early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter > max_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        return best_loss
    
    def train_lbfgs(self, beta=1.0):
        """Train with L-BFGS optimizer after Adam"""
        # Generate collocation points
        (x_interior, x_wall, u_wall, v_wall, theta_wall,
         x_inlet, u_inlet, v_inlet, p_inlet, theta_inlet,
         x_outlet, p_outlet) = self.generate_collocation_points()
        
        # Initialize L-BFGS
        self.initialize_lbfgs()
        
        # Store training data for closure
        self.train_data = (x_interior, x_wall, u_wall, v_wall, theta_wall,
                         x_inlet, u_inlet, v_inlet, p_inlet, theta_inlet,
                         x_outlet, p_outlet, beta)
        
        # L-BFGS optimization
        def closure():
            self.lbfgs_optimizer.zero_grad()
            
            loss, loss_g, loss_bc = self.compute_loss(*self.train_data)
            
            loss.backward()
            print(f"L-BFGS step: Loss = {loss.item():.6e}, PDE Loss = {loss_g.item():.6e}, BC Loss = {loss_bc.item():.6e}")
            
            return loss
        
        self.lbfgs_optimizer.step(closure)
        
        # Final loss calculation
        # with torch.no_grad():
        final_loss, final_loss_g, final_loss_bc = self.compute_loss(*self.train_data)
        
        print(f"Final loss after L-BFGS: {final_loss.item():.6e}")
        return final_loss.item()
    
    def train(self, adam_epochs=5000, beta=1.0):
        """Full training process: Adam followed by L-BFGS"""
        print("Starting training with Adam optimizer...")
        adam_loss = self.train_adam(adam_epochs, beta)
        
        print("\nFine-tuning with L-BFGS optimizer...")
        lbfgs_loss = self.train_lbfgs(beta)
        
        print(f"\nTraining complete. Final loss: {lbfgs_loss:.6e}")
        return lbfgs_loss
    
    def predict(self, x_test):
        """Make predictions at test points"""
        self.network.eval()
        # with torch.no_grad():
        if not torch.is_tensor(x_test):
            x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
        
        u, v, p, theta, psi = self.network(x_test)
        
        return u.detach().cpu().numpy(), v.detach().cpu().numpy(), p.detach().cpu().numpy(), theta.detach().cpu().numpy(), psi.detach().cpu().numpy()
    
    def validate_with_cfd(self, re=20):
        """Validate results against CFD data as in section 3.1"""
        # Generate points for a plane channel case (Fig 4)
        L = 1.0  # Channel length
        H = 0.4  # Channel height
        nx, ny = 100, 50
        x = np.linspace(0, L, nx)
        y = np.linspace(-H/2, H/2, ny)
        X, Y = np.meshgrid(x, y)
        xy = np.stack([X.flatten(), Y.flatten()], axis=1)
        
        # Make predictions
        u_pred, v_pred, p_pred, theta_pred, psi_pred = self.predict(xy)
        
        # Reshape for plotting
        U = u_pred.reshape(ny, nx)
        V = v_pred.reshape(ny, nx)
        P = p_pred.reshape(ny, nx)
        PSI = psi_pred.reshape(ny, nx)
        
        # Create a more comprehensive validation plot
        plt.figure(figsize=(16, 12))
        
        # Plot velocity heatmap
        plt.subplot(2, 2, 1)
        im = plt.pcolormesh(X, Y, U, cmap='rainbow', shading='auto')
        plt.colorbar(im, label='u-velocity')
        plt.title(f'Velocity Profile for Plane Channel at Re={re}')
        plt.xlabel('x')
        plt.ylabel('y')
        
        # Plot pressure heatmap
        plt.subplot(2, 2, 2)
        im = plt.pcolormesh(X, Y, P, cmap='rainbow', shading='auto')
        plt.colorbar(im, label='Pressure')
        plt.title('Pressure Field')
        plt.xlabel('x')
        plt.ylabel('y')
        
        # Plot streamwise velocity profiles at different x-locations
        plt.subplot(2, 2, 3)
        x_locs = [0.2, 0.4, 0.6, 0.8]
        for x_loc in x_locs:
            idx = np.argmin(np.abs(x - x_loc))
            plt.plot(U[:, idx], y, label=f'x={x_loc}')
        
        # Add analytical Poiseuille flow solution for comparison
        u_analytical = 1.5 * (1 - (2*y/H)**2)
        plt.plot(u_analytical, y, 'k--', label='Analytical')
        
        plt.xlabel('u-velocity')
        plt.ylabel('y')
        plt.title('Velocity Profiles at Different x-locations')
        plt.legend()
        plt.grid(True)
        
        # Plot streamlines
        plt.subplot(2, 2, 4)
        plt.streamplot(X, Y, U, V, density=1.5, color='b', linewidth=0.5)
        plt.title('Streamlines')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('pinn_validation_comprehensive.png')
        plt.show()
        
        # Calculate error compared to analytical solution
        u_analytical_reshaped = np.tile(u_analytical, (nx, 1)).T
        mae = np.mean(np.abs(U - u_analytical_reshaped))
        rmse = np.sqrt(np.mean((U - u_analytical_reshaped)**2))

        print(f"MAE vs. analytical solution: {mae:.6f}")
        print(f"RMSE vs. analytical solution: {rmse:.6f}")
        
        return U, V, P, PSI


# Main execution
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create PINN solver for Re=20 case
    print("Initializing PINN solver for plane channel flow...")
    solver = PINNSolver(re=20)
    
    # Train the model
    solver.train(adam_epochs=3000)
    # solver.train(adam_epochs=3)
    
    # Validate against CFD data
    print("Validating against benchmark case...")
    solver.validate_with_cfd()
    
    print("All done!")