import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Define domain parameters
x_min, x_max = 0.0, 1.0
t_min, t_max = 0.0, 1
N_collocation = 10_00_000
N_boundary = 10000
N_initial = 10000

# Multiphase parameters
class MultiphaseParams:
    def __init__(self, trainable=False):
        # End-point relative permeability values
        self.k0rg = nn.Parameter(torch.tensor(0.7), requires_grad=trainable)
        self.k0rw = nn.Parameter(torch.tensor(1.0), requires_grad=trainable)
        
        # Relative permeability exponents
        self.ng = nn.Parameter(torch.tensor(2.0), requires_grad=False)
        self.nw = nn.Parameter(torch.tensor(2.0), requires_grad=False)
        
        # Residual saturations
        self.Sgr = nn.Parameter(torch.tensor(0.0), requires_grad=trainable)
        self.Swr = nn.Parameter(torch.tensor(0.2), requires_grad=trainable)
        
        # Viscosities
        self.mu_g = nn.Parameter(torch.tensor(0.02), requires_grad=False)  # or 0.2 for large mobility ratio
        self.mu_w = nn.Parameter(torch.tensor(1.0), requires_grad=False)

# Neural network architecture
class PINNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation):
        super().__init__()
        self.activation = activation
        self.layer = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.layer.weight)
        nn.init.zeros_(self.layer.bias)
        
    def forward(self, x):
        return self.activation(self.layer(x))

class PINN(nn.Module):
    def __init__(self, hidden_layers=6, hidden_nodes=20, activation=nn.Tanh(), trainable_params=False):
        super().__init__()
        self.activation = activation
        self.params = MultiphaseParams(trainable=trainable_params)
        
        # Neural network layers
        self.input_layer = PINNBlock(2, hidden_nodes, activation)  # Input: (x, t)
        self.hidden_layers = nn.Sequential(*[
            PINNBlock(hidden_nodes, hidden_nodes, activation)
            for _ in range(hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_nodes, 1)  # Output: Sw (water saturation)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        
    def forward(self, x):
        """x is a tensor with shape (batch_size, 2) containing (x_D, t_D) pairs"""
        y = self.input_layer(x)
        y = self.hidden_layers(y)
        sw = self.output_layer(y)
        # Ensure saturation is bounded between Swr and 1
        # return torch.sigmoid(sw) * (1 - self.params.Swr) + self.params.Swr
        return sw
    
    def relative_permeability(self, sw):
        # Calculate normalized water saturation
        sw_norm = (sw - self.params.Swr) / (1 - self.params.Sgr - self.params.Swr)
        # sw_norm = torch.clip(sw_norm, 0.0, 1.0)
        
        # Corey-type relative permeability functions
        krw = self.params.k0rw * (sw_norm ** self.params.nw)
        sg_norm = 1.0 - sw_norm
        krg = self.params.k0rg * (sg_norm ** self.params.ng)
        
        return krw, krg
    
    def fractional_flow(self, sw):
        """Calculate gas fractional flow function"""
        krw, krg = self.relative_permeability(sw)
        
        # Calculate mobility ratio
        mobility_ratio = (krw * self.params.mu_g) / (krg * self.params.mu_w)
        
        # Calculate gas fractional flow
        fg = 1.0 / (1.0 + mobility_ratio)
        
        return fg
    
    def dfg_dsw(self, sw):
        """Calculate derivative of gas fractional flow function w.r.t water saturation"""
        sw_detached = sw.detach().requires_grad_(True)
        fg = self.fractional_flow(sw_detached)
        dfg = torch.autograd.grad(
            fg, sw_detached, 
            grad_outputs=torch.ones_like(fg),
            create_graph=True
        )[0]
        return dfg

# # Analytical solution for comparison
# def analytical_solution(x_vals, t_val, params):
#     """
#     Compute a more realistic analytical solution for the Buckley-Leverett equation
    
#     Parameters:
#     x_vals: array of x positions
#     t_val: specific time value
#     params: MultiphaseParams object with flow parameters
    
#     Returns:
#     Array of water saturation values
#     """
#     # Convert inputs to numpy if they're torch tensors
#     if isinstance(x_vals, torch.Tensor):
#         x_vals = x_vals.cpu().numpy()
    
#     # Initialize saturation array with initial condition (Sw = 1.0)
#     sw = np.ones_like(x_vals)
    
#     # Calculate simplified fractional flow curve and its derivative at a range of saturations
#     sw_range = np.linspace(params.Swr.item(), 1.0, 1000)
    
#     # Calculate normalized saturations for Corey model
#     sw_norm = (sw_range - params.Swr.item()) / (1.0 - params.Sgr.item() - params.Swr.item())
#     sw_norm = np.clip(sw_norm, 0.0, 1.0)
    
#     # Corey relative permeability model
#     krw = params.k0rw.item() * (sw_norm ** params.nw.item())
#     krg = params.k0rg.item() * ((1.0 - sw_norm) ** params.ng.item())
    
#     # Calculate fractional flow of water
#     fw = krw / params.mu_w.item() / (krw / params.mu_w.item() + krg / params.mu_g.item())
    
#     # Calculate derivative of fw with respect to Sw (using finite differences)
#     dfw_dsw = np.gradient(fw, sw_range)
    
#     # For Buckley-Leverett, the wave velocity is dfw/dsw
#     # Find the maximum dfw/dsw (shock front)
#     max_deriv_idx = np.argmax(dfw_dsw)
#     shock_sw = sw_range[max_deriv_idx]
#     shock_velocity = dfw_dsw[max_deriv_idx]
    
#     # Adjust shock velocity for simplified cases
#     if params.mu_g.item() > 0.1:  # Large mobility ratio case
#         shock_velocity = 0.3  # Approximate value
#     else:  # Small mobility ratio case
#         shock_velocity = 0.5  # Approximate value
    
#     shock_position = t_val * shock_velocity
    
#     # Apply shock front solution: behind shock front, Sw = Swr (or shock_sw for more realism)
#     sw[x_vals <= shock_position] = shock_sw
    
#     # Near the injection point, water saturation approaches 1-Sgr
#     transition_width = 0.05
#     near_injection = x_vals <= transition_width * shock_position
#     if np.any(near_injection):
#         sw[near_injection] = params.Swr.item()
    
#     return sw

# Generate training points
def generate_training_points():
    # Collocation points (randomly distributed in domain)
    x_collocation = torch.rand(N_collocation, 1) * (x_max - x_min) + x_min
    t_collocation = torch.rand(N_collocation, 1) * (t_max - t_min) + t_min
    collocation_points = torch.cat([x_collocation, t_collocation], dim=1).to(DEVICE)
    
    # Boundary points (x=0, all t)
    t_boundary = torch.linspace(t_min, t_max, N_boundary).reshape(-1, 1)
    x_boundary = torch.zeros_like(t_boundary)
    boundary_points = torch.cat([x_boundary, t_boundary], dim=1).to(DEVICE)
    
    # Initial points (t=0, all x)
    x_initial = torch.linspace(x_min, x_max, N_initial).reshape(-1, 1)
    t_initial = torch.zeros_like(x_initial)
    initial_points = torch.cat([x_initial, t_initial], dim=1).to(DEVICE)
    
    return collocation_points, boundary_points, initial_points

def compute_residual(model, x_t):
    """Compute the PDE residual at collocation points"""
    x, t = x_t[:, 0:1], x_t[:, 1:2]

    # We need gradients w.r.t inputs
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    # Get water saturation prediction
    inputs = torch.cat([x, t], dim=1)
    sw = model(inputs)
    
    # Compute derivatives using autograd
    dsw_dt = torch.autograd.grad(
        sw, t, 
        grad_outputs=torch.ones_like(sw),
        create_graph=True
    )[0]
    
    dsw_dx = torch.autograd.grad(
        sw, x, 
        grad_outputs=torch.ones_like(sw),
        create_graph=True
    )[0]
    
    # Compute gas saturation
    # sg = 1.0 - sw
    
    # Compute dfg/dsw (derivative of fractional flow function)
    dfg_dsw = model.dfg_dsw(sw)
    
    # Compute the PDE residual: dsw/dt + dfg/dsw * dsw/dx = 0
    # Note: dsw/dt = -dsg/dt, dsw/dx = -dsg/dx and dfg/dsw = -dfw/dsw
    residual = dsw_dt - dfg_dsw * dsw_dx
    
    # Option to add diffusion term as in the paper
    # diffusion_term = lambda * d²sw/dx²
    # Second derivative would be computed using autograd grad twice
    
    return residual

def train_model(model, with_observed_data=False, with_diffusion=False, 
                mobility_ratio="small", num_epochs=50000, casename=None):
    """Train the PINN model"""
    # Set gas viscosity based on mobility ratio
    if mobility_ratio == "large":
        model.params.mu_g.data = torch.tensor(0.2)
    else:  # small
        model.params.mu_g.data = torch.tensor(0.02)
    
    # Generate training points
    collocation_points, boundary_points, initial_points = generate_training_points()

    # print(collocation_points.shape, boundary_points.shape, initial_points.shape)
    
    # Generate observed data (if using)
    observed_data = None
    if with_observed_data:
        num_obs = 10000
        num_t = 100
        # This would be replaced with actual observed data
        # For now, we'll generate synthetic data from a simplified analytical solution
        x_obs = torch.linspace(x_min, x_max, num_obs).to(DEVICE)
        t_obs = torch.tensor([(1/num_obs) * i for i in range(num_t)]).to(DEVICE)
        
        # Create grid of observed points
        x_obs_grid = x_obs.repeat(len(t_obs))
        t_obs_grid = torch.repeat_interleave(t_obs, len(x_obs))
        observed_points = torch.stack([x_obs_grid, t_obs_grid], dim=1)
        
        # Generate "observed" values (simplified for this example)
        params = MultiphaseParams(trainable=False)
        if mobility_ratio == "large":
            params.mu_g = nn.Parameter(torch.tensor(0.2), requires_grad=False)
        
        sw_obs = []
        for t_val in t_obs:
            for x_val in x_obs:
                if x_val <= 0.5 * t_val:  # Simplified shock wave
                    sw_obs.append(params.Swr.item())
                else:
                    sw_obs.append(1.0)
        
        observed_values = torch.tensor(sw_obs, dtype=torch.float32).reshape(-1, 1).to(DEVICE)
        observed_data = (observed_points, observed_values)
    
    # Setup optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=0, factor=0.5, patience=50, verbose=True)
    
    # Training loop
    running_loss = []
    
    with tqdm(total=num_epochs, unit="epoch") as pbar:
        for epoch in range(num_epochs):
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute PDE residual loss at collocation points
            residual = compute_residual(model, collocation_points)
            residual_loss = torch.mean(residual**2)
            
            # Compute initial condition loss
            sw_initial = model(initial_points)
            initial_loss = torch.mean((sw_initial - 1.0)**2)  # Initial condition: Sw = 1.0 everywhere
            
            # Compute boundary condition loss
            sw_boundary = model(boundary_points)
            boundary_loss = torch.mean((sw_boundary - model.params.Swr)**2)  # Boundary condition: Sw = Swr at x=0
            
            # Combine losses
            loss = residual_loss + initial_loss + boundary_loss
            
            # Add observed data loss if provided
            if with_observed_data:
                obs_points, obs_values = observed_data
                sw_obs_pred = model(obs_points)
                observed_loss = torch.mean((sw_obs_pred - obs_values)**2)
                loss += observed_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            # scheduler.step(loss.item())
            scheduler.step(loss)
            
            # Record loss
            running_loss.append(loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4e}",
                "lr": f"{optimizer.param_groups[0]['lr']:.3e}"
            })
            pbar.update(1)

    # optimizer_2 = torch.optim.LBFGS(model.parameters(), lr=1e-1, max_iter=10000, history_size=10, line_search_fn='strong_wolfe')

    # with tqdm(total=num_epochs, unit="epoch") as pbar:
    #     for epoch in range(num_epochs):
            
    #         # Backward pass and optimize
    #         def closure():
    #             optimizer_2.zero_grad()

    #             # Compute PDE residual loss at collocation points
    #             residual = compute_residual(model, collocation_points)
    #             residual_loss = torch.mean(residual**2)
                
    #             # Compute initial condition loss
    #             sw_initial = model(initial_points)
    #             initial_loss = torch.mean((sw_initial - 1.0)**2)  # Initial condition: Sw = 1.0 everywhere
                
    #             # Compute boundary condition loss
    #             sw_boundary = model(boundary_points)
    #             boundary_loss = torch.mean((sw_boundary - model.params.Swr)**2)  # Boundary condition: Sw = Swr at x=0
                
    #             # Combine losses
    #             loss = residual_loss + initial_loss + boundary_loss
                
    #             # Add observed data loss if provided
    #             if with_observed_data:
    #                 obs_points, obs_values = observed_data
    #                 sw_obs_pred = model(obs_points)
    #                 observed_loss = torch.mean((sw_obs_pred - obs_values)**2)
    #                 loss += observed_loss
        
    #             loss.backward()
    #             return loss
            
    #         optimizer_2.step(closure)
            
    #         # Record loss
    #         running_loss.append(loss.item())
            
    #         # Update progress bar
    #         pbar.set_postfix({
    #             "loss": f"{loss.item():.4e}",
    #             "lr": f"{optimizer_2.param_groups[0]['lr']:.3e}"
    #         })
    #         pbar.update(1)
    
    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.plot(running_loss)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    os.makedirs(f'figures/{casename}/', exist_ok=True)
    plt.savefig(f'figures/{casename}/loss_history_{mobility_ratio}_mr_diffusion={with_diffusion}_with_observed_data={with_observed_data}.png')
    
    return running_loss


def evaluate_model(model, plot_times, with_observed_data, with_diffusion, mobility_ratio="small", casename=None):
    """Evaluate and visualize the model results with comparison to analytical solution"""
    # Create grid for evaluation
    x_eval = torch.linspace(x_min, x_max, 100).to(DEVICE)
    
    plt.figure(figsize=(14, 10))
    
    # Define line styles and colors for better visualization
    pred_styles = {'linestyle': '-', 'linewidth': 2}
    actual_styles = {'linestyle': '--', 'linewidth': 2, 'alpha': 0.7}
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Get parameters for analytical solution
    params = MultiphaseParams(trainable=False)
    if mobility_ratio == "large":
        params.mu_g.data = torch.tensor(0.2)
    
    # For each time point
    for i, t_val in enumerate(plot_times):
        color = colors[i % len(colors)]
        
        # Create input points at time t_val
        t_eval = torch.ones_like(x_eval) * t_val
        x_t_eval = torch.stack([x_eval, t_eval], dim=1)
        
        # Get model predictions
        with torch.no_grad():
            sw_pred = model(x_t_eval).cpu().numpy()
        
        # Plot model predictions
        plt.plot(x_eval.cpu().numpy(), sw_pred, label=f'Prediction t={t_val:.2f}', 
                 color=color, **pred_styles)
        
        # Generate analytical/actual solution
        # Simple shock-front approximation for Buckley-Leverett
        x_np = x_eval.cpu().numpy()
        sw_actual = np.ones_like(x_np)
        
        # Simplified shock velocity based on mobility ratio
        if mobility_ratio == "large":
            shock_velocity = 0.3
        else:
            shock_velocity = 0.5
            
        shock_position = t_val * shock_velocity
        sw_actual[x_np <= shock_position] = params.Swr.item()
        
        # Plot analytical/actual solution
        plt.plot(x_np, sw_actual, label=f'Actual t={t_val:.2f}', 
                 color=color, **actual_styles)
    
    # Add model parameters to the plot
    model_params_text = (
        f"Model Parameters:\n"
        f"k0rg = {model.params.k0rg.item():.3f}, "
        f"k0rw = {model.params.k0rw.item():.3f}\n"
        f"Swr = {model.params.Swr.item():.3f}, "
        f"Sgr = {model.params.Sgr.item():.3f}\n"
        f"μg = {model.params.mu_g.item():.3f}, "
        f"μw = {model.params.mu_w.item():.3f}"
    )
    plt.annotate(model_params_text, xy=(0.02, 0.02), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
    
    plt.xlabel('Dimensionless Distance (x_D)', fontsize=12)
    plt.ylabel('Water Saturation (Sw)', fontsize=12)
    plt.title(f'Water Saturation Profiles - {mobility_ratio.capitalize()} Mobility Ratio\n'
              f'{"With" if with_observed_data else "Without"} Observed Data, '
              f'{"With" if with_diffusion else "Without"} Diffusion', 
              fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True)
    plt.ylim(0, 1.05)
    
    # Save figure with descriptive filename
    os.makedirs(f'figures/{casename}', exist_ok=True)
    save_path = f'figures/{casename}/comparison_{mobility_ratio}_mr_diff={with_diffusion}_obs={with_observed_data}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Comparison plot saved to {save_path}")
    plt.show()
    
    # Add a second figure to show the error between predicted and actual
    plt.figure(figsize=(14, 6))
    
    for i, t_val in enumerate(plot_times):
        color = colors[i % len(colors)]
        
        # Create input points at time t_val
        t_eval = torch.ones_like(x_eval) * t_val
        x_t_eval = torch.stack([x_eval, t_eval], dim=1)
        
        # Get model predictions
        with torch.no_grad():
            sw_pred = model(x_t_eval).cpu().numpy()
        
        # Generate analytical/actual solution
        x_np = x_eval.cpu().numpy()
        sw_actual = np.ones_like(x_np)
        
        # Simplified shock velocity based on mobility ratio
        if mobility_ratio == "large":
            shock_velocity = 0.3
        else:
            shock_velocity = 0.5
            
        shock_position = t_val * shock_velocity
        sw_actual[x_np <= shock_position] = params.Swr.item()
        
        # Compute error
        error = np.abs(sw_pred.flatten() - sw_actual)
        
        # Plot error
        plt.plot(x_np, error, label=f't={t_val:.2f}', color=color)
    
    plt.xlabel('Dimensionless Distance (x_D)', fontsize=12)
    plt.ylabel('Absolute Error |Sw_pred - Sw_actual|', fontsize=12)
    plt.title(f'Error Analysis - {mobility_ratio.capitalize()} Mobility Ratio', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True)
    
    # Save error figure
    os.makedirs(f'figures/{casename}', exist_ok=True)
    error_save_path = f'figures/{casename}/error_{mobility_ratio}_mr_diff={with_diffusion}_obs={with_observed_data}.png'
    plt.savefig(error_save_path, dpi=300, bbox_inches='tight')
    
    print(f"Error analysis plot saved to {error_save_path}")
    plt.show()

def main():
    # Create dictionary to store models and results
    models = {
        "1": {"small": None, "large": None, "small_loss": None, "large_loss": None},
        "2": {"small": None, "large": None, "small_loss": None, "large_loss": None},
        "3": {"small": None, "large": None, "small_loss": None, "large_loss": None},
        "4": {"small": None, "large": None, "small_loss": None, "large_loss": None},
    }
    
    # Reduced number of epochs for demonstration
    num_epochs = 20000  # Reduce for testing, use 50000 for full training
    
    # Case 1: No observed data, non-trainable parameters, no diffusion
    print("Case 1: No observed data, non-trainable parameters, no diffusion")
    
    model_case1_small = PINN(trainable_params=False).to(DEVICE)
    
    casename = f"no_observed_data__no_diffusion__no_trainable__small_mobility__{num_epochs}"
    loss_case1_small = train_model(model_case1_small, with_observed_data=False, 
                           with_diffusion=False, mobility_ratio="small", num_epochs=num_epochs, casename=casename)
    evaluate_model(model_case1_small, plot_times=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 
                  mobility_ratio="small", with_observed_data=False, with_diffusion=False, casename=casename)
    models["1"]["small"] = model_case1_small
    models["1"]["small_loss"] = loss_case1_small
    
    model_case1_large = PINN(trainable_params=False).to(DEVICE)
    casename = f"no_observed_data__no_diffusion__no_trainable__large_mobility__{num_epochs}"
    loss_case1_large = train_model(model_case1_large, with_observed_data=False, 
                           with_diffusion=False, mobility_ratio="large", num_epochs=num_epochs, casename=casename)
    evaluate_model(model_case1_large, plot_times=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 
                  mobility_ratio="large", with_observed_data=False, with_diffusion=False, casename=casename)
    models["1"]["large"] = model_case1_large
    models["1"]["large_loss"] = loss_case1_large

    exit()
    
    # Case 2: With observed data, non-trainable parameters, no diffusion
    print("Case 2: With observed data, non-trainable parameters, no diffusion")
    
    model_case2_small = PINN(trainable_params=False).to(DEVICE)
    casename = f"with_observed_data__no_diffusion__no_trainable__small_mobility__{num_epochs}"
    loss_case2_small = train_model(model_case2_small, with_observed_data=True, 
                           with_diffusion=False, mobility_ratio="small", num_epochs=num_epochs, casename=casename)
    evaluate_model(model_case2_small, plot_times=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 
                  mobility_ratio="small", with_observed_data=True, with_diffusion=False, casename=casename)
    models["2"]["small"] = model_case2_small
    models["2"]["small_loss"] = loss_case2_small
    
    model_case2_large = PINN(trainable_params=False).to(DEVICE)
    casename = f"with_observed_data__no_diffusion__no_trainable__large_mobility__{num_epochs}"
    loss_case2_large = train_model(model_case2_large, with_observed_data=True, 
                           with_diffusion=False, mobility_ratio="large", num_epochs=num_epochs, casename=casename)
    evaluate_model(model_case2_large, plot_times=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 
                  mobility_ratio="large", with_observed_data=True, with_diffusion=False, casename=casename)
    models["2"]["large"] = model_case2_large
    models["2"]["large_loss"] = loss_case2_large
    
    # Case 3: With observed data, trainable parameters, no diffusion
    print("Case 3: With observed data, trainable parameters, no diffusion")
    
    model_case3_small = PINN(trainable_params=True).to(DEVICE)
    casename = f"with_observed_data__no_diffusion__with_trainable__small_mobility__{num_epochs}"
    loss_case3_small = train_model(model_case3_small, with_observed_data=True, 
                           with_diffusion=False, mobility_ratio="small", num_epochs=num_epochs, casename=casename)
    evaluate_model(model_case3_small, plot_times=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 
                  mobility_ratio="small", with_observed_data=True, with_diffusion=False, casename=casename)
    models["3"]["small"] = model_case3_small
    models["3"]["small_loss"] = loss_case3_small
    
    model_case3_large = PINN(trainable_params=True).to(DEVICE)
    casename = f"with_observed_data__no_diffusion__with_trainable__large_mobility__{num_epochs}"
    loss_case3_large = train_model(model_case3_large, with_observed_data=True, 
                           with_diffusion=False, mobility_ratio="large", num_epochs=num_epochs, casename=casename)
    evaluate_model(model_case3_large, plot_times=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 
                  mobility_ratio="large", with_observed_data=True, with_diffusion=False, casename=casename)
    models["3"]["large"] = model_case3_large
    models["3"]["large_loss"] = loss_case3_large
    
    # Case 4: With observed data, trainable parameters, with diffusion
    print("Case 4: With observed data, trainable parameters, with diffusion")
    
    model_case4_small = PINN(trainable_params=True).to(DEVICE)
    casename = f"with_observed_data__with_diffusion__with_trainable__small_mobility__{num_epochs}"
    loss_case4_small = train_model(model_case4_small, with_observed_data=True, 
                           with_diffusion=True, mobility_ratio="small", num_epochs=num_epochs, casename=casename)
    evaluate_model(model_case4_small, plot_times=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 
                  mobility_ratio="small", with_observed_data=True, with_diffusion=True, casename=casename)
    models["4"]["small"] = model_case4_small
    models["4"]["small_loss"] = loss_case4_small
    
    model_case4_large = PINN(trainable_params=True).to(DEVICE)
    casename = f"with_observed_data__with_diffusion__with_trainable__large_mobility__{num_epochs}"
    loss_case4_large = train_model(model_case4_large, with_observed_data=True, 
                           with_diffusion=True, mobility_ratio="large", num_epochs=num_epochs, casename=casename)
    evaluate_model(model_case4_large, plot_times=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 
                  mobility_ratio="large", with_observed_data=True, with_diffusion=True, casename=casename)
    models["4"]["large"] = model_case4_large
    models["4"]["large_loss"] = loss_case4_large
    
    # Print trained parameters for each case
    print("\nTrained Parameters Summary:")
    for case in ["3", "4"]:  # Only cases with trainable parameters
        for mr in ["small", "large"]:
            model = models[case][mr]
            print(f"\nCase {case} ({mr} mobility ratio):")
            print(f"k0rg: {model.params.k0rg.item():.4f}")
            print(f"k0rw: {model.params.k0rw.item():.4f}")
            print(f"Swr: {model.params.Swr.item():.4f}")
            print(f"Sgr: {model.params.Sgr.item():.4f}")

if __name__ == "__main__":
    main()