# PINNs vs WENO: Comparative Study

This repository presents a comprehensive comparison between Physics-Informed Neural Networks (PINNs) and Weighted Essentially Non-Oscillatory (WENO) schemes for solving benchmark problems in computational fluid dynamics (CFD), such as the 2D lid-driven cavity (Navier-Stokes equations) and the Buckley-Leverett equation.

- **PINN:** A deep learning approach that incorporates physical laws into the loss function to learn solutions to partial differential equations.
- **WENO:** A high-order finite difference method widely used for solving hyperbolic PDEs with sharp gradients or discontinuities.

The scripts in this repository train PINN models and solve the same problems using WENO schemes, then quantitatively and visually compare their performance in terms of error metrics and computational residuals.

## Highlights

- Automated training and evaluation for both PINN and WENO on standard benchmarks.
- Detailed error and residual analysis with visualizations and summary files.
- Performance metrics and boundary condition error comparisons.

---

🏆 **This project won the Best Thesis Award at the Mathematics Department, IIT Roorkee.**

---

## Requirements

- Python 3.x
- NumPy, Matplotlib, and other common scientific computing packages

## Usage

Run the main comparison scripts:
```bash
python combine_ns.py    # Navier-Stokes (Lid-driven cavity)
python combine_bl.py    # Buckley-Leverett equation
```

Results (figures and metrics) will be saved in the respective results folders.

---

## License

This project is intended for academic use. Please cite appropriately if used in research or publications.
