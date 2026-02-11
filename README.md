# ODE Solver Comparison Project

A comprehensive Python project for comparing different Ordinary Differential Equation (ODE) solvers using two representative test systems: a damped pendulum and a dual-polarization battery model.

## 🚀 Features

### Numerical Solvers
- **Euler Explicit**: Simple first-order explicit method
- **Runge-Kutta 4th Order (RK4)**: Classic fourth-order explicit method
- **Midpoint Rule**: Second-order explicit method
- **Adaptive Step Control**: Midpoint rule with automatic step size adjustment
- **Reference Solutions**: High-precision solutions using SciPy's `solve_ivp` with BDF method

### Test Systems
1. **Damped Pendulum**: Nonlinear oscillator with damping
2. **Battery Model**: Dual-polarization equivalent circuit model for lithium-ion batteries

### Visualization
- **Interactive Pendulum Animation**: Real-time synchronized visualization of multiple solver results
- **Battery Performance Plots**: Comprehensive analysis of voltage, current, and state-of-charge
- **Step Control Analysis**: Visualization of adaptive step size behavior and error estimates
- **Cloud-friendly Output**: Automatic detection of development environment (local vs. cloud)

## 📋 Requirements
numpy
matplotlib
scipy

## 🏃 Quick Start

### Pendulum Simulation
```bash
python main.py 
```
This will:

Solve the damped pendulum ODE using multiple methods
Display an interactive animation comparing all solutions
Show step control analysis plots

### Battery Model Simulation
```bash
python main_stiff.py 
```

This will:

Solve the stiff battery model equations
Compare explicit methods with implicit BDF solver
Display comprehensive battery performance plots

## Project Structure
├── main.py                                    # Pendulum simulation runner
├── main_stiff.py                             # Battery model simulation runner
├── solver/
│   ├── explicit_solver.py                   # Euler, RK4, Midpoint implementations
│   └── explicit_stepcontrol_solver.py       # Adaptive step size control
├── system_odes/
│   ├── pendulum_ode.py                      # Damped pendulum equations
│   └── dp_ec_battery_model.py               # Battery model equations
└── visualization/
    ├── pendulum/
    │   ├── visualize_pendulum.py            # Main pendulum animation class
    │   ├── pendulum_data.py                 # Data management and synchronization
    │   └── pendulum_plot_utils.py           # Plot initialization utilities
    ├── dp_ec_battery.py                     # Battery visualization
    ├── pendulum_stepcontrol.py              # Step control analysis plots
    └── helper.py                            # Environment-aware plotting utilities
