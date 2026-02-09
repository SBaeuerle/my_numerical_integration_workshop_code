import numpy as np
from visualization.pendulum.visualize_pendulum import VisualizePendulum
from solver.explicit_solver import euler_explicit
from system_odes.pendulum_ode import damped_pendulum_ode

# Time, step-width and initial conditions
t_end = 5
h = 0.01
z0 = [np.deg2rad(75),0]

t_values, u_values = euler_explicit(damped_pendulum_ode,[0,t_end],z0, h)
t_values_2, u_values_2 = euler_explicit(damped_pendulum_ode,[0,t_end],z0, 0.001)

show_reference = True
results = {
    'Euler_explicit': (t_values, u_values),
    'Euler_explicit_coarse': (t_values_2, u_values_2),
}

viz_pendel = VisualizePendulum(results, show_reference)
viz_pendel.animate()