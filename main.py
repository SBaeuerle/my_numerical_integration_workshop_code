import numpy as np
from visualization.pendulum.visualize_pendulum import VisualizePendulum
from solver.explicit_solver import euler_explicit
from system_odes.pendulum_ode import damped_pendulum_ode

# Time, step-width and initial conditions
t_end = 5
h = 0.01
z0 = [np.deg2rad(75),0]

t_values, u_values = euler_explicit(damped_pendulum_ode,[0,t_end],z0, h)



show_reference = True
results = {
    'Euler_explicit': (t_values, u_values),
}

viz_pendel = VisualizePendulum(results, show_reference)
viz_pendel.animate()