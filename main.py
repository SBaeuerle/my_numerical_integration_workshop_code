import numpy as np
from visualization.pendulum.visualize_pendulum import VisualizePendulum

from solver.explicit_solver import euler_explicit, mid_point_rule, RK4
from solver.explicit_stepcontrol_solver import stepcontrol_mid_point_rule
from system_odes.pendulum_ode import damped_pendulum_ode

from visualization.pendulum.visualize_pendulum import VisualizePendulum
from visualization.stepcontrol import visualize_stepcontrol

# Time, step-width and initial conditions
t_end = 5
h = 0.5
z0 = [np.deg2rad(75),0]

t_values, u_values = euler_explicit(damped_pendulum_ode,[0,t_end],z0, h)
t_values_2, u_values_2 = mid_point_rule(damped_pendulum_ode,[0,t_end],z0, h)
t_values_3, u_values_3 = RK4(damped_pendulum_ode,[0,t_end],z0, h )

t_values_4, u_values_4, h_values_4, relative_error_values_4 = stepcontrol_mid_point_rule(damped_pendulum_ode,[0,t_end],z0, h)

show_reference = True
results = {
    'Euler_explicit': (t_values, u_values),
    'Mid_point_rule': (t_values_2, u_values_2),
    'RK4': (t_values_3, u_values_3),
    'stepcontrol_mid_point_rule': (t_values_4, u_values_4),
}

viz_pendel = VisualizePendulum(results, show_reference)
viz_pendel.animate()

visualize_stepcontrol(t_values_4, u_values_4, h_values_4, relative_error_values_4)

