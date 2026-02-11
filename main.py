import numpy as np
from solver.explicit_solver import euler_explicit, RK4
from solver.explicit_stepcontrol_solver import stepcontrol_mid_point_rule
from system_odes.pendulum_ode import damped_pendulum_ode
from visualization.pendulum.visualize_pendulum import VisualizePendulum
from visualization.pendulum_stepcontrol import visualize_pendulum_stepcontrol

#################################################################################################
### Time, stepwidth and initial conditions
t_end = 5
h = 0.1
z0 = [np.deg2rad(75),0]


### Solve the ODEs


### Visualize the results
results = {
    # 'Euler_explicit': (t_Ee, u_Ee)
}
### Pendulum Animation
show_reference = True
viz_pendel = VisualizePendulum(results, show_reference)
viz_pendel.animate()

### Overview plot on errors
visualize_pendulum_stepcontrol(t_sc_mpr, u_sc_mpr, h_sc_mpr, error_sc_mpr )
