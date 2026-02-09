import numpy as np

from system_odes.dp_ec_battery_model import dp_ec_battery
from solver.explicit_solver import euler_explicit, mid_point_rule, RK4
from solver.explicit_stepcontrol_solver import stepcontrol_mid_point_rule
from visualization.dp_ec_battery import visualize_dp_ec_battery

# Parameters and time
t_end = 90
h = 0.1
z0 = [0.8,0,0]


t_values, u_values, h_container, error_container = stepcontrol_mid_point_rule(dp_ec_battery,[0,t_end],z0, h)


visualize_dp_ec_battery(t_values,u_values)









