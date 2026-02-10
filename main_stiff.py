import numpy as np

from system_odes.dp_ec_battery_model import dp_ec_battery
from solver.explicit_solver import euler_explicit, mid_point_rule, RK4
from solver.explicit_stepcontrol_solver import stepcontrol_mid_point_rule
from visualization.dp_ec_battery import visualize_dp_ec_battery

from scipy.integrate import solve_ivp

# Parameters and time
t_end = 90
h = 0.05
z0 = [0.8,0,0]


t_values, u_values, h_container, error_container = stepcontrol_mid_point_rule(dp_ec_battery,[0,t_end],z0, h)
t_RK4, u_RK4 =euler_explicit(dp_ec_battery,[0,t_end],z0, h)

sol_ode_BDF = solve_ivp(dp_ec_battery,[0,t_end],z0,method='BDF',rtol=1e-6,atol=1e-8)

results = {
    # 'stepcontrol_mid_point_rule': (t_values, u_values),
            'RK4': (t_RK4, u_RK4),
           # 'BDF': (sol_ode_BDF.t, sol_ode_BDF.y)
           }


visualize_dp_ec_battery(results)









