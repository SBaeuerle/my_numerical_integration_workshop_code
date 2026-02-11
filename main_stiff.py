import numpy as np
from system_odes.dp_ec_battery_model import dp_ec_battery
from solver.explicit_solver import euler_explicit, RK4
from solver.explicit_stepcontrol_solver import stepcontrol_mid_point_rule
from visualization.dp_ec_battery import visualize_dp_ec_battery
from scipy.integrate import solve_ivp

### Time, stepwidth and initial conditions
t_end = 100
h = 0.05
z0 = np.array([0.8,0,0])

### Solve the ODEs for the Equivalent Circuit - Dual Polarisation Battery model
t_Ee, u_Ee = euler_explicit(dp_ec_battery,[0,t_end],z0, h)
t_RK4, u_RK4 = RK4(dp_ec_battery,[0,t_end],z0, h)
t_sc_mpr, u_sc_mpr, h_sc_mpr, error_sc_mpr = stepcontrol_mid_point_rule(dp_ec_battery,[0,t_end],z0)
sol_ode = solve_ivp(dp_ec_battery,[0,t_end],z0,method='BDF',rtol=1e-6,atol=1e-8)

### Visualize the results
results = {
    'Euler explicit': (t_Ee, u_Ee),
    'RK4': (t_RK4, u_RK4),
    'stepcontrol_mid_point_rule': (t_sc_mpr, u_sc_mpr),
    'BDF': (sol_ode.t, sol_ode.y)
           }

visualize_dp_ec_battery(results)