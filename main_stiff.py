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


### Visualize the results
results = {
    # 'Euler explicit': (t_Ee, u_Ee)
           }

visualize_dp_ec_battery(results)