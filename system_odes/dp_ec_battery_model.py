import numpy as np

def get_battery_parameters():
    """Returns a dictionary of DP-Model parameters."""
    return {
        'R0': 0.005,     # Ohmic resistance [Ohm]
        'R1': 0.010,     # Fast polarization resistance [Ohm]
        'C1': 1.0,     # Fast capacitance [F] (tau1 = 0.01s)
        'R2': 0.050,     # Slow polarization resistance [Ohm]
        'C2': 2000.0,    # Slow capacitance [F] (tau2 = 100s)
        'Qn': 3600*10.0,  # Nominal capacity [As] (10Ah)
        'U_min': 3.0,    # Minimum voltage of the battery [V]
        'U_max': 4.2     # Maximum voltage of the battery [V]
    }


def current_profile(t):
    """
    Returns current in Amperes.
    Positive = Discharge, Negative = Charge.
    """
    # Let's do a 20A pulse between 10s and 30s
    if 10.0 <= t <= 30.0:
        return 20.0
    # And a small charging pulse (regen) between 60s and 70s
    elif 60.0 <= t <= 70.0:
        return -10.0
    else:
        return 0.0

def dp_ec_battery(t, z):
    soc, u1, u2 = z

    params = get_battery_parameters()
    R0 = params['R0']
    R1 = params['R1']
    R2 = params['R2']
    C1 = params['C1']
    C2 = params['C2']
    Qn = params['Qn']

    # Get current at time t (e.g., a pulse or constant discharge)
    i = current_profile(t)

    # dSoC/dt (Current in Amperes, Qn in Ampere-seconds)
    dsoc = -i / Qn

    # dU1/dt (Fast polarization)
    du1 = -u1 / (R1 * C1) + i / C1

    # dU2/dt (Slow polarization)
    du2 = -u2 / (R2 * C2) + i / C2

    return np.array([dsoc, du1, du2])