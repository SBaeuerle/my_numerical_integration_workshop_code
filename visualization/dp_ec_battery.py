import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from system_odes.dp_ec_battery_model import get_battery_parameters, current_profile, dp_ec_battery

# --- Calculate Terminal Voltage ---
def get_plotting_data(t_arr, z_arr):
    """Computes U_term, U_oc, and current array for plotting."""
    params = get_battery_parameters()

    R0 = params['R0']
    U_min, U_max = params['U_min'], params['U_max']

    soc = z_arr[:, 0]
    u1 = z_arr[:, 1]
    u2 = z_arr[:, 2]

    # Vectorized current for the time array
    i_arr = np.array([current_profile(t) for t in t_arr])

    # Linear OCV model
    u_oc = U_min + (U_max - U_min) * soc
    u_term = u_oc - u1 - u2 - i_arr * R0

    return i_arr, u_term, u1, u2, soc

def simulate_dp_ec_battery(t_container: np.ndarray, z_container: np.ndarray):

    z0 = z_container[1,:]
    t_span = [min(t_container), max(t_container)]

    # A. Reference Solver (Radau is great for stiff problems)
    sol_ref = solve_ivp(
        lambda t, z: dp_ec_battery(t, z),
        t_span, z0, method='Radau', rtol=1e-6, atol=1e-8
    )

    return sol_ref.t, sol_ref.y.T


def visualize_dp_ec_battery(t_user: np.ndarray, z_user: np.ndarray):

    # Get a reference solution
    t_ref, z_ref = simulate_dp_ec_battery(t_user, z_user)

    # Get Plotting data
    i_ref, u_term_ref, u1_ref, u2_ref, soc_ref = get_plotting_data(t_ref, z_ref)
    i_user, u_term_user, u1_user, u2_user, soc_user = get_plotting_data(t_user, z_user)

    total_steps_my = len(t_user)
    total_steps_ref = len(t_ref)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    fig.suptitle(f'Battery Simulation: {total_steps_my} Steps (Our Solver) vs. {total_steps_ref} Steps (Reference Solver)',
                 fontsize=12, fontweight='bold', y=0.98)

    # Row 1: Current Profile
    ax1.plot(t_ref, i_ref, 'k-', linewidth=2, label='Current I(t)')
    ax1.set_ylabel('Current / A')
    # ax1.set_title('Load Profile')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # Row 2: State of Charge (SoC)
    # Note: SoC changes are small, so we might zoom or use a fine scale
    ax2.plot(t_ref, soc_ref * 100, 'k-', alpha=0.5, label='Reference (SoC)')
    ax2.plot(t_user, soc_user * 100, 'r--', label='Our Solver')
    ax2.set_ylabel('SoC / %')
    ax2.set_title('State of Charge')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower left')

    # Row 3: Voltages (U1, U2, Uterm)
    # Plot Terminal Voltage on Left Axis
    l1, = ax3.plot(t_ref, u_term_ref, 'b--', label='U_term (Ref)')
    l2, = ax3.plot(t_user, u_term_user, 'b-', label='U_term')
    ax3.set_ylabel('Terminal Voltage / V', color='blue')
    ax3.tick_params(axis='y', labelcolor='blue')

    # Plot Internal Voltages (U1, U2) on Right Axis
    ax3_twin = ax3.twinx()
    l3, = ax3_twin.plot(t_ref, u1_ref, 'g--', label='U1 Ref (Fast)')
    l4, = ax3_twin.plot(t_ref, u2_ref, 'm--', label='U2 Ref (Slow)')
    l5, = ax3_twin.plot(t_user, u1_user, 'g-', label='U1 (Fast)')
    l6, = ax3_twin.plot(t_user, u2_user, 'm-', label='U2 (Slow)')
    ax3_twin.set_ylabel('Polarization V / V', color='green')
    ax3_twin.tick_params(axis='y', labelcolor='green')
    ax3_twin.set_ylim(-0.1, 0.5)  # Scale to see the small voltages

    # Combined Legend
    lines = [l1, l2, l3, l4, l5, l6]
    ax3.legend(lines, [l.get_label() for l in lines], loc='center right')
    ax3.set_xlabel('Time / s')
    ax3.set_title('Voltage Response (Terminal & Internal)')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
