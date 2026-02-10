import numpy as np
import matplotlib.pyplot as plt
from system_odes.dp_ec_battery_model import get_battery_parameters, current_profile


def get_plotting_data(t_arr, z_arr):
    """Computes U_term, U_oc, and current array for plotting."""
    params = get_battery_parameters()
    R0 = params['R0']
    U_min, U_max = params['U_min'], params['U_max']

    # Handle shape: we need (N, 3) for the unpacking below
    if z_arr.shape[0] == 3 and z_arr.shape[1] != 3:
        z_arr = z_arr.T

    soc = z_arr[:, 0]
    u1 = z_arr[:, 1]
    u2 = z_arr[:, 2]

    i_arr = np.array([current_profile(t) for t in t_arr])
    u_oc = U_min + (U_max - U_min) * soc
    u_term = u_oc - u1 - u2 - i_arr * R0

    return i_arr, u_term, u1, u2, soc


def visualize_dp_ec_battery(results_dict):
    """
    results_dict: { "Solver Name": (t_array, z_array), ... }
    """
    # 4 Rows: Effort, Input, State, Output
    fig, (ax_curr, ax_soc, ax_term,  ax_volt_1, ax_volt_2) = plt.subplots(5, 1, figsize=(11, 14), sharex=True,
                                                             gridspec_kw={'height_ratios': [0.8, 0.8, 1, 1, 1]})

    # Dynamic Title with Step Counts
    counts = [f"{len(t)} ({name})" for name, (t, z) in results_dict.items()]
    fig.suptitle('Battery Model Comparison: ' + ' vs '.join(counts),
                 fontsize=14, fontweight='bold', y=0.98)

    for name, (t, z) in results_dict.items():
        # Calculate plotting vectors
        i_arr, u_term, u1, u2, soc = get_plotting_data(t, z)

        # Plot 1: Cumulative Steps (Workload)
        ax_curr.plot(t, i_arr, label=name)

        # Plot 2: SoC
        ax_soc.plot(t, soc * 100, label=f"SoC ({name})")

        # Plot 3: U terminal
        ax_term.plot(t, u_term, label=f"U_terminal ({name})")

        # Plot 4: Voltage 1
        ax_volt_1.plot(t, u1, label=f"U_1 (fast - {name})")

        # Plot 5: Voltage 2
        ax_volt_2.plot(t, u2, label=f"U_2 (slow - {name})")


    # Row 1 Styling: Current Profile (Common to all)
    t_first, _ = next(iter(results_dict.values()))
    i_vals = [current_profile(ti) for ti in t_first]
    ax_curr.plot(t_first, i_vals, 'k-', linewidth=1.5)
    ax_curr.set_ylabel('Current / A')
    ax_curr.set_title('Load Profile')
    ax_curr.grid(True, alpha=0.3)

    # Row 2 Styling: SoC
    ax_soc.set_ylabel('SoC / %')
    ax_soc.set_title('State of Charge')
    ax_soc.grid(True, alpha=0.3)
    ax_soc.legend()

    # Row 3 Styling: Terminal Voltage
    ax_term.set_ylabel('Voltage / V')
    ax_term.set_title('Voltage terminal')
    ax_term.set_xlabel('Time / s')
    ax_term.grid(True, alpha=0.3)
    ax_term.legend()

    # Row 4 Styling: Voltage 1 (fast)
    ax_volt_1.set_ylabel('Voltage / V')
    ax_volt_1.set_title('Voltage 1 Response')
    ax_volt_1.set_xlabel('Time / s')
    ax_volt_1.grid(True, alpha=0.3)
    ax_volt_1.legend()

    # Row 5 Styling: Voltage 1 (slow)
    ax_volt_2.set_ylabel('Voltage / V')
    ax_volt_2.set_title('Voltage 2 Response')
    ax_volt_2.set_xlabel('Time / s')
    ax_volt_2.grid(True, alpha=0.3)
    ax_volt_2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()