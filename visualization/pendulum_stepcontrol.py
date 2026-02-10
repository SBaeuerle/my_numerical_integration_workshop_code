import matplotlib.pyplot as plt
import numpy as np
from visualization.helper import smart_plot

def visualize_pendulum_stepcontrol(t_vals, u_vals, h_vals, err_vals):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [1, 1.2]})

    # --- Top Plot: Physics (Angle) ---
    ax1.plot(t_vals, np.rad2deg(u_vals[:, 0]), color='black', linewidth=1.5, label='Angle')
    ax1.set_ylabel('Angle / deg')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # --- Bottom Plot: Step Width (h) ---
    color_h = 'tab:red'
    ax2.step(t_vals, h_vals, where='post', color=color_h, linewidth=2, label='Step Width (h)')
    ax2.set_xlabel('Time / s')
    ax2.set_ylabel('Step Width / s', color=color_h)
    ax2.tick_params(axis='y', labelcolor=color_h)
    ax2.set_yscale('log')  # Log scale is often better for h
    ax2.grid(True, alpha=0.3)

    # --- Twin Axis: Normalized Error Score ---
    ax3 = ax2.twinx()
    color_err = 'tab:green'
    ax3.plot(t_vals, err_vals, color=color_err, alpha=0.5, label='Norm. Error', marker='*')
    ax3.set_ylabel('Error Score (Normalized)', color=color_err)
    ax3.tick_params(axis='y', labelcolor=color_err)

    # Highlight the "Working Zone" [0.5, 1.0]
    ax3.axhline(y=1.0, color='green', linestyle='--', alpha=0.8, label='Max Tolerance')
    ax3.axhline(y=0.5, color='green', linestyle=':', alpha=0.8, label='Min Threshold')
    ax3.fill_between(t_vals, 0.5, 1.0, color='green', alpha=0.1, label='Target Zone')

    # Limit the error axis so it doesn't dwarf the h-axis
    ax3.set_ylim(0, 1.5)

    fig.suptitle('Adaptive Step-Control: Strategy Visualization', fontsize=14)
    plt.tight_layout()

    smart_plot(fig, 'pendulum_stepcontrol.png')