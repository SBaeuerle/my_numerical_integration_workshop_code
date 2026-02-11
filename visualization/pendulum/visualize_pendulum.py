from typing import Optional, Tuple, List, Dict
import os
import subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

# Keep your imports
from visualization.pendulum.pendulum_data import PendulumData
from visualization.pendulum.pendulum_plot_utils import PendulumPlotInitializer


class VisualizePendulum():
    def __init__(self, simulation_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
                 reference: bool = False, ref_step_width: float = 0.01) -> None:

        self.pendulum_data_runs = {}
        self.plot_initializer = PendulumPlotInitializer()
        self.ref_step_width = ref_step_width

        # 1. Load Data
        run_names = list(simulation_results.keys())
        first_run_name = run_names[0] if run_names else None

        for name, (t_vals, u_vals) in simulation_results.items():
            # Use the "Gatekeeper" logic we discussed implicitly here or in PendulumData
            current_pendulum_data = PendulumData(
                values_time=t_vals,
                values_state=u_vals,
                reference=(reference if name == first_run_name else False),
            )
            self.pendulum_data_runs[name] = current_pendulum_data

            if current_pendulum_data.reference:
                self.reference_pendulum_data = current_pendulum_data

        # 2. THE FIX: Synchronize all data to a single Master Timeline
        self._synchronize_data_for_animation()

    def _smart_display(self, ani, fig, save_path="pendulum_animation.mp4", port=8000):
        if os.getenv('CODESPACES') == 'true':
            print(f"🌐 Cloud detected: Saving to {save_path}...")
            plt.close(fig)
            ani.save(save_path, writer='ffmpeg', fps=30)

            # Start HTTP server in background
            print(f"🚀 Starting HTTP server on port {port}...")
            subprocess.Popen(
                ["python", "-m", "http.server", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            print(f"▶ Open in browser: http://localhost:{port}/{save_path}")
        else:
            print("💻 Local detected: Opening popup window...")
            plt.show()


    def _synchronize_data_for_animation(self):
        """
        Creates a 'Master Timeline' and resamples all simulation runs to match it
        using Forward Fill (Zero-Order Hold).
        """
        # A: Find the global time boundaries
        all_max_times = [data.values_time[-1] for data in self.pendulum_data_runs.values()]
        max_time = max(all_max_times)

        # B: Create the Master Time array (uniform grid)
        # We use ref_step_width as the 'tick' for the animation clock
        self.master_time = np.arange(0, max_time, self.ref_step_width)

        # C: Resample each run to match master_time exactly
        self.anim_data = {}  # Stores the normalized (N_master,) arrays

        for name, data in self.pendulum_data_runs.items():
            # precise_indices maps the Master Clock to the Simulation's coarser clock.
            # 'side=right' + subtracting 1 gives us the "Previous" neighbor (Forward Fill)
            idx = np.searchsorted(data.values_time, self.master_time, side='right') - 1

            # Safety clamp: ensure we don't go out of bounds (e.g. at t=0)
            idx = np.clip(idx, 0, len(data.values_time) - 1)

            # Create the synchronized arrays
            # This is the "Stepwise" look: we grab the value at 'idx' for every master tick
            synchronized_angle = data.values_angle[idx]

            self.anim_data[name] = {
                'time': self.master_time,  # Technically redundant, but keeps structure clean
                'angle': synchronized_angle
            }

        # Handle Reference Data Synchronization (if it exists)
        if self.reference_pendulum_data and self.reference_pendulum_data.values_angle_ref is not None:
            ref_data = self.reference_pendulum_data
            idx_ref = np.searchsorted(ref_data.values_time_ref, self.master_time, side='right') - 1
            idx_ref = np.clip(idx_ref, 0, len(ref_data.values_time_ref) - 1)

            self.anim_data['REFERENCE'] = {
                'angle': ref_data.values_angle_ref[idx_ref]
            }

    def _init_animation(self) -> Tuple:
        all_artists = []
        # Clear main lines
        for line in self.time_lines:
            line.set_data([], [])
            all_artists.append(line)
        for marker in self.time_markers:
            marker.set_data([], [])
            all_artists.append(marker)
        for line in self.pendulum_lines:
            line.set_data([], [])
            all_artists.append(line)
        for rect in self.pendulum_rects:
            rect.set_transform(self.ax_pend.transData)
            all_artists.append(rect)

        # Clear reference
        if self.reference_pendulum_line:
            self.reference_pendulum_line.set_data([], [])
            all_artists.append(self.reference_pendulum_line)
        if self.reference_pendulum_rect:
            self.reference_pendulum_rect.set_transform(self.ax_pend.transData)
            all_artists.append(self.reference_pendulum_rect)

        self.text_pend.set_text("")
        all_artists.append(self.text_pend)
        return tuple(all_artists)

    def _update_animation(self, frame_idx: int) -> Tuple:
        """
        frame_idx is now a direct index into self.master_time.
        No more time calculations needed here!
        """
        all_artists = []

        # 1. Update Global Time Text
        # frame_idx might exceed array length slightly if FuncAnimation buffers, so we clamp
        safe_idx = min(frame_idx, len(self.master_time) - 1)
        current_time = self.master_time[safe_idx]
        self.text_pend.set_text(f"{current_time:.2f} s")
        all_artists.append(self.text_pend)

        # 2. Update Simulation Runs
        for i, (name, _) in enumerate(self.pendulum_data_runs.items()):
            # Retrieve the PRE-CALCULATED synchronized data
            synced = self.anim_data[name]

            # Get the current angle directly using the frame index
            current_angle = synced['angle'][safe_idx]

            # --- Update Time Plot (History trace) ---
            # We plot the history up to the current frame
            # (Optional: Decimate this if it gets too slow, e.g. synced['time'][:safe_idx:5])
            self.time_lines[i].set_data(synced['time'][:safe_idx], np.rad2deg(synced['angle'][:safe_idx]))
            all_artists.append(self.time_lines[i])

            # Marker at the tip
            self.time_markers[i].set_data([current_time], [np.rad2deg(current_angle)])
            all_artists.append(self.time_markers[i])

            # --- Update Pendulum (Geometry) ---
            x = self.plot_initializer.length_pend * np.sin(current_angle)
            y = -self.plot_initializer.length_pend * np.cos(current_angle)

            # Line
            length_string = (self.plot_initializer.length_pend - self.plot_initializer.length_rect_long / 2)
            x_line = x * length_string / self.plot_initializer.length_pend
            y_line = y * length_string / self.plot_initializer.length_pend
            self.pendulum_lines[i].set_data([0, x_line], [0, y_line])
            all_artists.append(self.pendulum_lines[i])

            # Rectangle (Bob)
            angle_rad = np.arctan2(y, x)
            trans = Affine2D().rotate(angle_rad).translate(x, y) + self.ax_pend.transData
            self.pendulum_rects[i].set_transform(trans)
            all_artists.append(self.pendulum_rects[i])

        # 3. Update Reference (if exists)
        if 'REFERENCE' in self.anim_data and self.reference_pendulum_line:
            ref_angle = self.anim_data['REFERENCE']['angle'][safe_idx]

            x_ref = self.plot_initializer.length_pend * np.sin(ref_angle)
            y_ref = -self.plot_initializer.length_pend * np.cos(ref_angle)

            length_string_ref = (self.plot_initializer.length_pend - self.plot_initializer.length_rect_long / 2)
            x_line_ref = x_ref * length_string_ref / self.plot_initializer.length_pend
            y_line_ref = y_ref * length_string_ref / self.plot_initializer.length_pend

            self.reference_pendulum_line.set_data([0, x_line_ref], [0, y_line_ref])
            all_artists.append(self.reference_pendulum_line)

            angle_ref = np.arctan2(y_ref, x_ref)
            trans_ref = Affine2D().rotate(angle_ref).translate(x_ref, y_ref) + self.ax_pend.transData
            self.reference_pendulum_rect.set_transform(trans_ref)
            all_artists.append(self.reference_pendulum_rect)

        return tuple(all_artists)

    def animate(self) -> FuncAnimation:
        self._create_animation_figure()  # Calls your existing figure setup

        # Frames = Length of the Master Timeline
        total_frames = len(self.master_time)
        interval_ms = self.ref_step_width * 1000  # Real-time speed

        self.ani = FuncAnimation(
            self.fig,
            self._update_animation,
            frames=total_frames,
            init_func=self._init_animation,
            blit=True,
            interval=interval_ms,
            repeat=True
        )
        plt.suptitle("Pendulum Animation (Synchronized)")
        self._smart_display(self.ani, self.fig)
        return self.ani

    # (Keep your existing _create_animation_figure and plot methods as they were)
    # Just ensure _create_animation_figure uses 'self.pendulum_data_runs' as before.
    def _create_animation_figure(self) -> None:
        # Copy-paste your previous _create_animation_figure code here
        # It is compatible because self.pendulum_data_runs is still set in __init__
        self.fig, self.ax_time, self.ax_pend = self.plot_initializer.create_figure_and_axes()

        self.time_lines = []
        self.time_markers = []
        self.pendulum_lines = []
        self.pendulum_rects = []

        min_angle_deg = 0
        max_angle_deg = 0
        min_time = float('inf')
        max_time = float('-inf')

        for i, (name, data) in enumerate(self.pendulum_data_runs.items()):
            color = self.plot_initializer.colors[i % len(self.plot_initializer.colors)]

            if data.reference and data.values_time_ref is not None:
                self.reference_time_line = self.plot_initializer.create_reference_time_line(self.ax_time)
                self.reference_time_line.set_data(data.values_time_ref, np.rad2deg(data.values_angle_ref))

            line, marker = self.plot_initializer.create_time_plot_artists(self.ax_time, name, color)
            self.time_lines.append(line)
            self.time_markers.append(marker)

            min_angle_deg = min(min_angle_deg, np.min(np.rad2deg(data.values_angle)))
            max_angle_deg = max(max_angle_deg, np.max(np.rad2deg(data.values_angle)))
            min_time = min(min_time, np.min(data.values_time))
            max_time = max(max_time, np.max(data.values_time))

        self.plot_initializer.setup_time_axis(self.ax_time, min_time, max_time, min_angle_deg, max_angle_deg)
        self.plot_initializer.setup_pendulum_axis(self.ax_pend)

        for i, (name, data) in enumerate(self.pendulum_data_runs.items()):
            color = self.plot_initializer.colors[i % len(self.plot_initializer.colors)]
            line, rect = self.plot_initializer.create_pendulum_artists(self.ax_pend, color)
            self.pendulum_lines.append(line)
            self.pendulum_rects.append(rect)

        if self.reference_pendulum_data and self.reference_pendulum_data.values_angle_ref is not None:
            self.reference_pendulum_line, self.reference_pendulum_rect = \
                self.plot_initializer.create_reference_pendulum_artists(self.ax_pend)

        self.plot_initializer.create_pivot_point_artist(self.ax_pend)
        self.text_pend = self.plot_initializer.create_time_text_artist(self.ax_pend)    