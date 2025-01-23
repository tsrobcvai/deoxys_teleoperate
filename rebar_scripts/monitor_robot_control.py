import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import time

# Example data class to hold robot state, action, and time.
# class RobotData:
#     def __init__(self):
#         self.time = []
#         self.state = [[] for _ in range(7)]  # 7-value array for robot state
#         self.action = [[] for _ in range(7)]  # 7-value array for action
#
#     def update(self, new_time, new_state, new_action):
#         self.time.append(new_time)
#         for i in range(7):
#             self.state[i].append(new_state[i])
#             self.action[i].append(new_action[i])
#
# # Simulate getting new robot state and action data (replace with real data updates)
# def simulate_robot_data(robot_data, current_time):
#     new_state = [random.uniform(0, 100) for _ in range(7)]  # Random 7-value robot state
#     new_action = [random.uniform(-10, 10) for _ in range(7)]  # Random 7-value action
#     robot_data.update(current_time, new_state, new_action)


# Function to animate the plot with the robot data
def animate(i, times, lines_state, lines_action, state_points, action_points, axes):
    if len(times) == 0:
        return  # Wait until the robot process fills in the data
    # assert len(times) == len(state_points[0]) == len(action_points[0])
    # To solve the dismatch

    # times_copy = times.deepcopy()

    # Update each subplot for robot state and action
    for j in range(6):
        # print(f'interations: {j}')
        # print(f'len(times): {len(times)}')
        # print(f'len(state_points[j]): {len(state_points[j])}')
        # print(f'len(action_points[j]): {len(action_points[j])}')
        # print(f'clip_len: {min_len}')
        min_len = min(len(times), len(state_points[j]), len(action_points[j]))
        # lines_state[j].set_data(times, state_points[j])
        # lines_action[j].set_data(times, action_points[j])
        lines_state[j].set_data(times[:min_len], state_points[j][:min_len])
        lines_action[j].set_data(times[:min_len], action_points[j][:min_len])

    for ax in axes:
        ax.relim()  # Recalculate limits
        ax.autoscale_view()  # Update view to include new data

    return lines_state + lines_action

def animate_7(i, times, lines_state, lines_action, state_points, action_points, axes):
    if len(times) == 0:
        return  # Wait until the robot process fills in the data
    # assert len(times) == len(state_points[0]) == len(action_points[0])
    # To solve the dismatch

    # times_copy = times.deepcopy()

    # Update each subplot for robot state and action
    for j in range(7):
        # print(f'interations: {j}')
        # print(f'len(times): {len(times)}')
        # print(f'len(state_points[j]): {len(state_points[j])}')
        # print(f'len(action_points[j]): {len(action_points[j])}')
        # print(f'clip_len: {min_len}')
        min_len = min(len(times), len(state_points[j]), len(action_points[j]))
        # lines_state[j].set_data(times, state_points[j])
        # lines_action[j].set_data(times, action_points[j])
        lines_state[j].set_data(times[:min_len], state_points[j][:min_len])
        lines_action[j].set_data(times[:min_len], action_points[j][:min_len])

    for ax in axes:
        ax.relim()  # Recalculate limits
        ax.autoscale_view()  # Update view to include new data

    return lines_state + lines_action

# Function to save the figure when 'q' is pressed
def save_figure(event, fig, save_path):
    if event.char == 'q':  # If 'q' is pressed
        fig.savefig(save_path)
        print(f"Figure saved to {save_path}")

# Create a function to monitor robot state and action
def monitor(times, state_points, action_points, save_path):
    # Create the main Tkinter window
    root = tk.Tk()
    root.title("Robot State and Action Monitor")

    # Create a Matplotlib figure with 7 subplots for the 7 robot state/action values
    fig, axes = plt.subplots(6, 1, figsize=(8, 10), sharex=True)
    fig.tight_layout(pad=3.0)

    lines_state = []
    lines_action = []
    for i in range(6):
        line_state, = axes[i].plot([], [], 'r-', label=f'State {i + 1}')
        line_action, = axes[i].plot([], [], 'b--', label=f'Action {i + 1}')
        axes[i].set_title(f'Robot State and Action {i + 1}')
        axes[i].set_ylabel('Value')
        axes[i].legend(loc="upper left")
        lines_state.append(line_state)
        lines_action.append(line_action)

    axes[-1].set_xlabel('Time')

    # Link Matplotlib figure to Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Use Matplotlib's animation module to update the plot in real-time
    ani = animation.FuncAnimation(fig, animate,
                                  fargs=(times, lines_state, lines_action,
                                         state_points, action_points, axes), interval=50)

    # Bind the 'q' key to the save function
    root.bind("<Key>", lambda event: save_figure(event, fig, save_path))

    # Start the Tkinter main loop
    root.mainloop()

def torque_monitor(times, tau_meansured_points, tau_active_points, save_path):
    # Create the main Tkinter window
    root = tk.Tk()
    root.title("Tau_meansured and Tau_actived Monitor")

    # Create a Matplotlib figure with 7 subplots for the 7 robot state/action values
    fig, axes = plt.subplots(7, 1, figsize=(8, 10), sharex=True)
    fig.tight_layout(pad=3.0)

    lines_state = []
    lines_action = []
    for i in range(7):
        line_state, = axes[i].plot([], [], 'r-', label=f'Tau_meansured {i + 1}')
        line_action, = axes[i].plot([], [], 'b--', label=f'Tau_actived {i + 1}')
        axes[i].set_title(f'Joint {i + 1}')
        axes[i].set_ylabel('Value')

        # Add horizontal dashed lines at specific y-values
        # for y in [-20, -15, -10, -5, 0, 5, 10, 15, 20]:
        #     axes[i].axhline(y=y, color='gray', linestyle='--', linewidth=0.5)

        axes[i].legend(loc="upper left")
        lines_state.append(line_state)
        lines_action.append(line_action)

    axes[-1].set_xlabel('Time')

    # Link Matplotlib figure to Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Use Matplotlib's animation module to update the plot in real-time
    ani = animation.FuncAnimation(fig, animate_7,
                                  fargs=(times, lines_state, lines_action,
                                         tau_meansured_points, tau_active_points, axes), interval=50)

    # Bind the 'q' key to the save function
    root.bind("<Key>", lambda event: save_figure(event, fig, save_path))

    # Start the Tkinter main loop
    root.mainloop()

if __name__ == "__main__":
    # Define the save path for the figure
    save_path = "robot_state_action_figure.png"
    monitor(save_path)