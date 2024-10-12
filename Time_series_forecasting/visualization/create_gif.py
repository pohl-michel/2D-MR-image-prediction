import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for rendering on headless systems
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import scipy.io

FPS = 10

# Generate sample data for the original signal and the predicted signal
def generate_signals():
    t = np.linspace(0, 20, 100)  # Time points
    original_signal = np.sin(t)  # Original sine wave signal
    predicted_signal = np.sin(t - 1)  # Predicted signal shifted by 1 unit ahead
    return t, original_signal, predicted_signal

# Compute the dynamic y-axis limits
def get_dynamic_ylim(original_signal, predicted_signal):
    signal_min = min(np.min(original_signal), np.min(predicted_signal))
    signal_max = max(np.max(original_signal), np.max(predicted_signal))
    
    # Add a margin (10% of the signal range)
    margin = 0.1 * (signal_max - signal_min)
    
    # Return the limits with the margin applied
    return signal_min - margin, signal_max + margin

# Create the animation function
def animate_signals(original_signal, predicted_signal, t, horizon, nb_units):

    # Compute the dynamic y-axis limits
    y_min, y_max = get_dynamic_ylim(original_signal, predicted_signal)

    # Create the figure and axis for the animation
    fig, ax = plt.subplots()

    # Set x-axis and y-axis labels
    ax.set_xlabel("Time step index")
    ax.set_ylabel("Marker z coordinate (mm)")

    ax.set_xlim(0, 5)  # Initially set the x-axis limit (this will move)
    ax.set_ylim(y_min, y_max)  # Set the y-axis limit based on the signals

    # Plot empty lines for the original and predicted signals
    original_line, = ax.plot([], [], lw=2, label="Original Signal", color="black")
    predicted_line, = ax.plot([], [], lw=2, label="Predicted Signal", color="red")

    # Add a legend
    ax.legend()

    # Initialize the lines (empty)
    def init():
        original_line.set_data([], [])
        predicted_line.set_data([], [])
        return original_line, predicted_line

    # Update function for each frame of the animation
    def update(frame):
        current_time = t[frame]
        ax.set_xlim(current_time - nb_units, current_time)  # Set x-axis window size based on nb_units

        # Update the original signal to stop at (t - horizon)
        original_line.set_data(t[:frame - int(horizon * (len(t) / (t[-1] - t[0])))], original_signal[:frame - int(horizon * (len(t) / (t[-1] - t[0])))])

        # Update the predicted signal to start from current time
        predicted_line.set_data(t[:frame], predicted_signal[:frame])

        return original_line, predicted_line

    # Create the animation
    anim = FuncAnimation(
        fig, update, frames=len(t), init_func=init, blit=True, interval=20
    )

    # Save the animation as a GIF
    anim.save("signals_animation_with_horizon.gif", writer=PillowWriter(fps=FPS))

# Main code
if __name__ == "__main__":
    
    # Generate signals
    # t, original_signal, predicted_signal = generate_signals()

    input_sq_name = "Ext markers seq 2"
    input_sq_dir = "Time_series_forecasting/a. Input time series sequences"
    sq_filename = '%s/%s/data.mat' % (input_sq_dir, input_sq_name)

    dim_idx = 6 # z coordinate of marker 1 (according to resample_time_series_data.py)

    # Loading the original data
    time_data_mat = scipy.io.loadmat(sq_filename)
    org_time_data = time_data_mat['org_data']

    Tmax = org_time_data.shape[1]

    # Initializing the interpolated data array
    data_dim = org_time_data.shape[0]

    t = np.array(range(0, Tmax))  # Time points
    original_signal = org_time_data[dim_idx, :]
    predicted_signal = np.array([0] + [original_signal[tau - 1] for tau in range(1, Tmax)]) # mock prdiction lagging 1 time step behind

    # Set the prediction horizon (e.g., h = 1)
    horizon = 1

    # Set the number of x-axis units to show in the window (e.g., 7 units)
    nb_units = 7

    # Call the animation function with the specified horizon
    animate_signals(original_signal, predicted_signal, t, horizon, nb_units)

    print("GIF animation saved as 'signals_animation_with_horizon.gif'")