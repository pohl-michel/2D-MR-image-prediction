import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for rendering on headless systems
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import scipy.io

FPS = 3 # acquisition frequency

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
def animate_signals(original_signal, predicted_signal, t, horizon, nb_units, figsize, m, start_time):

    # Compute the dynamic y-axis limits
    y_min, y_max = get_dynamic_ylim(original_signal, predicted_signal)

    # Create the figure and axis for the animation
    fig, ax = plt.subplots(figsize=figsize)  # Set width to 6 and height to fig_height)

    # Set x-axis and y-axis labels
    ax.set_xlabel("Time step index")
    ax.set_ylabel("Marker z coordinate (mm)")

    ax.set_xlim(0, 5)  # Initially set the x-axis limit (this will move)
    ax.set_ylim(y_min, y_max)  # Set the y-axis limit based on the signals

    # Plot empty lines for the original and predicted signals
    original_line, = ax.plot([], [], lw=2, label="Original Signal", color="black") # lw: line width
    predicted_line, = ax.plot([], [], lw=1, label="Predicted Signal", color="red")

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

        ax.set_xlim(current_time - nb_units, current_time)

        # Update the original signal to stop at (t - horizon)
        original_line.set_data(t[:frame - horizon], original_signal[:frame - horizon])

        # Update the predicted signal to start after the warm-up period (only display after time step m)
        if frame > m:
            predicted_line.set_data(t[m:frame], predicted_signal[m:frame])
        else:
            predicted_line.set_data([], [])  # No data before m

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

    input_sq_name = "Ext markers seq 1  3.33 Hz" # seq 3 has few time points so good for experimenting...
    pred_sq_filename = "pred_result_variables Ext markers seq 1  3.33 Hz tmax_pred=740 DNI k=12 q=180 eta=0.01 sg=0.02 grd_tshld=100 h=7 nrlzed data.mat"

    input_sq_dir = "Time_series_forecasting/a. Input time series sequences"
    sq_filename = '%s/%s/data.mat' % (input_sq_dir, input_sq_name)
    pred_filename = os.path.join(os.path.dirname(__file__), pred_sq_filename)

    dim_idx = 6 # z coordinate of marker 1 (according to resample_time_series_data.py)

    # Loading the original data
    time_data_mat = scipy.io.loadmat(sq_filename)
    org_time_data = time_data_mat['org_data']

    # Loading the predicted data
    time_data_mat = scipy.io.loadmat(pred_filename)
    pred_time_data = time_data_mat['Ypred']

    Tmax = org_time_data.shape[1]

    # Initializing the interpolated data array
    data_dim = org_time_data.shape[0]

    t = np.array(range(0, Tmax))  # Time points
    original_signal = org_time_data[dim_idx, :]
    
    # mock predicted signal
    # predicted_signal = np.array([0] + [original_signal[tau - 1] for tau in range(1, Tmax)]) # mock prdiction lagging 1 time step behind
    predicted_signal = np.zeros_like(t)
    nb_predictions = pred_time_data.shape[1]

    # number of initial time steps to skip for the predicted signal filled with zeros (warm-up period)
    m = Tmax - nb_predictions  # You can adjust m depending on how long your warm-up period is

    predicted_signal[m:]= pred_time_data[dim_idx, :]

    # Set the prediction horizon (e.g., h = 1)
    horizon = 7

    # Set the number of x-axis units to show in the window (e.g., 7 units)
    nb_units = 28

    # Set the manual start time (e.g., start from time step index 10)
    start_time = nb_units + 1  # This can be set higher than nb_units

    # Set the figure width and height (e.g., 4*6 inches)
    figsize = (4, 6)

    # Call the animation function with the specified horizon
    animate_signals(original_signal, predicted_signal, t, horizon, nb_units, figsize, m, start_time)

    print("GIF animation saved as 'signals_animation_with_horizon.gif'")