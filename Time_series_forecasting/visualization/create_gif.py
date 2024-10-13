from functools import partial
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use the Agg backend for rendering on headless systems
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import scipy.io

ORG_DATA_KEY = "org_data"
PRED_DATA_KEY = "Ypred"
TIME_IDX = 1

# to modify manually - ideally load from JSON if code gets improved
parameters = {
    "display": {
        "dim_idx_to_plot": 6,  # z coordinate of marker 1 (according to resample_time_series_data.py)
        "horizon": 7,  # the prediction horizon - needs to match that of the file corresponding to the prediction
        "nb_displayed_points": 28,  # number of x-axis units to show in the window
        "figsize": (4, 6),
        "start_time": None,  # one can manually set it so that the 2nd frame corresponds to beginning of the prediction
        "line_properties": {
            "gt": {"lw": 2, "color": "black"},
            "prediction": {"lw": 1, "color": "red"},
        },  # lw: line width
        "delay_ms": 20,
        "fps": 3,  # acquisition frequency
        "init_xlim": (0, 5),
        "y_lim_margin_coeff": 0.1,
    },
    "paths": {
        "input_sq_dir": "Time_series_forecasting/a. Input time series sequences",
        "input_sq_name": "Ext markers seq 1  3.33 Hz",  # seq 3 has few time points so good for experimenting...
        "input_sq_mat_filename": "data.mat",
        "pred_sq_filename": "pred_result_variables Ext markers seq 1  3.33 Hz tmax_pred=740 DNI k=12 q=180 eta=0.01 sg=0.02 grd_tshld=100 h=7 nrlzed data.mat",
        "out_gif_filename": "signals_animation.gif",
    },
}


class ForecastingAnimation:

    def __init__(self, params):

        self.params = params

        # setting paths properly
        sq_filename = os.path.join(
            params["paths"]["input_sq_dir"],
            params["paths"]["input_sq_name"],
            params["paths"]["input_sq_mat_filename"],
        )
        crt_directory = os.path.dirname(__file__)
        pred_filename = os.path.join(crt_directory, params["paths"]["pred_sq_filename"])
        self.params["paths"]["out_gif_path"] = os.path.join(crt_directory, self.params["paths"]["out_gif_filename"])

        dim_idx = params["display"]["dim_idx_to_plot"]

        # Loading the original data
        org_data_mat = scipy.io.loadmat(sq_filename)
        org_time_data = org_data_mat[ORG_DATA_KEY]

        # Loading the predicted data
        pred_data_mat = scipy.io.loadmat(pred_filename)
        pred_time_data = pred_data_mat[PRED_DATA_KEY]

        # Number of timepoints in the original signal
        Tmax = org_time_data.shape[TIME_IDX]

        # Array representing all the timepoints
        self.t = np.array(range(0, Tmax))

        # Extracting the original signal along the dimension of interest
        self.original_signal = org_time_data[dim_idx, :]

        # Extracting predicted signal along the dimension of interest and align it with the original signal
        self.predicted_signal = np.zeros_like(self.t)
        nb_predictions = pred_time_data.shape[TIME_IDX]
        warm_up_length = (
            Tmax - nb_predictions
        )  # nb of init time steps to skip for the predicted signal filled with zeros
        self.predicted_signal[warm_up_length:] = pred_time_data[dim_idx, :]

        # start at the point where prediction begins by default; to center at x=0 at t = 0, one can choose start_time = warm_up_length instead
        if self.params["display"]["start_time"] is None:
            self.params["display"]["start_time"] = warm_up_length

        self.original_line, self.predicted_line = None, None

    def animate_signals(self):
        """Create the animation function"""

        # Compute the dynamic y-axis limits
        y_min, y_max = self.get_dynamic_ylim()

        # Create the figure and axis for the animation
        fig, ax = plt.subplots(figsize=self.params["display"]["figsize"])

        # Set x-axis and y-axis labels
        ax.set_xlabel("Time step index")
        ax.set_ylabel("Marker z coordinate (mm)")

        ax.set_xlim(*self.params["display"]["init_xlim"])  # Initially set the x-axis limit (this will move)
        ax.set_ylim(y_min, y_max)  # Set the y-axis limit based on the signals

        # Plot empty lines for the original and predicted signals
        (self.original_line,) = ax.plot(
            [], [], label="Original Signal", **self.params["display"]["line_properties"]["gt"]
        )
        (self.predicted_line,) = ax.plot(
            [], [], label="Predicted Signal", **self.params["display"]["line_properties"]["prediction"]
        )

        # Add a legend
        ax.legend()

        # Create the animation
        frames_indices = range(self.params["display"]["start_time"], len(self.t))
        anim = FuncAnimation(
            fig,
            partial(self.update, ax),
            frames=frames_indices,
            init_func=partial(self.init),
            blit=True,
            interval=self.params["display"]["delay_ms"],
        )

        # Save the animation as a GIF
        anim.save(self.params["paths"]["out_gif_path"], writer=PillowWriter(fps=self.params["display"]["fps"]))

    def get_dynamic_ylim(self):
        """Compute the dynamic y-axis limits"""

        signal_min = min(np.min(self.original_signal), np.min(self.predicted_signal))
        signal_max = max(np.max(self.original_signal), np.max(self.predicted_signal))

        # Add a margin (10% of the signal range)
        margin = self.params["display"]["y_lim_margin_coeff"] * (signal_max - signal_min)

        # Return the limits with the margin applied
        return signal_min - margin, signal_max + margin

    def init(self):
        """Initialize the lines (empty)"""

        self.original_line.set_data([], [])
        self.predicted_line.set_data([], [])
        return self.original_line, self.predicted_line

    def update(self, ax, frame):
        """Update function for each frame of the animation"""

        current_time = self.t[frame]
        ax.set_xlim(current_time - self.params["display"]["nb_displayed_points"], current_time)

        # Update the original signal to stop at (frame - horizon)
        idx_relative_to_hrz = frame - self.params["display"]["horizon"]
        if idx_relative_to_hrz > 0:
            self.original_line.set_data(self.t[:idx_relative_to_hrz], self.original_signal[:idx_relative_to_hrz])
        else:
            self.original_line.set_data([], [])

        # Only display the predicted signal after the warm_up_length
        warm_up_length = self.params["display"]["start_time"]
        if frame > warm_up_length:
            self.predicted_line.set_data(self.t[warm_up_length:frame], self.predicted_signal[warm_up_length:frame])
        else:
            self.predicted_line.set_data([], [])

        return self.original_line, self.predicted_line


# Main code
if __name__ == "__main__":

    animation = ForecastingAnimation(params=parameters)
    animation.animate_signals()
    print(f"GIF animation saved as {parameters['paths']['out_gif_filename']}")
