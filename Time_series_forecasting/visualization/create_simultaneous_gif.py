# Animated plot of the ground-truth vs predicted coordinates for all objects and directions (adaptation of create_gif.py)
# Author: Michel Pohl
# License : 3-clause BSD License

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
        "horizon": 7,  # the prediction horizon
        "nb_displayed_points": 28,  # number of x-axis units to show in the window
        "figsize": (12, 12),  # Increased figure size for 9 subplots
        "start_time": None,  # one can manually set it so that the 2nd frame corresponds to the beginning of the prediction
        "line_properties": {
            "gt": {"lw": 2, "color": "black"},  # Ground truth (original) line properties
            "prediction": {"lw": 1, "color": "red"},  # Prediction line properties
        },
        "delay_ms": 20,
        "fps": 3,  # acquisition frequency
        "init_xlim": (0, 5),  # Initial x-axis limit (moving window)
        "y_lim_margin_coeff": 0.1,  # Margin for y-limits (10% of signal range)
        "wspace": 0.4,  # horizontal spacing between subplots
        "hspace": 0.3,  # vertical spacing between subplots
        "left": 0.08,  # left margin
        "right": 0.95,  # right margin
        "top": 0.95,  # top margin
        "bottom": 0.08,  # bottom margin
    },
    "paths": {
        "input_sq_dir": "Time_series_forecasting/a. Input time series sequences",
        "input_sq_name": "Ext markers seq 1  3.33 Hz",
        "input_sq_mat_filename": "data.mat",
        "pred_sq_filename": "pred_result_variables Ext markers seq 1  3.33 Hz tmax_pred=740 DNI k=12 q=180 eta=0.01 sg=0.02 grd_tshld=100 h=7 nrlzed data.mat",
        "out_gif_filename": "signals_animation_multivariate.gif",  # Output GIF filename
    },
}


class ForecastingAnimation:

    def __init__(self, params):

        self.params = params

        # Setting paths properly
        sq_filename = os.path.join(
            params["paths"]["input_sq_dir"],
            params["paths"]["input_sq_name"],
            params["paths"]["input_sq_mat_filename"],
        )
        crt_directory = os.path.dirname(__file__)
        pred_filename = os.path.join(crt_directory, params["paths"]["pred_sq_filename"])
        self.params["paths"]["out_gif_path"] = os.path.join(crt_directory, self.params["paths"]["out_gif_filename"])

        # Loading the original data
        org_data_mat = scipy.io.loadmat(sq_filename)
        self.org_time_data = org_data_mat[ORG_DATA_KEY]  # shape: (num_objects * num_coordinates, Tmax)

        # Loading the predicted data
        pred_data_mat = scipy.io.loadmat(pred_filename)
        self.pred_time_data = pred_data_mat[PRED_DATA_KEY]

        # Number of timepoints in the original signal
        self.Tmax = self.org_time_data.shape[TIME_IDX]

        # Array representing all the timepoints
        self.t = np.arange(0, self.Tmax)

        # Determine warm-up length for the prediction
        nb_predictions = self.pred_time_data.shape[TIME_IDX]
        self.warm_up_length = self.Tmax - nb_predictions

        # Ensure default start time if not set
        if self.params["display"]["start_time"] is None:
            self.params["display"]["start_time"] = self.warm_up_length

        # Preallocate arrays for lines and axes
        self.fig, self.axes = None, None
        self.lines_gt, self.lines_pred = [], []  # Ground truth and predicted lines

    def animate_signals(self):
        """Create the animation function"""

        # Create the figure and 3x3 grid of axes
        self.fig, self.axes = plt.subplots(3, 3, figsize=self.params["display"]["figsize"])

        # Apply spacing and margins from parameters["display"]
        plt.subplots_adjust(
            wspace=self.params["display"]["wspace"],
            hspace=self.params["display"]["hspace"],
            left=self.params["display"]["left"],
            right=self.params["display"]["right"],
            top=self.params["display"]["top"],
            bottom=self.params["display"]["bottom"],
        )

        # Compute dynamic y-limits for each subplot
        y_lims = self.get_dynamic_ylims()

        # Initialize all 9 subplots (3 objects × 3 coordinates)
        for coord in range(3):
            for obj in range(3):
                idx = coord * 3 + obj
                ax = self.axes[coord, obj]

                # Set x-axis and y-axis labels
                ax.set_xlabel("Time step index")
                ax.set_ylabel(f"{['x', 'y', 'z'][coord]} coordinate of marker {obj+1}")

                # Set initial limits
                ax.set_xlim(*self.params["display"]["init_xlim"])
                ax.set_ylim(y_lims[idx])

                # Plot empty lines for the original and predicted signals
                (gt_line,) = ax.plot([], [], label="Original Signal", **self.params["display"]["line_properties"]["gt"])
                (pred_line,) = ax.plot(
                    [], [], label="Predicted Signal", **self.params["display"]["line_properties"]["prediction"]
                )

                self.lines_gt.append(gt_line)
                self.lines_pred.append(pred_line)

                # Add a legend to each subplot
                ax.legend()

        # Create the animation
        frames_indices = range(self.params["display"]["start_time"], len(self.t))
        anim = FuncAnimation(
            self.fig,
            partial(self.update),
            frames=frames_indices,
            init_func=self.init,
            blit=True,
            interval=self.params["display"]["delay_ms"],
        )

        # Save the animation as a GIF
        anim.save(self.params["paths"]["out_gif_path"], writer=PillowWriter(fps=self.params["display"]["fps"]))

    def get_dynamic_ylims(self):
        """Compute dynamic y-axis limits for each of the 9 subplots"""

        y_lims = []
        for coord in range(3):
            for obj in range(3):
                idx = coord * 3 + obj

                # Extract corresponding signals
                original_signal = self.org_time_data[idx, :]
                predicted_signal = np.zeros_like(self.t)
                predicted_signal[self.warm_up_length :] = self.pred_time_data[idx, :]

                # Calculate min and max with margin
                signal_min = min(np.min(original_signal), np.min(predicted_signal))
                signal_max = max(np.max(original_signal), np.max(predicted_signal))
                margin = self.params["display"]["y_lim_margin_coeff"] * (signal_max - signal_min)
                y_lims.append((signal_min - margin, signal_max + margin))

        return y_lims

    def init(self):
        """Initialize all the lines to be empty"""

        for gt_line, pred_line in zip(self.lines_gt, self.lines_pred):
            gt_line.set_data([], [])
            pred_line.set_data([], [])
        return self.lines_gt + self.lines_pred

    def update(self, frame):
        """Update function for each frame"""

        for coord in range(3):
            for obj in range(3):
                idx = coord * 3 + obj
                ax = self.axes[coord, obj]
                current_time = self.t[frame]

                # Set the moving window for the x-axis
                ax.set_xlim(current_time - self.params["display"]["nb_displayed_points"], current_time)

                # Update the original signal to stop at (frame - horizon)
                idx_relative_to_hrz = frame - self.params["display"]["horizon"]
                if idx_relative_to_hrz > 0:
                    self.lines_gt[idx].set_data(
                        self.t[:idx_relative_to_hrz], self.org_time_data[idx, :idx_relative_to_hrz]
                    )
                else:
                    self.lines_gt[idx].set_data([], [])

                # Update the predicted signal (after the warm-up length)
                if frame > self.warm_up_length:
                    self.lines_pred[idx].set_data(
                        self.t[self.warm_up_length : frame], self.pred_time_data[idx, : frame - self.warm_up_length]
                    )
                else:
                    self.lines_pred[idx].set_data([], [])

        return self.lines_gt + self.lines_pred


# Main code
if __name__ == "__main__":

    animation = ForecastingAnimation(params=parameters)
    animation.animate_signals()
    print(f"GIF animation saved as {parameters['paths']['out_gif_filename']}")
