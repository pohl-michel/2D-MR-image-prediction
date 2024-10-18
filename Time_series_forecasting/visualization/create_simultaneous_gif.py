# Animated plot of the ground-truth vs predicted coordinates for all objects and directions (adaptation of create_gif.py)
# The field "pred_sq_filename" of parameters["paths"] corresponds to the prediction result (.mat file in "d. RNN variables (temp)")
# saved when running signal_prediction_main.m (with beh_par.SAVE_PRED_RESULTS set to "true").
#
# Note using the gifsicle program in linux can help reduce the size of the output gif, for instance:
# gifsicle -O3 --colors 16 --interlace --threads my_input.gif -o compressed_output.gif
#
# Author: Michel Pohl
# License : 3-clause BSD License

from abc import ABC, abstractmethod
from functools import partial
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use the Agg backend for rendering on headless systems
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import scipy.io

ORG_DATA_KEY = "org_data"
PRED_DATA_KEY = "Ypred"
DIM_IDX, TIME_IDX = 0, 1
T_MAX = 50  # Plots only the first T_MAX time steps (for debugging for instance) - None if plotting entire sequence

# JSON config file to load - can be configured manually
# JSON_CONFIG_FILENAME = "external_markers_sq_1_config.json"
JSON_CONFIG_FILENAME = "third_im_seq_pca_weight_pred_config.json"


class ForecastingAnimation(ABC):

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

        # Dimensionality of the data - we assume it's the same as in the predicted signal
        self.data_dim = self.org_time_data.shape[DIM_IDX]

        # Number of timepoints in the original signal - we assume it's the same as in the predicted signal
        self.Tmax = self.org_time_data.shape[TIME_IDX]

        # Determine warm-up length for the prediction
        nb_predictions = self.pred_time_data.shape[TIME_IDX]
        self.warm_up_length = self.Tmax - nb_predictions

        if T_MAX is not None:
            self.Tmax = T_MAX
            self.org_time_data = self.org_time_data[:, :T_MAX]
            self.pred_time_data = self.pred_time_data[:, : T_MAX - self.warm_up_length]

        # Array representing all the timepoints
        self.t = np.arange(0, self.Tmax)

        # Ensure default start time if not set
        if self.params["display"]["start_time"] is None:
            self.params["display"]["start_time"] = self.warm_up_length

        # Preallocate arrays for lines and axes
        self.fig, self.axes = None, None
        self.lines_gt, self.lines_pred = [], []  # Ground truth and predicted lines

    def animate_signals(self):
        """Create the animation function"""

        # Create the figure and grid of axes
        self._create_subplots()

        # Apply spacing and margins from parameters["display"]
        plt.subplots_adjust(
            wspace=self.params["display"]["wspace"],  # horizontal spacing between subplots
            hspace=self.params["display"]["hspace"],  # vertical spacing between subplots
            left=self.params["display"]["left"],  # left margin
            right=self.params["display"]["right"],  # right margin
            top=self.params["display"]["top"],  # top margin
            bottom=self.params["display"]["bottom"],  # bottom margin
        )

        # Compute dynamic y-limits for each subplot
        y_lims = self.get_dynamic_ylims()

        # Initialize all 9 subplots (3 objects × 3 coordinates)
        for idx in range(self.data_dim):
            ax = self._get_ax_from_dim_idx(idx)

            # Set x-axis and y-axis labels
            ax.set_xlabel("Time step index", fontsize=self.params["display"]["fontsize"]["xy_labels"])
            ax.set_ylabel(self._get_ylabel_from_dim_idx(idx), fontsize=self.params["display"]["fontsize"]["xy_labels"])

            # Set initial limits
            ax.set_xlim(*self.params["display"]["init_xlim"])
            ax.set_ylim(y_lims[idx])

            # Adjust the font size of the tick labels on both x and y axes
            ax.tick_params(axis="both", which="major", labelsize=self.params["display"]["fontsize"]["tick_labels"])

            # Plot empty lines for the original and predicted signals
            (gt_line,) = ax.plot([], [], label="Original Signal", **self.params["display"]["line_properties"]["gt"])
            (pred_line,) = ax.plot(
                [], [], label="Predicted Signal", **self.params["display"]["line_properties"]["prediction"]
            )
            self.lines_gt.append(gt_line)
            self.lines_pred.append(pred_line)

            # Add a legend to each subplot
            ax.legend(fontsize=self.params["display"]["fontsize"]["legend"])

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

        # Save the animation as a GIF - fps ideally corresponds to the acquisition frequency in Hz
        anim.save(self.params["paths"]["out_gif_path"], writer=PillowWriter(fps=self.params["display"]["fps"]))

    def get_dynamic_ylims(self):
        """Compute dynamic y-axis limits for each of the 9 subplots"""

        y_lims = []
        for idx in range(self.data_dim):

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

        for idx in range(self.data_dim):
            ax = self._get_ax_from_dim_idx(idx)

            current_time = self.t[frame]

            # Set the moving window for the x-axis - # nb_displayed_points is nb of x-axis units to show in the window
            ax.set_xlim(current_time - self.params["display"]["nb_displayed_points"], current_time)

            # Update the original signal to stop at (frame - horizon)
            idx_relative_to_hrz = frame - self.params["display"]["horizon"]
            if idx_relative_to_hrz > 0:
                self.lines_gt[idx].set_data(self.t[:idx_relative_to_hrz], self.org_time_data[idx, :idx_relative_to_hrz])
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

    @abstractmethod
    def _create_subplots(self):
        """Updates self.fig and self.axes"""

    @abstractmethod
    def _get_ax_from_dim_idx(self, idx):
        """Return the current ax object given the provided dimension index"""

    @abstractmethod
    def _get_ylabel_from_dim_idx(self, idx):
        """Return the ylabel given the provided dimension index"""


class ForecastingAnimation3DObjects(ForecastingAnimation):

    POS_DIMENSIONALITY = 3  # 3D position

    def __init__(self, params):
        super().__init__(params)

        # Number of objects (we assume the division is exact and there is no remainder)
        self.nb_obj = self.data_dim // self.POS_DIMENSIONALITY

    def _get_crd_obj_from_idx(self, idx: int) -> tuple[int]:
        """Returns the coordinate and object index from the dimension index in the initial data (if prediction of 3D
        objects)."""

        coord = idx // self.POS_DIMENSIONALITY
        obj = idx % self.POS_DIMENSIONALITY

        return coord, obj

    def _create_subplots(self):

        self.fig, self.axes = plt.subplots(
            nrows=self.POS_DIMENSIONALITY,
            ncols=self.nb_obj,
            squeeze=False,
            figsize=tuple(self.params["display"]["figsize"]),
        )

    def _get_ax_from_dim_idx(self, idx):

        coord, obj = self._get_crd_obj_from_idx(idx)
        ax = self.axes[coord, obj]

        return ax

    def _get_ylabel_from_dim_idx(self, idx):

        coord, obj = self._get_crd_obj_from_idx(idx)
        ylabel = f"{['x', 'y', 'z'][coord]} coordinate of marker {obj+1}"

        return ylabel


class ForecastingAnimationPCA(ForecastingAnimation):

    N_ROWS = 1

    def _create_subplots(self):

        self.fig, self.axes = plt.subplots(
            nrows=self.N_ROWS,
            ncols=self.data_dim,
            figsize=tuple(self.params["display"]["figsize"]),
        )

    def _get_ax_from_dim_idx(self, idx):
        return self.axes[idx]

    def _get_ylabel_from_dim_idx(self, idx):
        return f"PCA weight of order {idx}"


# Main code
if __name__ == "__main__":

    json_config_path = os.path.join(os.path.dirname(__file__), JSON_CONFIG_FILENAME)
    with open(json_config_path, "r") as parameters_file:
        parameters = json.load(parameters_file)

    if parameters["object_3d_pos"]:
        animation = ForecastingAnimation3DObjects(params=parameters)
    else:
        animation = ForecastingAnimationPCA(params=parameters)

    animation.animate_signals()
    print(f"GIF animation saved as {parameters['paths']['out_gif_filename']}")
