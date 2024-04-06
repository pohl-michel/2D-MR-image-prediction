# Script loading multidimensional time series data and performs either of the following:
# - it upsamples the data using interpolation with cubic splines and adds noise to the interpolated missing data
# - it downsamples the data
# This script also plots the resampled data.
#
# Note : the marker position data has the following form :
#            [ x_1(t_1), ..., x_1(t_M)]
#            [ x_2(t_1), ..., x_2(t_M)]
#            [ x_3(t_1), ..., x_3(t_M)]
#            [ y_1(t_1), ..., y_1(t_M)]
# data     = [ y_2(t_1), ..., y_2(t_M)]
#            [ y_3(t_1), ..., y_3(t_M)]
#            [ z_1(t_1), ..., z_1(t_M)]
#            [ z_2(t_1), ..., z_2(t_M)]
#            [ z_3(t_1), ..., z_3(t_M)]
# The data structure (x,y,z) is used for plotting, but resampling works regardless of that structure
#
#
# Author : Pohl Michel
# Date : October 4, 2022
# Version : v2.0
# License : 3-clause BSD License


import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import os


# Behavior
SAVE_DATA = True
DISPLAY_FIGURES = True

# Parameters
# resample_factor = 3
resample_factor = 1/3 
    # If the original data has t values, the output data will have t*resample_factor values along the time axis
    # If resample_factor < 1, then this script performs downsampling
noise_factor = 1/150         # We set the Gaussian noise std deviation as sg = pos_amplitude*noise_factor
num_precision = 0.1          # Numerical precision of the input and output data (e.g., 0.1 -> 1 decimal digit precision)
input_frequency = 10         # Frequency of the input data in Hz, used for the output filenames.
crd_idx_zoom_plot = 6        # z1 # Coordinate index of the first plot

# Paths
input_sq_name = "Ext markers seq 9"
input_sq_dir = "Time_series_forecasting/a. Input time series sequences"
sq_filename = '%s/%s/data.mat' % (input_sq_dir, input_sq_name)


# Loading the original data
time_data_mat = scipy.io.loadmat(sq_filename)
org_time_data = time_data_mat['org_data']

org_Tmax = org_time_data.shape[1]
new_Tmax = int(resample_factor*org_Tmax) #int in case we are downsampling data

# Initializing the interpolated data array
data_dim = org_time_data.shape[0]
resampled_data = np.zeros((data_dim, new_Tmax))

new_frequency = input_frequency*resample_factor # in Hz

if resample_factor < 1:
    print("Downsampling data")
    for t in range(new_Tmax):
        t_org = int(t/resample_factor)
        resampled_data[:, t] = org_time_data[:, t_org]
    
    out_dir_name = '%s/%s %.2f Hz' % (input_sq_dir, input_sq_name, new_frequency)
    new_data_plot_title = f"Downsampled data ({new_frequency:.2f} Hz)"

else: 
    print("Upsampling data")

    # Copying the input data in a temporary array
    temp = np.zeros((data_dim, new_Tmax))
    for t in range(org_Tmax):
        temp[:, resample_factor * t] = org_time_data[:, t]
        for delta_t in range(resample_factor - 1):
            temp[:, resample_factor * t + delta_t + 1] = np.nan   # broadcasting

    # Upsampling the input data using cubic interpolation with Pandas
    for dim_idx in range(data_dim):
        upsampled_series = pd.Series(temp[dim_idx, :])
        upsampled_series = upsampled_series.interpolate(method = 'cubicspline')
            # ref 1: https://pandas.pydata.org/docs/reference/api/pandas.core.resample.Resampler.interpolate.html
            # ref 2: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html#scipy.interpolate.CubicSpline
        resampled_data[dim_idx, :] = upsampled_series.to_numpy()

    # Initializing the output data (interpolated data with additional noise)
    pos_amplitude = np.amax(org_time_data, axis=1) - np.amin(org_time_data, axis=1)

    # Adding noise to the upsampled data
    for dim_idx in range(data_dim):
        noise_new = np.random.normal(0, noise_factor*pos_amplitude[dim_idx], (1, new_Tmax))
        for t in range(org_Tmax):
            noise_new[0, resample_factor * t] = 0
        resampled_data[dim_idx, :] = resampled_data[dim_idx, :] + noise_new

    # Truncating output data to the precision specified by num_precision
    resampled_data = num_precision*np.floor(resampled_data//num_precision)

    out_dir_name = '%s/%s %d Hz noise factor %f' % (input_sq_dir, input_sq_name, new_frequency, noise_factor)
    new_data_plot_title = f"Upsampled data ({new_frequency} Hz, cubic spline interpolation, Gaussian noise)"


if SAVE_DATA:

    # Creating the output directory
    os.makedirs(out_dir_name)

    # Saving the output data
    output_mat_name = '%s/data.mat' % out_dir_name
    scipy.io.savemat(output_mat_name, {'org_data': resampled_data})


if DISPLAY_FIGURES:

    # At first we display a zoomed image between t=0s and t=10s (not the entire time series) for a particular dimension
    t_max_zoom_s = 10 # in s
    t_max_zoom_org = t_max_zoom_s*input_frequency                      # nb of time steps
    t_max_zoom_new = int(t_max_zoom_s*input_frequency*resample_factor) # nb of time steps

    t_new = np.linspace(0, t_max_zoom_s, t_max_zoom_new, endpoint=True)
    t_org = np.linspace(0, t_max_zoom_s, t_max_zoom_org, endpoint=True)

    f = plt.figure(1)
    plt.plot(t_new, resampled_data[crd_idx_zoom_plot, :t_max_zoom_new], ".") # z1
    plt.title(new_data_plot_title)
    plt.xlabel('Time (s)')
    plt.ylabel('z coordinate (mm)')
    f.show()

    f2 = plt.figure(2)
    plt.plot(t_org, org_time_data[crd_idx_zoom_plot, :t_max_zoom_org], ".")  # z1
    plt.title("Original data at 10 Hz")
    plt.xlabel('Time (s)')
    plt.ylabel('z coordinate (mm)')
    f2.show()

    t_max_s = org_Tmax/input_frequency # in s
    t_new = np.linspace(0, t_max_s, new_Tmax)
    t_org = np.linspace(0, t_max_s, org_Tmax)

    coord_idx_dict = {
        0 : "x",
        1 : "y",
        2 : "z"
    }

    for coord_idx in range(3):
        for mkr_idx in range(3):

            g1 = plt.figure(3 + 2 * mkr_idx + 1 + coord_idx*6)
            plt.plot(t_org, org_time_data[coord_idx*3 + mkr_idx, :], ".")
            plt.title("Original data at 10 Hz")
            plt.xlabel('Time (s)')
            plt.ylabel('%s coordinate (mm) of marker %d' % (coord_idx_dict[coord_idx], mkr_idx + 1))
            g1.show()

            g2 = plt.figure(3 + 2 * mkr_idx + 2 + coord_idx*6)
            plt.plot(t_new/3, resampled_data[coord_idx*3 + mkr_idx, :], ".")
            plt.title(new_data_plot_title)
            plt.xlabel('Time (s)')
            plt.ylabel('%s coordinate (mm) of marker %d' % (coord_idx_dict[coord_idx], mkr_idx + 1))
            g2.show()

    plt.show()
    plt.ion