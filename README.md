# 2D frame forecasting in cine MRI

## Overview

This repository contains code for multivariate [time-series forecasting](https://github.com/pohl-michel/2D-MR-image-prediction/tree/main/Time_series_forecasting) and video forecasting. The following methods are implemented for time-series forecasting:
 -  online learning algorithms for RNNs (sequence-specific models):
     - real-time recurrent learning (RTRL)
     - [unbiased online recurrent optimization (UORO)](https://arxiv.org/abs/1702.05043)
     - [sparse one-step approximation (SnAp-1)](https://arxiv.org/abs/2006.07232)
     - [decoupled neural interfaces (DNI)](http://proceedings.mlr.press/v70/jaderberg17a.html)
 - encoder-only transformers (population-based and sequence-specific models).
 - linear baselines: 
     - ordinary least-squares (OLS) regression
     - least-mean-squares (LMS)

For video prediction, time-dependent, dense deformation fields are estimated using the Lucas–Kanade optical-flow algorithm. Then, we forecast the motion projection onto the low-dimensional PCA subspace using one of the algorithms mentioned above. Lastly, the initial frame is warped using the predicted deformations to obtain the predicted frames.

The domain application is respiratory motion forecasting, as this code addresses prediction of the positions of external chest and abdomen markers as well as chest 2D cine-MRI sequences. However, the algorithms implemented are more general and can be applied to the prediction of any time series and quasi-periodic, simple videos. 

This repository contains two main folders:
 - "Time_series_forecasting": self-contained code for time-series forecasting.
 - "Image_prediction": code for cine-MR image forecasting. This folder calls functions from "Time_series_forecasting" to predict the low-dimensional representation of motion in the PCA subspace.

Readers interested mainly in time-series forecasting should refer to the [README.md file in the folder "Time_series_forecasting"](https://github.com/pohl-michel/2D-MR-image-prediction/blob/main/Time_series_forecasting/README.md); this README focuses on video forecasting.

<p align="center">
  <img src="Image_prediction/visualization/2._sq_sl010_sag_Xcs=125_SnAp-1_k=12_q=110_eta=0.02_sg=0.02_h=6_3_cpts_t=181_to_200_cropped.gif" width="45%" alt="Chest cine-MRI frame forecast example 1">
  <img src="Image_prediction/visualization/3._sq_sl010_sag_Xcs=80_SnAp-1_k=30_q=70_eta=0.02_sg=0.02_h=6_4_cpts_t=181_to_200_cropped.gif" width="45%" alt="Chest cine-MRI frame forecast example 2">
  <br>
  <img src="Image_prediction/visualization/4. sq sl014 sag Xcs=165 SnAp-1 k=6 q=110 eta=0.01 sg=0.02 h=6 3 cpts_t=181_to_200_cropped.gif" width="45%" alt="Liver cine-MRI frame forecast example 1">
  <img src="Image_prediction/visualization/5. sq sl014 sag Xcs=95 SnAp-1 k=6 q=110 eta=0.01 sg=0.02 h=6 2 cpts_t=181_to_200_cropped.gif" width="45%" alt="Liver cine-MRI frame forecast example 2">
</p>
<p align="center"> <em>Representative cine-MRI frame forecasts for thoraco-abdominal sequences. <br> Each panel compares ground truth (left side) and predicted frames using SnAp-1 (right side) at a fixed horizon of h=6 time steps.</em> </p>


## How to run the image-prediction pipeline

Image sequence prediction can be performed from MATLAB by executing "Image_prediction/image_prediction_main.m". 

Its behavior is governed by the `beh_par` structure array in "load_impred_behavior_parameters.m". The main execution modes are:
 1. `beh_par.OPTIMIZE_NB_PCA_CP = true`: optimization of the forecasting hyperparameters, including the PCA subspace dimension, $n_{\text{cp}}$.
 2. `beh_par.OPTIMIZE_NB_PCA_CP = false`: inference using the selected prediction parameters.


## Data

The cine-MRI sequences included in this repository were obtained by preprocessing data from the following public datasets:
 - chest MRI: [4D MRI lung data](https://vision.ee.ethz.ch/datsets.html) from ETH Zürich.
 - liver MRI: [2D navigator frames](http://open-science.ub.ovgu.de/xmlui/handle/684882692/88) from Otto-von-Guericke University Magdeburg.

The input images loaded by "image_prediction_main.m" are located in "Image_prediction/input_imgs/2D images". The input sequences are specified in the cell array `input_im_dir_suffix_tab` in "image_prediction_main.m".


## Configuration parameters

Sequence-specific configuration files are located in the corresponding input-image subdirectory:

| Filename         | Parameter scope                                                       |
| --------         | -------                                                               |
| pred_par.xlsx    | Low-dimensional motion representation forecasting                     |
| OF_calc_par.xlsx | Optical flow                                                          |
| disp_par.xlsx    | Figure display and saving                                             |
| im_seq_par.xlsx  | Input image sequence and region of interest used for evaluation |

Additional image-prediction parameters can be configured in the following .m files:

| Filename                                  | Parameter scope          |
| --------                                  | -------                  |
| Image_prediction/load_warp_par.m          | Image warping/sampling   |
| Image_prediction/image_prediction_main.m  | • `pred_meths`: prediction methods used<br>• `br_model_par.nb_pca_cp_tab`: value of $n_{\text{cp}}$ (or maximum value of $n_{\text{cp}}$, $n_{\text{cp}}^{\text{max}}$, in the validation range $\{1, \ldots, n_{\text{cp}}^{\text{max}}\}$ in hyperparameter-optimization mode)|
| Time_series_forecasting/load_pred_par.m   | Time-series forecasting parameters, overriding those defined in  "pred_par.xlsx" for the specific input sequence  |
| Time_series_forecasting/load_hyperpar_cv_info.m | Hyperparameter grids for grid search on the validation set |


## Lucas–Kanade optical-flow calculation

The script "Image_prediction/OF2D_param_optim_main.m" runs grid search to optimize optical-flow parameters. The parameter grid is specified in "Image_prediction/load_OFeval_parameters.m".

Optical-flow fields can also be computed from "Image_prediction/image_prediction_main.m" using the parameter values in "OF_calc_par.xlsx" by setting:
`beh_par.COMPUTE_OPTICAL_FLOW = true` in "load_impred_behavior_parameters.m".


## Requirements

The main image-prediction workflow is written in MATLAB. Transformers were implemented using Python code (importing the PyTorch/Optuna libraries) interfaced with the common backbone evaluation code in MATLAB. If using conda or miniconda, PyTorch can be installed for example with:

`conda install pytorch pytorch-cuda -c pytorch -c nvidia` (when using an NVIDIA GPU, Anaconda prompt)

The [Python interpreter used by MATLAB](https://uk.mathworks.com/help/matlab/ref/pyenv.html) can then be set with:

`pyenv("Version", "C:\Users\username\miniconda3\envs\my_environment\python.exe");` (MATLAB command line)

Hyperparameter optimization uses parallel processing through `parfor` loops in MATLAB. The code can be run without the MATLAB Parallel Computing Toolbox by replacing `parfor` loops with `for` loops, at the cost of longer runtime.


## References

This repository supports the claims in the following research articles. Please cite them if you use this code in your research.
 - Michel Pohl, Mitsuru Uesaka, Hiroyuki Takahashi, Kazuyuki Demachi, Ritu Bhusal Chhatkuli, ["Real-time respiratory motion forecasting with online learning of recurrent neural networks for accurate targeting in externally guided radiotherapy"](https://doi.org/10.1016/j.cmpb.2025.108828), Computer Methods and Programs in Biomedicine (2025) [[arXiv]](https://doi.org/10.48550/arXiv.2403.01607)
 - Michel Pohl, Mitsuru Uesaka, Hiroyuki Takahashi, Kazuyuki Demachi, Ritu Bhusal Chhatkuli, ["Frame forecasting in cine MRI using the PCA respiratory motion model: comparing recurrent neural networks trained online and transformers"](https://doi.org/10.1016/j.compmedimag.2026.102755), Computerized Medical Imaging and Graphics (2026) [[arXiv]](https://doi.org/10.48550/arXiv.2410.05882)

A detailed description of the Lucas–Kanade registration algorithm used as a basis for this repository can be found in:
 - Michel Pohl, Mitsuru Uesaka, Kazuyuki Demachi, and Ritu Bhusal Chhatkuli, ["Prediction of the motion of chest internal points using a recurrent neural network trained with real-time recurrent learning for latency compensation in lung cancer radiotherapy"](https://doi.org/10.1016/j.compmedimag.2021.101941), Computerized Medical Imaging and Graphics, 2021 [[arXiv]](https://doi.org/10.48550/arXiv.2207.05951). 