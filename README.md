# Respiratory motion signal and 2D cine-MRI forecasting using RNNs trained online and transformers


## Overview

This repository contains code for multivariate [time-series forecasting](https://github.com/pohl-michel/2D-MR-image-prediction/tree/main/Time_series_forecasting) using RNNs trained online and transformers, and thoraco-abdominal cine-MR image prediction, combining dense motion-field estimation, PCA-based motion modeling, temporal dynamics prediction, and image warping.

The following methods are implemented for time-series forecasting:
 -  online learning algorithms for RNNs (sequence-specific models):
     - real-time recurrent learning (RTRL)
     - [unbiased online recurrent optimization (UORO)](https://arxiv.org/abs/1702.05043)
     - [sparse one-step approximation (SnAp-1)](https://arxiv.org/abs/2006.07232)
     - [decoupled neural interfaces (DNI)](http://proceedings.mlr.press/v70/jaderberg17a.html)
 - encoder-only transformers (population-based and sequence-specific models)
 - linear autoregressive baselines, including ordinary least-squares (OLS) regression and least-mean-squares (LMS)

For cine-MRI prediction, the deformation vector field between the initial and current frames is estimated using the Lucas–Kanade optical-flow algorithm. This vector field is projected onto a low-dimensional subspace computed via PCA, and the projection coordinates are forecast using one of the time-series forecasting algorithms listed above. Lastly, the initial frame is warped using the predicted deformations to generate frame forecasts.

The main application is respiratory motion forecasting, encompassing prediction of thoraco-abdominal external-marker positions and 2D cine-MRI sequences. However, the algorithms implemented are more general and can be applied to the prediction of other multivariate time series and quasi-periodic, simple videos. 

This repository contains two main folders:
 - `Time_series_forecasting`: self-contained code for general-purpose time-series forecasting, including RNNs trained with online learning algorithms, transformer models, and linear baselines.
 - `Image_prediction`: code for cine-MR image forecasting, including optical-flow estimation, PCA-based motion modeling, temporal dynamics forecasting, and image synthesis via initial frame resampling. This folder calls functions from `Time_series_forecasting` to forecast the low-dimensional PCA representation of motion.

Readers interested primarily in time-series forecasting should refer to the [`Time_series_forecasting` README](https://github.com/pohl-michel/2D-MR-image-prediction/blob/main/Time_series_forecasting/README.md). The present README focuses on cine-MR image forecasting.

<p align="center">
  <img src="Image_prediction/visualization/2._sq_sl010_sag_Xcs=125_SnAp-1_k=12_q=110_eta=0.02_sg=0.02_h=6_3_cpts_t=181_to_200_cropped.gif" width="45%" alt="Chest cine-MRI frame forecast example 1">
  <img src="Image_prediction/visualization/3._sq_sl010_sag_Xcs=80_SnAp-1_k=30_q=70_eta=0.02_sg=0.02_h=6_4_cpts_t=181_to_200_cropped.gif" width="45%" alt="Chest cine-MRI frame forecast example 2">
  <br>
  <img src="Image_prediction/visualization/4. sq sl014 sag Xcs=165 SnAp-1 k=6 q=110 eta=0.01 sg=0.02 h=6 3 cpts_t=181_to_200_cropped.gif" width="45%" alt="Liver cine-MRI frame forecast example 1">
  <img src="Image_prediction/visualization/5. sq sl014 sag Xcs=95 SnAp-1 k=6 q=110 eta=0.01 sg=0.02 h=6 2 cpts_t=181_to_200_cropped.gif" width="45%" alt="Liver cine-MRI frame forecast example 2">
</p>
<p align="center"> <em>Representative cine-MR image forecasts using the ETH Zürich dataset. Each panel compares ground-truth frames on the left with SnAp-1 predictions on the right at a fixed horizon of h=6 time steps.</em> </p>


## How to run the image-prediction pipeline

Image sequence prediction can be performed from MATLAB by executing `Image_prediction/image_prediction_main.m`. 

Its behavior is controlled by the `beh_par` structure array in `load_impred_behavior_parameters.m`. The main execution modes are:
 1. `beh_par.OPTIMIZE_NB_PCA_CP = true`: optimization of the forecasting hyperparameters, including the PCA subspace dimension, $n_{\text{cp}}$. This mode can be computationally expensive, especially when evaluating several prediction methods, horizons, PCA-subspace dimensions, and random initializations.
 2. `beh_par.OPTIMIZE_NB_PCA_CP = false`: inference using the inference using the specified prediction parameters; see [Configuration parameters](#configuration-parameters) below.

When `beh_par.IM_PREDICTION` and `beh_par.SAVE_PRED_IM` are both set to `true`, predicted images and performance results are respectively saved in the (automatically created) `tmp_imgs` and `tmp_txt_files` subdirectories of the `Image_prediction` folder, except in hyperparameter tuning mode, where only performance is recorded.

Setting `beh_par.COMPUTE_OPTICAL_FLOW = true` enables optical-flow field estimation at the beginning of the image-prediction pipeline, using the parameter values in `OF_calc_par.xlsx`.

*Note:* The script `Image_prediction/OF2D_param_optim_main.m` runs grid search to optimize optical-flow parameters, using the parameter grid specified in `Image_prediction/load_OFeval_parameters.m`.

## Image data

The cine-MRI sequences included in this repository were obtained by preprocessing data from the following public datasets:
 - chest MRI: [4D MRI lung data](https://vision.ee.ethz.ch/datsets.html) from ETH Zürich.
 - liver MRI: [2D navigator frames](http://open-science.ub.ovgu.de/xmlui/handle/684882692/88) from Otto-von-Guericke University Magdeburg.

The input images loaded by `image_prediction_main.m` are located in `Image_prediction/input_imgs/2D images`. The input sequences are specified in the cell array `input_im_dir_suffix_tab` in `image_prediction_main.m`.


## Configuration parameters

Sequence-specific configuration files are located in the corresponding input image subdirectory (Table 1).

<table align="center">
  <thead>
    <tr>
      <th>Filename</th>
      <th>Parameter scope</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>pred_par.xlsx</code></td>
      <td>Low-dimensional motion-representation forecasting</td>
    </tr>
    <tr>
      <td><code>OF_calc_par.xlsx</code></td>
      <td>Optical-flow field estimation</td>
    </tr>
    <tr>
      <td><code>disp_par.xlsx</code></td>
      <td>Figure display and saving</td>
    </tr>
    <tr>
      <td><code>im_seq_par.xlsx</code></td>
      <td>Properties of the input image sequence and region of interest used for evaluation</td>
    </tr>
  </tbody>
</table>

<p align="center">
  <em>Table 1: Files specifying sequence-specific parameters.</em>
</p>

Additional image-prediction parameters can be configured in the `.m` files listed in Table 2.

<!-- Keeping markdown table because I don't know how to render LaTeX inside html tables inside Markdown -->
| Folder | Filename                         | Parameters          |
| -------- | ------                                 | -------                  |
| `Image_prediction` | `load_impred_behavior_parameters.m`  | `beh_par`: flags defining pipeline behavior  |
|  — | `load_warp_par.m`          | `warp_par`: parameters for image synthesis via warping/sampling   |
|  — | `image_prediction_main.m`  | • `pred_meths`: prediction methods used<br>• `br_model_par.nb_pca_cp_tab`: value of $n_{\text{cp}}$ (or maximum value of $n_{\text{cp}}$, $n_{\text{cp}}^{\text{max}}$, defining the validation range $\{1, \ldots, n_{\text{cp}}^{\text{max}}\}$ in hyperparameter-optimization mode)|
| `Time_series_forecasting` | `load_pred_par.m`   | `pred_par`: time-series forecasting parameters, overriding the sequence-specific ones defined in  `pred_par.xlsx` |
| — | `load_hyperpar_cv_info.m` | `hppars`: hyperparameter grid for grid search on the validation set |

<p align="center">
  <em>Table 2: Additional image-prediction parameters.</em>
</p>


## Requirements

The main image-prediction workflow is written in MATLAB. Transformer-based forecasting requires a Python environment (with PyTorch and Optuna) accessible from MATLAB, as transformer models were implemented using Python code interfaced with the common backbone evaluation code in MATLAB. If using [conda or miniconda](https://docs.conda.io/en/latest/#) and using an NVIDIA GPU, PyTorch can be installed with:

`conda install pytorch pytorch-cuda -c pytorch -c nvidia`

The command above should be run in a conda environment used by the same operating system as MATLAB. On Windows, this is typically done from the Anaconda Prompt.

The [Python interpreter used by MATLAB](https://uk.mathworks.com/help/matlab/ref/pyenv.html) can then be set on MATLAB command line with:

`pyenv("Version", "C:\Users\username\miniconda3\envs\my_environment\python.exe");`

Hyperparameter optimization uses parallel processing through `parfor` loops in MATLAB. The code can be run without the MATLAB Parallel Computing Toolbox by replacing `parfor` loops with `for` loops, at the cost of longer runtime.


## References

This repository supports the claims in the following research articles. Please cite them if you use this code in your research.
 - Michel Pohl, Mitsuru Uesaka, Hiroyuki Takahashi, Kazuyuki Demachi, Ritu Bhusal Chhatkuli, ["Real-time respiratory motion forecasting with online learning of recurrent neural networks for accurate targeting in externally guided radiotherapy"](https://doi.org/10.1016/j.cmpb.2025.108828), Computer Methods and Programs in Biomedicine (2025) [[arXiv]](https://doi.org/10.48550/arXiv.2403.01607)
 - Michel Pohl, Mitsuru Uesaka, Hiroyuki Takahashi, Kazuyuki Demachi, Ritu Bhusal Chhatkuli, ["Frame forecasting in cine MRI using the PCA respiratory motion model: comparing recurrent neural networks trained online and transformers"](https://doi.org/10.1016/j.compmedimag.2026.102755), Computerized Medical Imaging and Graphics (2026) [[arXiv]](https://doi.org/10.48550/arXiv.2410.05882)

A detailed description of the Lucas–Kanade registration algorithm used as a basis for this repository can be found in:
 - Michel Pohl, Mitsuru Uesaka, Kazuyuki Demachi, and Ritu Bhusal Chhatkuli, ["Prediction of the motion of chest internal points using a recurrent neural network trained with real-time recurrent learning for latency compensation in lung cancer radiotherapy"](https://doi.org/10.1016/j.compmedimag.2021.101941), Computerized Medical Imaging and Graphics (2021) [[arXiv]](https://doi.org/10.48550/arXiv.2207.05951). 