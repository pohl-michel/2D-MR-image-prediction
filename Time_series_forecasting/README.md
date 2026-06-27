# Multivariate time-series forecasting using RNNs trained online and transformers

## Overview

This folder contains code for multivariate time-series forecasting using the following algorithms:
 - online learning algorithms for vanilla/standard RNNs (sequence-specific models):
   - real-time recurrent learning (RTRL)
   - [unbiased online recurrent optimization (UORO)](https://arxiv.org/abs/1702.05043)
   - [sparse one-step approximation (SnAp-1)](https://arxiv.org/abs/2006.07232)
   - [decoupled neural interfaces (DNI)](http://proceedings.mlr.press/v70/jaderberg17a.html)
 - encoder-only transformers (sequence-specific model and population-based model trained on multiple sequences)
 - sequence-specific autoregressive baselines: 
    - coordinate-wise support vector regression (SVR) with an RBF kernel
    - multivariate linear models: ordinary least squares (OLS) linear regression and least mean squares (LMS)

All models above are implemented in MATLAB, except for the transformers, whose training and inference are conducted in Python (via the [Pytorch library](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html)), with MATLAB interfacing for performance evaluation. Hyperparameter tuning for the population transformer uses the [Optuna library](https://optuna.org/). 

<p align="center">
  <img src="visualization/markers_seq_1_3.33_Hz_DNI_k=12_q=180_eta=0.01_sg=0.02_grd_tshld=100_h=7.gif" width="80%" alt="External-marker position forecast example">
</p>

<p align="center">
  <em>
    Prediction of the 3D positions of 3 markers on the chest and abdomen of an individual lying face up using DNI at a horizon of 2.1s. The sampling rate is 3.33Hz and the z-axis corresponds to the spine direction.
  </em>
</p>


## How to run the time-series forecasting code

### Inference with specified parameters

Time-series forecasting with a specified algorithm, sequence, and set of parameters, can be run by executing `signal_prediction_main.m` in MATLAB from the repository root or from `Time_series_forecasting`. 

### Hyperparameter optimization

There are two scripts for hyperparameter optimization:
 1. To run grid search on the validation set for sequence-specific algorithms for different horizon values, run `sigpred_hyperparameter_optimization_main.m` in MATLAB from the root folder or from `Time_series_forecasting`. 
 2. To run grid search for the population transformer model followed by retraining with the best hyperparameters for different horizon values, run the command-line app `train_population_transformer.sh` from the Bash terminal in the "Time_series_forecasting" folder. An example configuration is provided in `.vscode/launch.json`.    

**Notes**:
 - Grid search-based validation can be computationally expensive, especially when evaluating several prediction methods, horizons, and random parameter initializations.
 - One can evaluate the performance of population transformers for different horizons and sequences with the best parameters by running `sigpred_hyperparameter_optimization_main.m`.
 
### Expected outputs

For MATLAB scripts, prediction figures are saved in `b. Prediction results (figures)` and `c. Prediction results (images)`, while numerical performance logs are written to `e. Log txt files`. Those folders are created automatically if they do not exist.

When running `train_population_transformer.sh`, grid-search results for population transformers and transformer models with the best hyperparameters are saved in the `models` folder. Several `.pt` files are saved for each horizon because models for each configuration are trained with multiple random initializations (to take stochasticity into account during evaluation). These trained models are then loaded when running the evaluation scripts `signal_prediction_main.m` and `sigpred_hyperparameter_optimization_main.m`.

### Transformer experiments

For transformers, Jupyter notebooks are provided in the `experiments` folder to visualize predictions and training/validation losses. They also help examine input-signal preprocessing steps (i.e., data resampling and augmentation) and data standardization influence on predictions, for the population transformer.


## Time-series data

The time-series data are located in `a. Input time series sequences` and include:

1. 3D positions of 3 external markers on the chest and abdomen of healthy individuals breathing during intervals ranging from 73s to 222s.
2. Time-varying PCA coefficients derived from dense deformation fields between the initial frame and subsequent frames in sagittal MR image sequences lasting 63–83 s.

The 10 Hz marker-position data was originally introduced in the following work: Krilavicius et al., ["Predicting Respiratory Motion for Real-Time Tumour Tracking in Radiotherapy"](https://doi.org/10.48550/arXiv.1508.00749), arXiv preprint (2015). The position data resampled at 3.33Hz and 30Hz (with Gaussian noise added after upsampling) were also included to evaluate the influence of frequency on forecasting performance.

The MRI sequences used to derive the time-dependent PCA coefficients are located in `Image_prediction/input_imgs/2D images` (cf. ["Image data" section](../README.md#image-data) in the main README file of this repository).

The input time-series sequences used by the MATLAB scripts can be selected in `load_sigpred_path_parameters.m` by commenting or uncommenting the corresponding text strings. Training and validation sequences for population transformer models are specified in the JSON configuration files in the folder `other_prediction_functions`.

For each time-series sequence, the `data_type` field in the associated `pred_par.xlsx` file determines whether it contains the position of 3D objects (`data_type` set to 1) or more general time series (`data_type` set to 2), which affects evaluation metrics (cf. ["References" section](../README.md#references) of the main README file of this repository).

**Data curation scripts**

This folder also includes auxiliary scripts for data curation: `convert_csv_to_mat.m`, which converts the original 10 Hz marker-position .csv files into .mat files, and `resample_time_series_data.py`, which resamples the marker-position time series at specified frequencies.

## Configuration parameters

The files with the main parameters that can be configured are listed in the table below: 

Folder | File | Parameters |
---    |--- |--- |
Input time-series subfolder of `a. Input time series sequences` | `pred_par.xlsx` | Sequence-specific forecasting parameters, including horizon, signal-history length, and algorithm-specific settings. |
 `Time_series_forecasting` | `disp_par.xlsx` | Display and plotting parameters. |
 — | `load_pred_par.m` | Additional method-specific forecasting parameters and defaults. |
 — | `load_hyperpar_cv_info.m` | Hyperparameter grids for validation-set search for sequence-specific models. |
 — | `load_sigpred_behavior_parameters.m` | Workflow flags for training, evaluation, and saving. |
 — | `load_sigpred_path_parameters.m` | Input-sequence selection and path configuration. |
 `other_prediction_functions` | `pop_transformer_training_*.json` | Population-transformer settings. |

For population transformers, the `horizons`, `nb_runs`, `param_grid`, and `dev_folders` fields in the `pop_transformer_training_*.json` files specify the horizon range, the number of models initialized and trained for each hyperparameter combination, the hyperparameter grid, and the sequences used for training and validation, respectively.


## Requirements / References

See the ["Requirements"](../README.md#requirements) and ["References"](../README.md#references) sections of the main README file of this repository.



