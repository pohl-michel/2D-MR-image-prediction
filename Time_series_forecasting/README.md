This folder contains the implementation of several algorithms for predicting multidimensional time series:
 - an ordinary least squares (OLS) linear autoregressive model
 - least mean squares (LMS)
 - recurrent neural networks (RNN) trained with:
   - real-time recurrent learning (RTRL)
   - unbiased online recurrent optimization (UORO) 
   - decoupled neural interfaces (DNI)
   - sparse one-step approximation (SnAp-1)
 - transformer-encoder models: both a sequence-specific baseline model and a population model (trained on multiple sequences)

All models above are implemented in MATLAB, except for the transformers, whose training and inference are conducted in Python (via the [Pytorch library](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html)), with MATLAB interfacing for performance evaluation. Hyperparameter tuning for the population transformer uses the [Optuna library](https://optuna.org/). 

This folder is an extension of one of our [former repository for time series forecasting with RTRL and UORO](https://github.com/pohl-michel/time-series-forecasting-with-UORO-RTRL-LMS-and-linear-regression/tree/main). For instance, we added here other prediction methods, including a simpler implementation of RTRL, and evaluation metrics that are not restricted to 3D object position forecasting.

The figure below illustrates the prediction of the 3D positions of 3 markers placed on the chest and abdomen of an individual lying face upwards 2.1s in advance using DNI. The sampling rate is 3.33Hz and the z-axis corresponds to the spine direction.

<img src="visualization/markers_seq_1_3.33_Hz_DNI_k=12_q=180_eta=0.01_sg=0.02_grd_tshld=100_h=7.gif" width="65%" height="65%"/>

This code in this folder supports the claims in the following research articles:
1. Michel Pohl, Mitsuru Uesaka, Hiroyuki Takahashi, Kazuyuki Demachi, Ritu Bhusal Chhatkuli, ["Real-time respiratory motion forecasting with online learning of recurrent neural networks for accurate targeting in externally guided radiotherapy"](https://doi.org/10.1016/j.cmpb.2025.108828), *Computer Methods and Programs in Biomedicine* (2025) [[arXiv open-access version]](https://doi.org/10.48550/arXiv.2403.01607)
2. Michel Pohl, Mitsuru Uesaka, Hiroyuki Takahashi, Kazuyuki Demachi, Ritu Bhusal Chhatkuli, ["Future frame prediction in chest and liver cine MRI using the PCA respiratory motion model: comparing transformers and dynamically trained recurrent neural networks"](https://doi.org/10.48550/arXiv.2410.05882), arXiv preprint arXiv:2410.05882 (2026)

Please consider citing these articles if you reuse this code in your research. As an additional resource, the following paper, whose results can be reproduced with this code, provides further details regarding 3D marker position forecasting with UORO:

3. Michel Pohl, Mitsuru Uesaka, Hiroyuki Takahashi, Kazuyuki Demachi, Ritu Bhusal Chhatkuli, ["Prediction of the Position of External Markers Using a Recurrent Neural Network Trained With Unbiased Online Recurrent Optimization for Safe Lung Cancer Radiotherapy"](https://doi.org/10.1016/j.cmpb.2022.106908), *Computer Methods and Programs in Biomedicine* (2022) [[arXiv open-access version]](https://doi.org/10.48550/arXiv.2106.01100) [[blog article]](https://pohl-michel.github.io/blog/articles/predicting-respiratory-motion-online-learning-rnn/article.html)

The directory "a. Input time series sequences" contains the following data:
 1. 3D positions of external markers placed on the chest and abdomen of healthy individuals breathing during intervals ranging from 73s to 222s. The sampling frequency in the original dataset is 10Hz; we also included the data resampled at 3.33Hz and 30Hz (with additional noise after upsampling). The 10Hz data was originally introduced in the following work: Krilavicius, Tomas, et al. [“Predicting Respiratory Motion for Real-Time Tumour Tracking in Radiotherapy”](https://doi.org/10.48550/arXiv.1508.00749), arXiv preprint arXiv:1508.00749 (2015).
 2. Time-varying weights derived from PCA applied to the dense deformation field between the initial frame and frame at time $t$ in sagittal magnetic resonance (MR) image sequences (lasting from 63s to 83s). Those 2D image sequences are located in the folder "Image_prediction/input_imgs/2D images"; the original data before preprocessing and applying PCA was downloaded from [chest 4D MRI](https://vision.ee.ethz.ch/datsets.html) and [2D liver MR image](http://open-science.ub.ovgu.de/xmlui/handle/684882692/88) datasets from ETH Zürich and Otto-von-Guericke University Magdeburg, respectively.

The main executable files in the repository are:
 1. "signal_prediction_main.m": it runs prediction with a given algorithm and set of hyperparameters (including the horizon), that can be configured in the files "pred_par.xlsx" (inside "a. Input time series sequences/my_sequence") and "load_pred_par.m". The workflow can be set in "load_sigpred_behavior_parameters.m".
 2. "sigpred_hyperparameter_optimization_main.m": it runs grid search on the validation set to determine the optimal hyperparameters for each sequence, algorithm (except population transformers), and horizon, and provides information regarding hyperparameter influence on prediction accuracy. The range of hyperparameters can be selected manually in "load_hyperpar_cv_info.m".
 3. "train_population_transformer.sh": it runs hyperparameter tuning for population transformers using grid search, followed by training using the best hyperparameters, looping over different prediction horizons. The json configuration files for hyperparameter tuning are located in the folder "other_prediction_functions". In particular, the "horizons", "nb_runs", "param_grid", and "dev_folders" fields specify the horizon range, the number of models initialized and trained for each hyperparameter combination, the hyperparameter grid, and the sequences used for training and validation. The models trained with the best hyperparameters are saved in the "models" folder; we save exactly "nb_runs" models for each horizon, to take stochasticity into account. Those are loaded by the two scripts above for model evaluation (script 2 above only does evaluation but not hyperparameter tuning for population transformers, as it only loads the transformer with optimized hyperparameters).

Regarding exploration with transformers, we provided jupyter notebooks in the "experiments" folder to help vizualize and analyze the predictions, the training and validation loss, and specifically for the population transformer, input signal preprocessing steps (resampling and augmentation), and standardization effect. Before using transformers, one needs to install the Pytorch library and [set the Python interpreter path in Matlab](https://uk.mathworks.com/help/matlab/ref/pyenv.html), for instance, if using conda/miniconda:
 - `conda install pytorch pytorch-cuda -c pytorch -c nvidia` (if using an NVidia GPU, on the Anaconda prompt)
 - `pyenv("Version", "C:\Users\username\miniconda3\envs\my_environment\python.exe");` (on the Matlab command line)

The input time-series sequences used in scripts 1 and 2 above can be selected in "load_sigpred_path_parameters.m" by commenting or uncommenting the corresponding text strings. For each sequence, the "data_type" field in the associated "pred_par.xlsx" file determines whether it contains the position of 3D objects ("data_type" set to 1) or more general time series ("data_type" set to 2), which affects the evaluation metrics (cf. the articles above).

"sigpred_hyperparameter_optimization_main.m" uses parallel computations to make grid search faster; the MATLAB parallel processing toolbox is required to run that script, except if one replaces the `parfor` instructions by `for` instructions (which would yield higher processing time). The GPU can be used to increase training and inference speed for RNN and transformers by setting `beh_par.GPU_COMPUTING` to `true` (this also requires the parallel processing toolbox). For instance, we found that calculations were faster with the GPU when using RTRL with a relatively high number of hidden units.

This folder also includes the following two auxiliary scripts for data curation:
 1. "convert_csv_to_mat.m" converts the original .csv file (external marker positions sampled at 10Hz) from the [article of Krilavicius et al.](https://doi.org/10.48550/arXiv.1508.00749) in the "Original data" folder into the "data.mat" files in the "a. Input time series sequences" folder.
 2. "resample_time_series_data.py" resamples time-series data in the latter folder at the specified frequency.