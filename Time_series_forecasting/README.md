This folder contains the implementation of several algorithms for predicting multidimensional time series:
 - an ordinary least squares (OLS) linear autoregressive model
 - least mean squares (LMS)
 - recurrent neural networks (RNN) trained with:
   - real-time recurrent learning (RTRL)
   - unbiased online recurrent optimization (UORO) 
   - decoupled neural interfaces (DNI)
   - sparse 1-step approximation (SnAp-1)
 - transformer-encoder models: both a sequence-specific baseline model and a population model (trained on multiple sequences)

It is both a module whose functions are called by other functions in the "Image_prediction" folder for video prediction, and a self-contained folder with scripts to perform time series forecasting. This folder is an extension of one of our former Github repository: https://github.com/pohl-michel/time-series-forecasting-with-UORO-RTRL-LMS-and-linear-regression/tree/main. For instance, we added here other prediction methods, including a simpler implementation of RTRL, and evaluation metrics that are not restricted to 3D object position forecasting.

The figure below illustrates prediction of the 3D positions of 3 markers placed on the chest and abdomen of an individual lying face upwards 2.1s in advance using DNI. The sampling rate is 3.33Hz and the z-axis corresponding to the spine direction.

<!-- <img src="visualization/markers_seq_1_3.33_Hz_DNI_k=12_q=180_eta=0.01_sg=0.02_grd_tshld=100_h=7.gif" width="65%" height="65%"/> -->

This code in this folder supports the claims in the following research articles:
1. Michel Pohl, Mitsuru Uesaka, Hiroyuki Takahashi, Kazuyuki Demachi, Ritu Bhusal Chhatkuli, ["Real-time respiratory motion forecasting with online learning of recurrent neural networks for accurate targeting in externally guided radiotherapy"](https://doi.org/10.1016/j.cmpb.2025.108828), *Computer Methods and Programs in Biomedicine* (2025) [[arXiv open-access version]](https://doi.org/10.48550/arXiv.2403.01607)
2. Michel Pohl, Mitsuru Uesaka, Hiroyuki Takahashi, Kazuyuki Demachi, Ritu Bhusal Chhatkuli, ["Future frame prediction in chest and liver cine MRI using the PCA respiratory motion model: comparing transformers and dynamically trained recurrent neural networks"](https://doi.org/10.48550/arXiv.2410.05882), arXiv preprint arXiv:2410.05882 (2026)

Please consider citing these articles if you reuse this code in your research. As an additional resource, the following paper, whose results can be reproduced with this code, also provides further details regarding 3D marker position forecasting with UORO:

3. Michel Pohl, Mitsuru Uesaka, Hiroyuki Takahashi, Kazuyuki Demachi, Ritu Bhusal Chhatkuli, ["Prediction of the Position of External Markers Using a Recurrent Neural Network Trained With Unbiased Online Recurrent Optimization for Safe Lung Cancer Radiotherapy"](https://doi.org/10.1016/j.cmpb.2022.106908), *Computer Methods and Programs in Biomedicine* (2022) [[arXiv open-access version]](https://doi.org/10.48550/arXiv.2106.01100) [[blog article]](https://pohl-michel.github.io/blog/articles/predicting-respiratory-motion-online-learning-rnn/article.html)

The data provided in the directories "a. Input time series sequences" consists of the following:
 1. 3D positions of external markers placed on the chest and abdomen of healthy individuals breathing during intervals ranging from 73s to 222s. The sampling frequency in the original dataset is 10Hz; we included a downsampled version at 3.33Hz and an upsampled version (with additional noise) at 30Hz. The 10Hz data was originally introduced in the following work: Krilavicius, Tomas, et al. [“Predicting Respiratory Motion for Real-Time Tumour Tracking in Radiotherapy”](https://doi.org/10.48550/arXiv.1508.00749), arXiv preprint arXiv:1508.00749 (2015).
 2. Time-varying weights derived from principal component analysis (PCA) applied to the dense deformation field between the initial frame and frame at time $t$ in sagittal magnetic resonance (MR) image sequences. Those 2D image sequences are located in the folder "/Image_prediction/input_imgs/2D images"; the original data before preprocessing was downloaded from a [chest 4D MRI dataset (ETH Zürich)](https://vision.ee.ethz.ch/datsets.html) and [2D liver MR image dataset from Otto-von-Guericke University Magdeburg](http://open-science.ub.ovgu.de/xmlui/handle/684882692/88).

The main executable files in the repository are:
 1. "signal_prediction_main.m": it runs prediction with a given algorithm and set of hyperparameters (including the horizon), that can be selected manually in the files "pred_par.xlsx" (inside "a. Input time series sequences/my_sequence") and "load_pred_par.m". The workflow configuration can be set in "load_sigpred_behavior_parameters.m".
 2. "sigpred_hyperparameter_optimization_main.m": it runs grid search on the validation set to determine the optimal hyperparameters for each sequence, algorithm, and horizon and provides information on hyperparameter influence on prediction accuracy. The range of hyperparameters can be selected manually in "load_hyperpar_cv_info.m".

The input time-series sequences used in both scripts can be selected in "load_sigpred_path_parameters.m" by commenting or uncommenting the corresponding text strings. For each sequence, the "data_type" field in the associated "pred_par.xlsx" file determines whether it contains the position of 3D objects ("data_type" set to 1) or more general time series ("data_type" set to 2), which affects the evaluation metrics (cf. the articles above). Display parameters can be chosen in "disp_par.xlsx". 

"sigpred_hyperparameter_optimization_main.m" uses parallel computations to make grid search faster; the MATLAB parallel processing toolbox is required to run that script, except if one replaces the `parfor` instructions by `for` instructions (which would yield higher processing time). The GPU can be used to increase RNN training and inference speed by setting `beh_par.GPU_COMPUTING` to `true` (this also requires the parallel processing toolbox). We found that calculations were faster with the GPU when using RTRL with a relatively high number of hidden units.

This folder also includes the following two auxiliary scripts for data curation:
 1. "convert_csv_to_mat.m" converts the original .csv file (external marker positions sampled at 10Hz) from the [article of Krilavicius et al.](https://doi.org/10.48550/arXiv.1508.00749) in the "Original data" folder into the "data.mat" files in the "a. Input time series sequences" folder.
 2. "resample_time_series_data.py" resamples time series data in the latter folder at the specified frequency; Gaussian noise is added after upsampling. We included the data resampled at 3.33Hz and 30Hz in the folder "a. Input time series sequences", as it was used in the [article on time-series forecasting mentioned above](https://doi.org/10.1016/j.cmpb.2022.106908).