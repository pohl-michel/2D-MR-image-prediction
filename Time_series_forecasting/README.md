This folder contains code to predict multidimensional time-series data using least mean squares (LMS), multivariate linear regression, or recurrent neural networks (RNN) trained with real-time recurrent learning (RTRL), unbiased online recurrent optimization (UORO), decoupled neural interfaces (DNI), and sparse 1-step approximation (SnAp-1).

It is both a module containing code used for video prediction by calling functions in the "Image_prediction" folder, and a self-contained folder with scripts to perform time series forecasting. This folder is an extension of our former Github repository https://github.com/pohl-michel/time-series-forecasting-with-UORO-RTRL-LMS-and-linear-regression/tree/main in that it additionally contains the implementation of DNI, Snap-1, and another simpler implementation of RTRL, and we added performance evaluation metrics that are not necessarily restricted to the forecast of 3D objects (the former repository used a version of the MAE, RMSE, etc., for 3D objects only).  

The figure below illustrates prediction of the 3D position of 3 markers placed on the chest 2.1s (7 time steps) in advance using DNI (the sampling rate is 3.33Hz). 

<img src="visualization/markers_seq_1_3.33_Hz_DNI_k=12_q=180_eta=0.01_sg=0.02_grd_tshld=100_h=7.gif" width="65%" height="65%"/>

This code supports the claims in the following research articles:
 - Michel Pohl, Mitsuru Uesaka, Hiroyuki Takahashi, Kazuyuki Demachi, Ritu Bhusal Chhatkuli, "Prediction of the Position of External Markers Using a Recurrent Neural Network Trained With Unbiased Online Recurrent Optimization for Safe Lung Cancer Radiotherapy", Computer Methods and Programs in Biomedicine (2022): 106908. [[published version]](https://doi.org/10.1016/j.cmpb.2022.106908) [[accepted manuscript (preprint)]](https://doi.org/10.48550/arXiv.2106.01100)
 - Michel Pohl, Mitsuru Uesaka, Hiroyuki Takahashi, Kazuyuki Demachi, Ritu Bhusal Chhatkuli, "Respiratory motion forecasting with online learning of recurrent neural networks for safety enhancement in externally guided radiotherapy." arXiv preprint arXiv:2403.01607 (2024). [[preprint]](https://doi.org/10.48550/arXiv.2403.01607)
 - Michel Pohl, Mitsuru Uesaka, Hiroyuki Takahashi, Kazuyuki Demachi, Ritu Bhusal Chhatkuli, "Future frame prediction in chest cine MR imaging using the PCA respiratory motion model and dynamically trained recurrent neural networks", arXiv preprint arXiv:2410.05882, 2024 [[preprint]](https://doi.org/10.48550/arXiv.2410.05882)

 Please consider citing our articles if you use this code in your research.

Also, please do not hesitate to read my blog article summarizing this research and share it or leave a comment if you are interested:
 - https://medium.com/towards-data-science/forecasting-respiratory-motion-using-online-learning-of-rnns-for-safe-radiotherapy-bdf4947ad22f (Medium)
 - https://pohl-michel.github.io/blog/articles/predicting-respiratory-motion-online-learning-rnn/article.html (personal blog)

The data provided in the directories "1. Input time series sequences" consists of the following:
 1. three-dimensional position of external markers placed on the chest and abdomen of healthy individuals breathing during intervals from 73s to 222s. The markers move because of the respiratory motion. The sampling frequency in the original dataset is 10Hz, we included a downsampled version at 3.33Hz and an upsampled version (with additional noise) at 30Hz. The 10Hz data was used originally in the following article: Krilavicius, Tomas, et al. “Predicting Respiratory Motion for Real-Time Tumour Tracking in Radiotherapy.” ArXiv:1508.00749 [Physics], Aug. 2015. arXiv.org, https://doi.org/10.48550/arXiv.1508.00749.
 2. time-varying weights derived from principal component analysis (PCA) applied to the deformation between the first frame and frame at time $t$ in sagittal magnetic resonance (MR) image sequences. The 2D sequences that we used are located in the folder "/Image_prediction/input_imgs/2D images", and the original corresponding 3D image sequences is the "4D MRI lung data" dataset publicly available on the [website of ETH Zürich](https://bmic.ee.ethz.ch/research/datasets.html).

The two main scripts in the repository are:
 1. "signal_prediction_main.m": it performs prediction with a given algorithm and set of hyper-parameters, which can be selected manually in the files "pred_par.xlsx" and "load_pred_par.m". The workflow configuration can be set in "load_sigpred_behavior_parameters.m".
 2. "sigpred_hyperparameter_optimization_main.m": it performs grid search on the cross-validation set to determine the optimal hyper-parameters for each sequence and provides information about the influence of each hyper-parameter on the prediction accuracy. The range of hyper-parameters can be selected manually in "load_hyperpar_cv_info.m".

The input time-series sequences used in both scripts can be selected in "load_path_parameters.m" by commenting or uncommenting the corresponding text strings. For each sequence, the "data_type" field in the associated "pred_par.xlsx" file determines whether it is considered as the position of 3D objects ("data_type" set to 1) or more general time series ("data_type" set to 2), which impacts the metrics used for performance evaluation. The latter are defined in the articles cited above. The display parameters can be chosen in "disp_par.xlsx". 

"sigpred_hyperparameter_optimization_main.m" uses parallel computations to make grid search faster. Therefore, the parallel processing toolbox of Matlab is normally required for that script. However, tt can also be used without that toolbox by replacing all the `parfor` instructions by `for` instructions, at the expense of a higher processing time. One can also use GPU computing to try to make the RNN calculations faster by setting the variable `beh_par.GPU_COMPUTING` to `true` (which requires the parallel processing toolbox). Calculations are faster with the GPU when using RTRL with a relatively high number of hidden units.

There also two auxiliary scripts for data curation in this folder:
 1. "convert_csv_to_mat.m" converts the original .csv data from the article of Krilavicius et al. in the "Original data" folder into the "data.mat" files corresponding to the time series sampled at 10Hz in the "Input time series sequences" folder.
 2. "resample_time_series_data.py" resamples the latter 10Hz time series at the specified frequency (we included the data resampled at 3.33Hz and 30Hz as it was used for one of the papers above). Noise is added before upsampling.


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/prediction-of-the-position-of-external/multivariate-time-series-forecasting-on-3)](https://paperswithcode.com/sota/multivariate-time-series-forecasting-on-3?p=prediction-of-the-position-of-external)
