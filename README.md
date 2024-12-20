This repository contains code performing the following:
1. [time series forecasting](https://github.com/pohl-michel/2D-MR-image-prediction/tree/main/Time_series_forecasting)
2. video forecasting

Specifically, various methods for online learning of recurrent neural networks (RNN) are used to execute the two tasks above:
 - real-time recurrent learning (RTRL)
 - [unbiased online recurrent optimization (UORO)](https://arxiv.org/abs/1702.05043)
 - [sparse-1 step approximation (SnAp-1)](https://arxiv.org/abs/2006.07232)
 - [decoupled neural interfaces (DNI)](http://proceedings.mlr.press/v70/jaderberg17a.html)

The case scenarios proposed correspond to respiratory motion forecasting: external marker position prediction for the first application, and 2D magnetic resonance chest image sequence prediction for the second application. However, the code is general and could be applied to the prediction of time series and (simple) videos in general. We invite the readers interested specifically in time series forecasting to refer to the [README.md file located in the "Time_series_forecasting" folder](https://github.com/pohl-michel/2D-MR-image-prediction/blob/main/Time_series_forecasting/README.md) as this README file focuses mostly on video forecasting for brevity.

<img src="Image_prediction/visualization/2._sq_sl010_sag_Xcs=125_SnAp-1_k=12_q=110_eta=0.02_sg=0.02_h=6_3_cpts_t=181_to_200_cropped.gif" width="40%" height="40%"/>
<img src="Image_prediction/visualization/3._sq_sl010_sag_Xcs=80_SnAp-1_k=30_q=70_eta=0.02_sg=0.02_h=6_4_cpts_t=181_to_200_cropped.gif" width="40%" height="40%"/>
<img src="Image_prediction/visualization/4. sq sl014 sag Xcs=165 SnAp-1 k=6 q=110 eta=0.01 sg=0.02 h=6 3 cpts_t=181_to_200_cropped.gif" width="40%" height="40%"/>
<img src="Image_prediction/visualization/5. sq sl014 sag Xcs=95 SnAp-1 k=6 q=110 eta=0.01 sg=0.02 h=6 2 cpts_t=181_to_200_cropped.gif" width="40%" height="40%"/>

Left: ground-truth / right: predicted image 6 time steps in advance using SnAp-1.

This repository supports the claims in the following research articles, that provide detailed technical information. Please consider citing them if you use this code in your research.
1. Michel Pohl, Mitsuru Uesaka, Hiroyuki Takahashi, Kazuyuki Demachi, Ritu Bhusal Chhatkuli, ["Respiratory motion forecasting with online learning of recurrent neural networks for safety enhancement in externally guided radiotherapy"](https://doi.org/10.48550/arXiv.2403.01607), arXiv preprint arXiv:2403.01607, 2024  
2. Michel Pohl, Mitsuru Uesaka, Hiroyuki Takahashi, Kazuyuki Demachi, Ritu Bhusal Chhatkuli, ["Future frame prediction in chest cine MR imaging using the PCA respiratory motion model and dynamically trained recurrent neural networks"](https://doi.org/10.48550/arXiv.2410.05882), arXiv preprint arXiv:2410.05882, 2024

The repository is split into two different folders:
 - "Time series_forecasting" contains scripts and functions that perform time series forecasting. It is essentially self-contained / independent, and is an extension of our previous work [[code]](https://github.com/pohl-michel/time-series-forecasting-with-UORO-RTRL-LMS-and-linear-regression) [[associated research article]](https://doi.org/10.48550/arXiv.2106.01100). cf the README file within that folder for more details.
 - "Image_prediction" contains scripts and functions that perform video forecasting (e.g., motion computation, encoding, decoding, and image warping). Those call functions from the former folder, that are used to predict the (low-dimensional) compressed representation of motion within these videos.

The main script performing image prediction is "image_prediction_main.m". The behavior of "image_prediction_main.m" is controlled by "load_impred_behavior_parameters.m" and essentially has two modes:
 1. optimization of the number of principal components and hyperparameters associated with the forecasting methods (corresponding to "beh_par.OPTIMIZE_NB_PCA_CP = true" in "load_impred_behavior_parameters.m"). Parallel computations are performed (which requires the parallel computing toolbox). In case one does not have access to the parallel computing toolbox, one can still run the script by replacing the "parfor" loops by "for" loops.
 2. inference with given prediction parameters (corresponding to "beh_par.OPTIMIZE_NB_PCA_CP = false" in "load_impred_behavior_parameters.m")

The parameters corresponding to image prediction can be modified in the following .m files:
 - Image_prediction/load_warp_par.m : file containing the parameters related to image warping.
 - Image_prediction/image_prediction_main.m : the variables "pred_meths" and "br_model_par.nb_pca_cp_tab" contain the prediction methods used and the number of principal components for inference (or the maximum number of principal comopnents when doing hyper-parameter optimisation), respectively. The input sequence names are specified in the "input_im_dir_suffix_tab" variable.
 - Time_series_forecasting/load_pred_par.m : file containing time-series forecasting parameters. It loads the "pred_par.xlsx" file in each sequence directory and parameters in those files can also be modified.
 - Time_series_forecasting/load_hyperpar_cv_info.m : file containing the range of hyper-parameters used for cross-validation.

The input images loaded by "image_prediction_main.m" are located in the "input_imgs/2D images" directory. The directory corresponding to each image sequence contain several files, among which:
 - pred_par.xlsx : contains parameter values related to low-dimensional representation forecasting
 - OF_calc_par.xlsx : contains optical flow computation parameters
 - disp_par.xlsx : contains parameters related to the display of images or figures 
 - im_seq_par.xlsx : contains parameters related to the input image sequence itself (image dimensions, number of images...)

The implementation of the pyramidal and iterative Lucas-Kanade optical flow algorithm for 2D images in this repository is an adaptation of that for 3D images in [one of my former repository](https://github.com/pohl-michel/Lucas-Kanade-pyramidal-optical-flow-for-3D-image-sequences). The script "OF2D_param_optim_main.m" optimizes the parameters associated with deformation vector field computation using that algorithm. The parameter grid is specified in the file "load_OF_hyperparameters.m". One can also compute the optical flow for a given set of parameters using "image_prediction_main.m". 

The complete description of the registration algorithm can be found in the following article: 
Michel Pohl, Mitsuru Uesaka, Kazuyuki Demachi, Ritu Bhusal Chhatkuli, "Prediction of the motion of chest internal points using a recurrent neural network trained with real-time recurrent learning for latency compensation in lung cancer radiotherapy", Computerized Medical Imaging and Graphics, Volume 91, 2021, 101941, ISSN 0895-6111 [[journal version with restricted access]](https://doi.org/10.1016/j.compmedimag.2021.101941) [[accepted manuscript version, openly available]](https://doi.org/10.48550/arXiv.2207.05951). Please consider citing that work if you use this code to perform image registration in your research.

The cine-MR sequences in the "Image_prediction/input_imgs" folder come originally from a public dataset from ETH Zürich publicly accessible online: [4D MRI lung data](https://bmic.ee.ethz.ch/research/datasets.html).
We selected 2 sagittal cross-sections for each of the 4D sequences, resampled them so that the resolution becomes 1mm*1mm per pixel, and shifted them so that the first image corresponds to the middle of expiration. Please cite the following article if you use that data in your work: "Boye, D. et al. - Population based modeling of respiratory lung motion and prediction from partial information - Proc. SPIE 8669, Medical Imaging 2013: Image Processing, 86690U (March 13, 2013); doi:10.1117/12.2007076"

PS (25th October 2024): When attempting to run image_prediction_main.py please first run the program with `beh_par.COMPUTE_OPTICAL_FLOW = true` (and the other user adjustable parameters set to `false`) inside "load_impred_behavior_parameters.m", and run it in a second time with the provided `beh_par` parameters. There is a small bug a the moment preventing to run the whole pipeline at once, which will be corrected later.
