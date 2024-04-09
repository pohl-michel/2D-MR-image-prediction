This repository contains code performing the following:
1. time series forecasting
2. video forecasting

Specifically, various methods for online learning of recurrent neural networks (RNN) are implemented in order to execute the two tasks above:
 - real-time recurrent learning (RTRL)
 - [unbiased online recurrent optimization (UORO)](https://arxiv.org/abs/1702.05043)
 - [sparse-1 step approximation (SnAp-1)](https://arxiv.org/abs/2006.07232)
 - [decoupled neural interfaces (DNI)](http://proceedings.mlr.press/v70/jaderberg17a.html)

Video forecasting is executed by performing the following steps in order:
1. computation of the "push" deformation vector field $\vec{u}(\vec{x}, t_n)$ between the image at time $t_1$ and $t_n$ for all indices $n$.
2. computation of the projection of $\vec{u}(\vec{x}, t_n)$ onto principal deformation vector fields (the principal components) $\vec{u}_i(\vec{x})$. The coordinates of that projection are the time-dependent weights $w_i(t)$.
3. prediction of the weights $w_i(t)$ using various methods including the RNN learning algorithms mentioned above.
4. warping forward the image at time $t_1$ using the deformation field reconstructed using the predicted weights $w_i(t+h)$ where $h$ is the prediction horizon.

The examples given correspond to respiratory motion forecasting: external marker position prediction for the first application, and 2D magnetic resonance chest image sequence prediction for the second application. However, the code is general and could be applied to the prediction of time series and (simple) videos in general.   

This repository supports the claims in the following research articles, that provide detailed technical information:
1. Michel Pohl, Mitsuru Uesaka, Hiroyuki Takahashi, Kazuyuki Demachi, Ritu Bhusal Chhatkuli, ["Respiratory motion forecasting with online learning of recurrent neural networks for safety enhancement in externally guided radiotherapy"](https://arxiv.org/abs/2403.01607), arXiv preprint arXiv:2403.01607, 2024  
2. other article to submit

Please consider citing these articles if you use this code in your research.

The repository is split into two different folders:
 - "Time series_forecasting" contains scripts and functions that perform time series forecasting. It is essentially self-contained / independent, and is an extension of the repository https://github.com/pohl-michel/time-series-forecasting-with-UORO-RTRL-LMS-and-linear-regression associated with [our former research article](https://arxiv.org/abs/2106.01100). Readers specifically interested in time-series forecasting should refer to the readme file within that folder.
 - "Image_prediction" contains scripts and function that perform video forecasting. Those call functions from the former folder, that are used to predict $w_i(t)$, which is the (low-dimensional) compressed representation of motion within these videos.

The "Image_prediction" folder contains the following scripts:
 - image_prediction_main.m : main script performing image prediction 
 - OF2D_param_optim_main.m : script optimising parameters associated with deformation vector field computation using the pyramidal Lucas-Kanade optical flow method
 - dcm2Dseq_from_real_mha_dcm3Dseq.m : script that converts 4D chest MR scans into 2D videos that are used as input for the forecasting script.
 - create_4Dmri_dataset.py : temporary script for creating 2D image sequences from 4D MR images from the Magdeburg university dataset (experimental)

The behavior of "image_prediction_main.m" is controlled by "load_impred_behavior_parameters.m" and essentially has two modes:
 1. optimization of the number of principal components and hyperparameters associated with the forecasting methods (corresponding to "beh_par.OPTIMIZE_NB_PCA_CP = true" in "load_impred_behavior_parameters.m"). Parallel computations are performed (which requires the parallel computing toolbox). In case one does not have access to the parallel computing toolbox, one can still run the script by replacing the "parfor" loops by "for" loops.
 2. inference with given prediction parameters (corresponding to "beh_par.OPTIMIZE_NB_PCA_CP = false" in "load_impred_behavior_parameters.m")

The parameters corresponding to image prediction can be modified in the following parts of the code:
 - load_pred_par.m : file containing time-series forecasting parameters. It loads the "pred_par.xlsx" file in each sequence directory and paramters in those files can also be modified.
 - load_warp_par.m : file containing the parameters related to image warping
 - image_prediction_main.m : the variables "pred_meths" and "br_model_par.nb_pca_cp_tab" contain the prediction methods used and the number of principal components for inference (or the maximum number of principal comopnents when doing hyper-parameter optimisation), respectively. The input sequence names are specified in the "input_im_dir_suffix_tab" variable.
 - load_hyperpar_cv_info.m : file containing the range of hyper-parameters used for cross-validation.

The input images loaded by "image_prediction_main.m" are located in the "input_imgs/2D images" directory. The directory corresponding to each image sequence contain several files, among which:
 - pred_par.xlsx : contains parameter values related to low-dimensional representation forecasting
 - OF_calc_par.xlsx : contains optical flow computation parameters
 - disp_par.xlsx : contains parameters related to the display of image or figure results 
 - im_seq_par.xlsx : contains parameters related to the input image seuqnece itself (image dimensions, number of images...)

The output .mat variables, text files, images, and figures, are located in the "tmp_vars", "tmp_txt_files", "tmp_imgs", and 'tmp_figs" folders, respectively. 

