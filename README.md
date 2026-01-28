This repository contains code for the following tasks:
1. [time series forecasting](https://github.com/pohl-michel/2D-MR-image-prediction/tree/main/Time_series_forecasting)
2. video forecasting

Specifically, we implemented several methods for that purpose:
 -  online learning algorithms for recurrent neural networks (RNN), for sequence-specific learning (MATLAB)
     - real-time recurrent learning (RTRL)
     - [unbiased online recurrent optimization (UORO)](https://arxiv.org/abs/1702.05043)
     - [sparse-1 step approximation (SnAp-1)](https://arxiv.org/abs/2006.07232)
     - [decoupled neural interfaces (DNI)](http://proceedings.mlr.press/v70/jaderberg17a.html)
 - encoder-only transformers (both population-based and sequence-specific models) trained with backpropagation (Pytorch)

The domain application is respiratory motion forecasting: prediction of the positions of external markers on the chest and abdomen, and 2D chest magnetic resonance (MR) image sequence prediction. However, the method is general and can be applied to the prediction of any time series and quasi-periodic, simple videos. We invite the readers interested specifically in time series forecasting to refer to the [README.md file located in the "Time_series_forecasting" folder](https://github.com/pohl-michel/2D-MR-image-prediction/blob/main/Time_series_forecasting/README.md); this README file focuses mostly on video forecasting for brevity.

<img src="Image_prediction/visualization/2._sq_sl010_sag_Xcs=125_SnAp-1_k=12_q=110_eta=0.02_sg=0.02_h=6_3_cpts_t=181_to_200_cropped.gif" width="40%" height="40%"/>
<img src="Image_prediction/visualization/3._sq_sl010_sag_Xcs=80_SnAp-1_k=30_q=70_eta=0.02_sg=0.02_h=6_4_cpts_t=181_to_200_cropped.gif" width="40%" height="40%"/>
<img src="Image_prediction/visualization/4. sq sl014 sag Xcs=165 SnAp-1 k=6 q=110 eta=0.01 sg=0.02 h=6 3 cpts_t=181_to_200_cropped.gif" width="40%" height="40%"/>
<img src="Image_prediction/visualization/5. sq sl014 sag Xcs=95 SnAp-1 k=6 q=110 eta=0.01 sg=0.02 h=6 2 cpts_t=181_to_200_cropped.gif" width="40%" height="40%"/>

Left: ground-truth / right: predicted image 6 time steps in advance using SnAp-1.

This repository supports the claims in the following research articles, which provide detailed technical information. Please consider citing them if you use this code in your research.
1. Michel Pohl, Mitsuru Uesaka, Hiroyuki Takahashi, Kazuyuki Demachi, Ritu Bhusal Chhatkuli, ["Real-time respiratory motion forecasting with online learning of recurrent neural networks for accurate targeting in externally guided radiotherapy"](https://doi.org/10.1016/j.cmpb.2025.108828), *Computer Methods and Programs in Biomedicine* (2025) [[arXiv open-access version]](https://doi.org/10.48550/arXiv.2403.01607)
2. Michel Pohl, Mitsuru Uesaka, Hiroyuki Takahashi, Kazuyuki Demachi, Ritu Bhusal Chhatkuli, ["Future frame prediction in chest and liver cine MRI using the PCA respiratory motion model: comparing transformers and dynamically trained recurrent neural networks"](https://doi.org/10.48550/arXiv.2410.05882), arXiv preprint arXiv:2410.05882 (2026)

The repository is essentially split into two different folders:
 1. The "Time_series_forecasting" folder contains scripts and functions that perform time series forecasting. It is essentially self-contained / independent, and is an extension of our previous work [[code]](https://github.com/pohl-michel/time-series-forecasting-with-UORO-RTRL-LMS-and-linear-regression) [[associated research article]](https://doi.org/10.48550/arXiv.2106.01100).
 2. The "Image_prediction" folder contains scripts and functions that perform video forecasting (e.g., motion estimation, encoding, and decoding, and image warping). Those call functions from the former folder to predict the low-dimensional representation of motion in the input video.

The main script performing image prediction is "image_prediction_main.m". The behavior of "image_prediction_main.m" is controlled by "load_impred_behavior_parameters.m" and has essentially two modes:
 1. "beh_par.OPTIMIZE_NB_PCA_CP = true": optimization of the dimension of the subspace of the PCA motion model and hyperparameters associated with the forecasting methods. Optimization is done with parallel processing using the MATLAB parallel computing toolbox. The script can also be run without parallel processing by replacing the "parfor" loops by "for" loops.
 2. "beh_par.OPTIMIZE_NB_PCA_CP = false": inference with the selected prediction parameters.

The input images loaded by "image_prediction_main.m" are located in the "Image_prediction/input_imgs/2D images" directory. The subdirectory corresponding to each image sequence contain configuration files, among which:
 - pred_par.xlsx : parameters related to low-dimensional motion representation forecasting
 - OF_calc_par.xlsx : optical-flow algorithm parameters
 - disp_par.xlsx : parameters related to figure display or saving
 - im_seq_par.xlsx : parameters related to the input image sequence and the region of interest (ROI) for evaluation

Image prediction parameters can be configured in the following .m files:
 - Image_prediction/load_warp_par.m : file containing the parameters related to image warping.
 - Image_prediction/image_prediction_main.m : the variables "pred_meths" and "br_model_par.nb_pca_cp_tab" contain the prediction methods used and the number of principal components for inference (or the maximum number of principal components when doing hyperparameter optimisation), respectively. The names of the input sequences are specified in the "input_im_dir_suffix_tab" variable.
 - Time_series_forecasting/load_pred_par.m : file containing parameters related to each time-series forecasting algorithm, overriding those in the "pred_par.xlsx" file corresponding to the input sequence.
 - Time_series_forecasting/load_hyperpar_cv_info.m : file containing the range of hyperparameters corresponding to each forecasting method for grid search on the validation set.

The implementation of the pyramidal and iterative Lucas-Kanade optical-flow algorithm for 2D image sequences in this repository is an adaptation of that for 3D image sequences here: https://github.com/pohl-michel/Lucas-Kanade-pyramidal-optical-flow-for-3D-image-sequences. The script "Image_prediction/OF2D_param_optim_main.m" performs grid search to optimize optical-flow parameters. The grid is specified in the file "Image_prediction/load_OFeval_parameters.m". The optical-flow field for an arbitrary set of parameters can also be computed in "image_prediction_main.m". 

A detailed description of the registration algorithm can be found in the following article: Michel Pohl, Mitsuru Uesaka, Kazuyuki Demachi, Ritu Bhusal Chhatkuli, ["Prediction of the motion of chest internal points using a recurrent neural network trained with real-time recurrent learning for latency compensation in lung cancer radiotherapy"](https://doi.org/10.1016/j.compmedimag.2021.101941), *Computerized Medical Imaging and Graphics* (2021) [[arXiv open-access version]](https://doi.org/10.48550/arXiv.2207.05951). 

The cine-MR image sequences in the "Image_prediction/input_imgs" folder resulted from processing the original data in the following public datasets (details in [this paper](https://doi.org/10.48550/arXiv.2410.05882)):
 - chest image dataset: [ETH Zürich - 4D MRI lung data](https://vision.ee.ethz.ch/datsets.html).
 - liver image dataset: [2D navigator frames from Otto-von-Guericke University Magdeburg](http://open-science.ub.ovgu.de/xmlui/handle/684882692/88)