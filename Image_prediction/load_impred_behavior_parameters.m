function [ beh_par ] = load_impred_behavior_parameters()
% The structure beh_par contains important information about the behavior of the whole algorithm,
% and its fields should be set manually.
% 
% To do: continue documentation for each field
%
% Author : Pohl Michel
% Date : Sept 18th, 2022
% Version : v1.0
% License : 3-clause BSD License


%% IMPORTANT PARAMETERS

beh_par.SAVE_ROI_DISPLAY = false;
beh_par.SAVE_MEAN_IMAGE = false;
beh_par.SAVE_ORG_IM_SQ = false;

beh_par.COMPUTE_OPTICAL_FLOW = false;
beh_par.SAVE_OF_JPG = false;
beh_par.EVAL_INIT_OF_WARP = false;
beh_par.SAVE_INIT_OF_WARP_JPG = false;

beh_par.PCA_OF_DVF = true;
beh_par.SAVE_PCA_CP_WEIGHTS_JPG = false;
beh_par.EVAL_PCA_RECONSTRUCT = false;
beh_par.SAVE_PCA_RECONSTR_JPG = false;

beh_par.TRAIN_EVAL_PREDICTOR = true;
beh_par.SAVE_PRED_RESULTS = true;

beh_par.IM_PREDICTION = true;
beh_par.SAVE_PRED_IM = true;
beh_par.NO_PRED_AT_ALL = false; 

beh_par.OPTIMIZE_NB_PCA_CP = false;

beh_par.GPU_COMPUTING = false; 
    % use of the GPU when performing prediction

beh_par.CROP_FOR_DISP_SAVE = false;
    % if CROP_FOR_DISP_SAVE is set to true, then the calculations only involve the tumor area, i.e. the area specified by x_m, x_M, y_m, y_M. 
    % for the sequences with already cropped images, this has no effect since the image parameters are set such that x_m = 1, x_M = L, etc. (cf xls file)

beh_par.EVALUATE_IN_ROI = false;
    % if EVALUATE_IN_ROI is set to true, the errors are calculated using only the pixels in the region of interest (ROI)


%% OTHER PARAMETERS (do not modify)
  
beh_par.SAVE_WARPED_IM = beh_par.SAVE_INIT_OF_WARP_JPG || beh_par.SAVE_PCA_RECONSTR_JPG || beh_par.SAVE_PRED_IM;
beh_par.SAVE_PREDICTION_PLOT = true;
beh_par.EVALUATE_PREDICTION = true;

end