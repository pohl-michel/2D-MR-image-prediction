function [ beh_par ] = load_impred_behavior_parameters()
% laoding behavior parameters
    % modifier la fonction au cours de l'écriture du programme (en m'inspirant du programme de master 3D)
    % pour le moment c'est un copier collé de la version 2D

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
    % just look at the content of the structure eval_results at the moment.

beh_par.OPTIMIZE_NB_PCA_CP = false;

beh_par.GPU_COMPUTING = false; 
    % use of the GPU when performing prediction

beh_par.CROP_FOR_DISP_SAVE = false;
    % if the optical flow is calculated with the entire chest images, and CROP_FOR_DISP_SAVE = true then the optical flow is displayed only around the
    % tumor area, or the area specified by x_m, x_M, y_m, y_M, z_m, z_M. 
    % for the sequence(s) with already cropped images this has no effect since the image parameters are set such x_m = 1, x_M = L, etc. (cf xls file)
beh_par.EVALUATE_IN_ROI = false;



%% OTHER PARAMETERS (do not modify)
  
beh_par.SAVE_WARPED_IM = beh_par.SAVE_INIT_OF_WARP_JPG || beh_par.SAVE_PCA_RECONSTR_JPG || beh_par.SAVE_PRED_IM;
beh_par.SAVE_PREDICTION_PLOT = true;
beh_par.EVALUATE_PREDICTION = true;

end