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
    % For saving the initial image of the sequence with the region of interest (ROI) in a red rectangle 
beh_par.SAVE_ORG_IM_SQ = false;
    % For saving jpg images of the original image sequence that we try to predict
beh_par.SAVE_MEAN_IMAGE = false;
    % For saving the average image (over every time step) of the test set

beh_par.COMPUTE_OPTICAL_FLOW = true;
    % For computing the optical flow (OF) / deformation vector field (DVF) at every time step and saving the results as mat files
beh_par.SAVE_OF_JPG = true;
    % For saving DVF images
beh_par.EVAL_INIT_OF_WARP = true;
    % For computing statistics describing deformable image registration (DIR) accuracy of the test set
beh_par.SAVE_INIT_OF_WARP_JPG = true;
    % For saving the image at t=1 warped by the initial DVF/OF at time t for each time step t of the test set

beh_par.OPTIMIZE_NB_PCA_CP = true;
    % For optimizing the number of PCA components for prediction using hyper-parameter grid search
beh_par.REGISTRATION_ERROR_CV = true;
    % For optimizing the number of PCA components based on the DVF registration NRMSE rather than cross-correlation between the initial image and the warped images 

beh_par.PCA_OF_DVF = false;
    % For computing principal component analysis (PCA) from the DVF data
beh_par.SAVE_PCA_CP_WEIGHTS_JPG = false;
    % For saving jpg images of the principal components (the 2D principal deformation vectors)
beh_par.EVAL_PCA_RECONSTRUCT = false;
    % For evaluating the quality of the DVFs reconstructed with a few principal components by warping the initial image at t=1 by the DVF at time t for each
    % time step t of the test set
beh_par.SAVE_PCA_RECONSTR_JPG = false;
    % For saving the image at t=1 warped by the DVF/OF reconstructed from PCA at time t for each time step t of the test set

beh_par.TRAIN_EVAL_PREDICTOR = true;
    % For training and evaluating the prediction/forecast of the PCA weights
beh_par.SAVE_PRED_RESULTS = true;
    % For saving a mat file containing the prediction results (predicted PCA weights, loss function, and prediction time)

beh_par.NO_PRED_AT_ALL = false; 
    % The original images are used instead of predicted images as a base case for performance evaluation
beh_par.IM_PREDICTION = true; 
    % Performing image prediction
beh_par.SAVE_PRED_IM = true; 
    % Saving the predicted images and the thermal difference images

beh_par.CROP_FOR_DISP_SAVE = false;
    % For saving the deformation vector field images, predicted images, or error images only in the ROI, i.e. the area specified by x_m, x_M, y_m, y_M
beh_par.EVALUATE_IN_ROI = false;
    % if EVALUATE_IN_ROI is set to true, the prediction errors are calculated using only the pixels in the region of interest (ROI)


%% OTHER PARAMETERS (do not modify)
beh_par.SAVE_WARPED_IM = beh_par.SAVE_INIT_OF_WARP_JPG || beh_par.SAVE_PCA_RECONSTR_JPG || beh_par.SAVE_PRED_IM;
beh_par.SAVE_PREDICTION_PLOT = true;
beh_par.EVALUATE_PREDICTION = true;

end