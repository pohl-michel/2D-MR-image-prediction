function [ beh_par ] = load_impred_behavior_parameters()
% LOAD_IMPRED_BEHAVIOR_PARAMETERS Loads and returns the behavior parameters for the image prediction algorithm.
%
% INPUTS:
%   (None)
%
% OUTPUTS:
% - beh_par (struct): Structure containing the behavior parameters for the image prediction algorithm
%
% Author : Pohl Michel
% License : 3-clause BSD License


%% USER-ADJUSTABLE PARAMETERS

beh_par.SAVE_ROI_DISPLAY = false;  % Save the first image in the sequence with the region of interest (ROI) in a red rectangle
beh_par.SAVE_ORG_IM_SQ = false;    % Save jpg images of the original image sequence
beh_par.SAVE_MEAN_IMAGE = false;   % Save the average image of the test set

beh_par.COMPUTE_OPTICAL_FLOW = true;   % Compute and save (as .mat files) optical flow at every time step
beh_par.SAVE_OF_JPG = true;            % Save optical flow images as jpg files
beh_par.EVAL_INIT_OF_WARP = true;      % Evaluate deformable image registration (DIR) accuracy of the test set
beh_par.SAVE_INIT_OF_WARP_JPG = true;  % Save the first image warped by the initial deformation vector field (DVF) at every time step of the test set

beh_par.OPTIMIZE_NB_PCA_CP = true;     % Optimize number of PCA components via grid search
beh_par.REGISTRATION_ERROR_CV = true;  % Optimize PCA components based on registration nRMSE with predicted DVF (instead of cross-correlation between the original and warped images)

beh_par.PCA_OF_DVF = true;               % Compute PCA from the DVF data
beh_par.SAVE_PCA_CP_WEIGHTS_JPG = true;  % Save 2D principal deformation vectors fields (PCA components) as jpg images
beh_par.EVAL_PCA_RECONSTRUCT = false;     % Evaluate quality of DVF reconstructed using PCA (by warping the first image at t=1 by the DVF at time t when t corresponds to the test set)
beh_par.SAVE_PCA_RECONSTR_JPG = false;    % Save the image at t=1 warped by the DVF reconstructed from PCA at time t for each time step t of the test set

beh_par.TRAIN_EVAL_PREDICTOR = true;  % Train the PCA weight forecasting algorithm and evaluate it
beh_par.SAVE_PRED_RESULTS = true;     % Save a .mat file containing the prediction results (predicted PCA weights, loss function, and time performance)

beh_par.NO_PRED_AT_ALL = false;   % Use original images instead of predictions for performance evaluation
beh_par.IM_PREDICTION = true;     % Perform image prediction
beh_par.SAVE_PRED_IM = true;      % Save the predicted images and thermal difference images

beh_par.CROP_FOR_DISP_SAVE = false;  % Save the initial sequence, deformation vector field images, predicted images, or error images only in the ROI
beh_par.EVALUATE_IN_ROI = true;     % Evaluate prediction errors within the ROI as well - the number of PCA components is selected using the whole image though


%% PARAMETERS AUTOMATICALLY SET
beh_par.SAVE_WARPED_IM = beh_par.SAVE_INIT_OF_WARP_JPG || beh_par.SAVE_PCA_RECONSTR_JPG || beh_par.SAVE_PRED_IM;
beh_par.SAVE_PREDICTION_PLOT = true;
beh_par.EVALUATE_PREDICTION = true;
if beh_par.IM_PREDICTION
    beh_par.TRAIN_EVAL_PREDICTOR = true; % because the "time_signal_pred_results" variable is required in the chain
    beh_par.SAVE_PRED_RESULTS = true; % because the image prediction step will load the predicted PCA components from disk.
end
beh_par.NORMALIZE_EIGENVECTORS = true; % otherwise different machines or versions of matlab can return principal components and weights with opposite signs
if beh_par.OPTIMIZE_NB_PCA_CP % to avoid having a bug when logging results
    beh_par.PCA_OF_DVF = false;
end

end