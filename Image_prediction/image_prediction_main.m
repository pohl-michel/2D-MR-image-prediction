% Image Prediction Using PCA and Dynamic Training with RNNs
% 
% This script performs image prediction by projecting deformation vector fields (DVFs) 
% onto a Principal Component Analysis (PCA) space and predicting these projections using 
% various methods. It supports dynamic training using RNN-based methods such as UORO, 
% RTRL, and SnAp-1. Additionally, the script can automatically optimize hyper-parameters 
% and the number of PCA components through grid search, and it provides a detailed analysis 
% of how these factors influence prediction performance.
%
% Key Features:
% - Supports a range of prediction methods including linear regression, RTRL, UORO, SnAp-1, and more.
% - Optimizes the number of PCA components and other hyper-parameters.
% - Evaluates the influence of hyper-parameters on the prediction results.
% - Provides options for computing and saving intermediate results, including optical flow (OF), 
%   PCA components, predicted deformations, and warped images.
% - Fully compatible with parallel computing, which can speed up computations.
%
% Author : Pohl Michel
% License : 3-clause BSD License


clear all
close all
clc

% Move to the 'Image_prediction' folder if it exists in the current directory
if any(strcmp({dir(pwd).name},'Image_prediction'))
    cd Image_prediction
end

% Add required paths for various functions used in this script
addpath(genpath('1. Optical flow calculation'))    
addpath(genpath('2. Optical flow evaluation'))    
addpath(genpath('3. Image warping and prediction'))    
addpath(genpath('4. PCA of the DVF')) 
addpath(genpath('5. Auxiliary functions (loading, saving files and parameters)'))    
addpath(genpath('6. Auxiliary functions (calculus)'))

%% PARAMETERS 

% Set random number generator for reproducibility, using Mersenne Twister algorithm
rng(0, "twister");

% Load behavior-related parameters (controls program flow and options)
beh_par = load_impred_behavior_parameters();

% Load directory paths and input/output file parameters
path_par = load_impred_path_parameters();

% Input image sequences
input_im_dir_suffix_tab = [
    % string('add your input sequence directory here');
    string('2. sq sl010 sag Xcs=125');
    string('3. sq sl010 sag Xcs=80');   
    string('4. sq sl014 sag Xcs=165');  
    string('5. sq sl014 sag Xcs=95');  
    ];

% Prediction methods to evaluate if beh_par.OPTIMIZE_NB_PCA_CP == true, otherwise the prediction method is that specified in load_pred_par.m
pred_meths = {'multivariate linear regression', 'LMS', 'UORO', 'SnAp-1', 'DNI', 'RTRL v2', 'no prediction', 'fixed W'};

% Set the number of PCA components to use for each sequence
br_model_par.nb_pca_cp_tab = [4, 4, 4, 4]; % length = nb of sequences to process

% Number of image sequences to process
nb_seq = length(input_im_dir_suffix_tab);

for im_seq_idx = 1:nb_seq

    % Define the input directory for the current image sequence
    path_par.input_im_dir_suffix = input_im_dir_suffix_tab(im_seq_idx);
    path_par.input_im_dir = sprintf('%s\\%s', path_par.input_im_dir_pref, path_par.input_im_dir_suffix);  
    
    % Load optical flow parameters, which are specific to the input sequence
    OF_par = load_OF_param(path_par);
   
    % Load image parameters
    im_par = load_im_param(path_par);
    
    % Load parameters for display options (visualization settings)
    disp_par = load_impred_display_parameters(path_par);

    % Load parameters for image warping
    warp_par = load_warp_par();
    nb_runs_for_cc_eval = warp_par.nb_runs_for_cc_eval;  % number of evaluation runs for image warping
    
    % Parameters of the breathing model (here the PCA respiratory model)
    br_model_par.nb_pca_cp = br_model_par.nb_pca_cp_tab(im_seq_idx);  % that's the max nb of PCA components when performing hyper-parameter optimization (i.e., when beh_par.OPTIMIZE_NB_PCA_CP is set to true)

    % Directory for time-series data used in prediction
    path_par.time_series_dir = path_par.input_im_dir_suffix;  % for the function 'train_and_predict' 
    
    %% ---------------------------------------------------------------------------------------------------------------------------------------------------
    %  MAIN PROGRAM STARTS HERE --------------------------------------------------------------------------------------------------------------------------
    %  --------------------------------------------------------------------------------------------------------------------------------------------------- 

    % Initialize structures for storing evaluation results and hyper-parameters
    [eval_results, time_signal_pred_results, hppars] = init_structs(beh_par);

    % Save original image sequences and region of interest (ROI)
    if beh_par.SAVE_ORG_IM_SQ
        save2Dimseq(im_par, path_par, disp_par);
    end       
    if beh_par.SAVE_ROI_DISPLAY
        saveROIposition(im_par, path_par, disp_par);
    end    
    
    % Compute the optical flow if needed
    if beh_par.COMPUTE_OPTICAL_FLOW
        eval_results = compute_2Dof(OF_par, im_par, path_par);  % bug fix/improvement: rather pass eval_results as a parameter to update 
    end

    % Save the computed optical flow as images (JPG format)
    if beh_par.SAVE_OF_JPG
        save2DOFjpg(beh_par, path_par, disp_par, OF_par, im_par);
    end
    
    % Add the directory containing functions related to time series forecasting
    cd ..
    addpath(genpath('Time_series_forecasting'))   
    path_par = update_path_par_move_parent_dir(path_par);

    % Hyper-parameter optimization
    if beh_par.OPTIMIZE_NB_PCA_CP
        % Loop through prediction methods and perform hyper-parameter optimization and evaluation of the test set for each of these methods
        for pred_meth_idx = 1:length(pred_meths)
            pred_meth = pred_meths{pred_meth_idx};
            [eval_results, best_pred_par_struct, best_pca_cp_tab] = select_nb_pca_cp(beh_par, disp_par, OF_par, im_par, path_par, br_model_par, eval_results, warp_par, pred_meth);   
            eval_im_pred_best_par(eval_results, best_pred_par_struct, best_pca_cp_tab, beh_par, disp_par, OF_par, im_par, path_par, br_model_par, warp_par, pred_meth)
        end
        
    else
        % If PCA component optimization is not enabled (OPTIMIZE_NB_PCA_CP = false), the program uses predefined prediction parameters.

        % Load prediction parameters specific to the input sequence
        pred_par = load_pred_par(path_par);
        pred_par.t_eval_start = 1 + pred_par.tmax_cv;  % set evaluation start time (beginning of the test set)
        pred_par.nb_predictions = im_par.nb_im - pred_par.t_eval_start + 1;  % set the nb. of predictions based on remaining images after the eval. start time

        % Save the mean image computed from the image sequence
        if beh_par.SAVE_MEAN_IMAGE
           save_mean_image(im_par, path_par, disp_par, pred_par); 
        end
    
        % Evaluate the warping results based on the initial optical flow (no prediction applied yet)
        if beh_par.EVAL_INIT_OF_WARP 
            dvf_type = 'initial DVF';
            eval_results = eval_of_warp_corr(dvf_type, im_par, OF_par, path_par, warp_par, pred_par, br_model_par, disp_par, beh_par, eval_results, time_signal_pred_results);
        end

        % Evaluate performance corresponding to edge case where the previous image is used as the prediction
        if beh_par.NO_PRED_AT_ALL
            % pred_par.horizon = 1; % to modify as needed
            dvf_type = 'no prediction';
            eval_results = eval_of_warp_corr(dvf_type, im_par, OF_par, path_par, warp_par, pred_par, br_model_par, disp_par, beh_par, eval_results, time_signal_pred_results);        
        end        
        
        % Perform PCA on the time-varying deformation vector field (DVF) to reduce dimensionality
        if beh_par.PCA_OF_DVF
            [W, F, Xmean, eval_results] = compute_PCA_of_DVF(beh_par, disp_par, OF_par, im_par, path_par, pred_par, br_model_par, eval_results);
        end

        % Evaluate the warping results based on the DVF reconstructed from PCA components
        if beh_par.EVAL_PCA_RECONSTRUCT
            dvf_type = 'DVF from PCA';
            eval_results = eval_of_warp_corr(dvf_type, im_par, OF_par, path_par, warp_par, pred_par, br_model_par, disp_par, beh_par, eval_results, time_signal_pred_results);
        end            
        
        % Train and evaluate a model to predict the PCA weights
        if beh_par.TRAIN_EVAL_PREDICTOR
            path_par.time_series_data_filename = write_PCAweights_mat_filename(OF_par, path_par, br_model_par);
            [Ypred, avg_pred_time, pred_loss_function] = train_and_predict(path_par, pred_par, beh_par, br_model_par);
            time_signal_pred_results = pred_eval(beh_par, path_par, pred_par, disp_par, Ypred, avg_pred_time);
            write_time_series_pred_log_file(path_par, beh_par, pred_par, time_signal_pred_results);    
        end        

        % Reconstructing the future DVFs based on the predicted PCA weights and warping the initial image to perform video prediction
        if beh_par.IM_PREDICTION
            % warping with the predicted optical flow
            dvf_type = 'predicted DVF';  
            
            % update the nb. of eval. runs to ensure consistency with the time signal results
            warp_par.nb_runs_for_cc_eval = min(nb_runs_for_cc_eval, time_signal_pred_results.nb_correct_runs);
            time_signal_pred_results.nb_correct_runs = min(nb_runs_for_cc_eval, time_signal_pred_results.nb_correct_runs);      
            
            % DVF reconstruction and warping
            eval_results = eval_of_warp_corr(dvf_type, im_par, OF_par, path_par, warp_par, pred_par, br_model_par, disp_par, beh_par, eval_results, time_signal_pred_results);
        end

        % Write a log file containing all results relative to image prediction
        write_im_pred_log_file(path_par, beh_par, im_par, OF_par, hppars, pred_par, br_model_par, warp_par, eval_results);
        
    end

    % Clean up the path and return to the initial directory
    rmpath(genpath('Time_series_forecasting'))   
    cd(path_par.im_pred_dir)
    path_par = update_path_par_return_child_dir(path_par);    

end


function [eval_results, time_signal_pred_results, hppars] = init_structs(beh_par)
% Helper function initializing the results and hyper-parameter structures

    eval_results = struct();
        eval_results.whole_im = struct();  % For whole image evaluations
        if beh_par.EVALUATE_IN_ROI
            eval_results.roi = struct();  % For region-of-interest evaluations
        end
        time_signal_pred_results = struct();
        hppars = struct();
end
