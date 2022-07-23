% This script performs image prediction by predicting the projection of deformation vector fields onto a PCA space with various methods, including
% RNNs dynamically trained using UORO, RTRL and SnAp-1. 
% It can automatically select the optimal prediction hyper-parameters and number of PCA components using grid search, and analyze their influence on the prediction performance. 
%
% Author : Pohl Michel
% Date : September 26th, 2021
% Version : v1.0
% License : 3-clause BSD License

clear all
close all
clc

rmpath(genpath('PhD CMIG work'));
addpath(genpath('1. Optical flow calculation'))    
addpath(genpath('2. Optical flow evaluation'))    
addpath(genpath('3. Image warping and prediction'))    
addpath(genpath('4. PCA of the DVF')) 
addpath(genpath('Auxiliary functions (loading, saving files and parameters)'))    
addpath(genpath('Auxiliary functions (calculus)'))    

%% PARAMETERS 

% Program behavior
beh_par = load_impred_behavior_parameters();

% Directories 
path_par = load_path_parameters();

% Input image sequences
input_im_dir_suffix_tab = [
    %string('write here the sequence name');
     string('2. sq sl010 sag Xcs=125');
%     string('3. sq sl010 sag Xcs=80');   
%     string('4. sq sl014 sag Xcs=165');  
%     string('5. sq sl014 sag Xcs=95');  
    
    ];

%br_model_par.nb_pca_cp_tab = [3, 3, 3, 3]; % length = nb of sequences to process
%br_model_par.nb_pca_cp_tab = [3, 3, 2];
br_model_par.nb_pca_cp_tab = [3];

nb_seq = length(input_im_dir_suffix_tab);
for im_seq_idx = 1:nb_seq

    % directory of the input images (text string inside the function)
    path_par.input_im_dir_suffix = input_im_dir_suffix_tab(im_seq_idx);
    path_par.input_im_dir = sprintf('%s\\%s', path_par.input_im_dir_pref, path_par.input_im_dir_suffix);  
    
    % Parameters concerning optical flow - depend on the input sequence - they are in an excel file
    OF_par = load_OF_param(path_par);
   
    % Image parameters
    im_par = load_im_param(path_par);
    
    % Display parameters
    disp_par = load_display_parameters(path_par);

    % Parameters for image warping
    warp_par = load_warp_par();
    nb_runs_for_cc_eval = warp_par.nb_runs_for_cc_eval;
    
    % Parameters concerning the breathing model (PCA for instance)
    br_model_par.nb_pca_cp = br_model_par.nb_pca_cp_tab(im_seq_idx);
        % maximum number of PCA components taken into account if optimization of all hyper-parameters (beh_par.OPTIMIZE_NB_PCA_CP = true)

    % PCA weights filename
    OF_param_str = sprintf_OF_param(OF_par);
    path_par.time_series_data_filename = sprintf('%s\\PCA weights %s %s %d cpts.mat', path_par.temp_var_dir, path_par.input_im_dir_suffix, OF_param_str, br_model_par.nb_pca_cp);       
    
    path_par.time_series_dir = path_par.input_im_dir_suffix; % for the function 'train and predict' 
    
    %% ---------------------------------------------------------------------------------------------------------------------------------------------------
    %  PROGRAM -------------------------------------------------------------------------------------------------------------------------------------------
    %  --------------------------------------------------------------------------------------------------------------------------------------------------- 

    % eval_results = initialize_eval_results();
    eval_results = struct();
    time_signal_pred_results = struct();
    hppars = struct();

    if beh_par.SAVE_ORG_IM_SQ
        save2Dimseq(im_par, path_par, disp_par);
    end        
    
    if beh_par.SAVE_ROI_DISPLAY
        saveROIposition(im_par, path_par, disp_par);
    end    
    
    if beh_par.COMPUTE_OPTICAL_FLOW
        eval_results = compute_2Dof(OF_par, im_par, path_par); 
    end

    if beh_par.SAVE_OF_JPG
        save_2DOF(beh_par, path_par, disp_par, OF_par, im_par);
    end
    
    if beh_par.OPTIMIZE_NB_PCA_CP
       
        % The functions for prediction are in the folder 'Time series forecasting'
        cd ..
        addpath(genpath('Time series forecasting'))   
        path_par = update_path_par_move_parent_dir(path_par);

        [eval_results, best_pred_par_struct, best_pca_cp_tab] = select_nb_pca_cp(beh_par, disp_par, OF_par, im_par, path_par, br_model_par, eval_results, warp_par);   
        eval_im_pred_best_par(eval_results, best_pred_par_struct, best_pca_cp_tab, beh_par, disp_par, OF_par, im_par, path_par, br_model_par, warp_par)
        
        rmpath(genpath('Time series forecasting'))   
        cd 'PhD pred 2D MRI'
        path_par = update_path_par_return_child_dir(path_par);
        
    else

        % The functions for prediction are in the folder 'Time series forecasting'
        cd ..
        addpath(genpath('Time series forecasting'))   
        path_par = update_path_par_move_parent_dir(path_par);

        % Parameters concerning the prediction of the position of objects
        pred_par = load_pred_par(path_par);
        pred_par.t_eval_start = 1 + pred_par.tmax_cv; % car je veux faire l'eval sur l'ensemble de test
        % � ce moment pr�cis du d�veloppement du code, on ne s'int�resse pas � la validation crois�e
        pred_par.nb_predictions = im_par.nb_im - pred_par.t_eval_start + 1;    

        if beh_par.SAVE_MEAN_IMAGE
           save_mean_image(im_par, path_par, disp_par, pred_par); 
        end
    
        if beh_par.EVAL_INIT_OF_WARP 
            of_type_idx = 1; % warping with the initial optical flow
            eval_results = eval_of_warp_corr( of_type_idx, im_par, OF_par, path_par, warp_par, pred_par, br_model_par, disp_par, beh_par, eval_results, time_signal_pred_results);
        end

        if beh_par.NO_PRED_AT_ALL
            %pred_par.horizon = 7; % to modify as needed
            of_type_idx = 3; % no warping - evaluation when no prediction is performed
            eval_results = eval_of_warp_corr( of_type_idx, im_par, OF_par, path_par, warp_par, pred_par, br_model_par, disp_par, beh_par, eval_results, time_signal_pred_results);        
        end        
        
        if beh_par.PCA_OF_DVF
            [W, F, Xmean, eval_results] = PCA_of_DVF(beh_par, disp_par, OF_par, im_par, path_par, pred_par, br_model_par, eval_results);
        end

        if beh_par.EVAL_PCA_RECONSTRUCT
            of_type_idx = 2; % warping with the DVF reconstructed from PCA
            eval_results = eval_of_warp_corr( of_type_idx, im_par, OF_par, path_par, warp_par, pred_par, br_model_par, disp_par, beh_par, eval_results, time_signal_pred_results);
        end            
        
        if beh_par.TRAIN_EVAL_PREDICTOR

            [Ypred, avg_pred_time, pred_loss_function] = train_and_predict(path_par, pred_par, beh_par);
            time_signal_pred_results = pred_eval(beh_par, path_par, pred_par, disp_par, Ypred, avg_pred_time);
            write_time_series_pred_log_file(path_par, beh_par, pred_par, time_signal_pred_results);    
        end        

        if beh_par.IM_PREDICTION
            of_type_idx = 3; % warping with the predicted optical flow
            warp_par.nb_runs_for_cc_eval = min(nb_runs_for_cc_eval, time_signal_pred_results.nb_correct_runs);
            time_signal_pred_results.nb_correct_runs = min(nb_runs_for_cc_eval, time_signal_pred_results.nb_correct_runs);              
            eval_results = eval_of_warp_corr( of_type_idx, im_par, OF_par, path_par, warp_par, pred_par, br_model_par, disp_par, beh_par, eval_results, time_signal_pred_results);
        end

        write_im_pred_log_file(path_par, beh_par, im_par, OF_par, hppars, pred_par, br_model_par, warp_par, eval_results);
        
        rmpath(genpath('Time series forecasting'))   
        cd 'PhD pred 2D MRI'
        path_par = update_path_par_return_child_dir(path_par);
        
    end

end