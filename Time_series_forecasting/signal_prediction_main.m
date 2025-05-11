% Prediction of multi-dimensional time-series data using different methods: RNNs trained with RTRL, UORO, DNI, and SnAp-1, LMS and linear regression.
%
% Author : Pohl Michel
% Date : September 27th, 2021
% Version : v1.0
% License : 3-clause BSD License

clear all
close all
clc

% current directory
pwd_split_cell = strsplit(pwd, '\');
pwdir = string(pwd_split_cell(end));

if any(strcmp({dir(pwd).name},'Time_series_forecasting'))
    cd Time_series_forecasting
end

addpath(genpath('1. Main prediction functions'))    
addpath(genpath('2. RNN prediction functions'))    
addpath(genpath('other_prediction_functions'))    
addpath(genpath('4. Auxiliary functions (calculus)')) 
addpath(genpath('5. Auxiliary functions (loading, saving files and parameters)'))    
addpath(genpath('6. Auxiliary functions (plotting)'))

%% PARAMETERS 

% random number generator - generator seed set to 2 and the algorithm set to Mersenne Twister
rng(0, "twister");

% Program behavior
beh_par = load_sigpred_behavior_parameters();

% Directories 
path_par = load_sigpred_path_parameters();

% Display parameters
disp_par = load_sigpred_display_parameters(path_par);

% Prediction parameters
% cf line 56

%% ---------------------------------------------------------------------------------------------------------------------------------------------------
%  PROGRAM -------------------------------------------------------------------------------------------------------------------------------------------
%  --------------------------------------------------------------------------------------------------------------------------------------------------- 

nb_seq = length(path_par.time_series_dir_tab);
for seq_idx = 1:nb_seq

    path_par.time_series_dir = path_par.time_series_dir_tab(seq_idx);
    path_par.input_seq_dir = sprintf('%s\\%s', path_par.parent_seq_dir, path_par.time_series_dir_tab(seq_idx));
    path_par.time_series_data_filename = sprintf('%s\\%s', path_par.input_seq_dir, path_par.time_series_data_filename_suffix);
    
    % Parameters concerning the prediction of the position of objects / signals
    pred_par = load_pred_par(path_par);
    pred_par.t_eval_start = 1 + pred_par.tmax_cv; % car je veux faire l'eval sur l'ensemble de test
    pred_par.nb_predictions = pred_par.tmax_pred - pred_par.t_eval_start + 1;
    
    if (pred_par.data_type == 2) % prediction of PCA components
        path_par.input_im_dir_suffix = path_par.time_series_dir; % because path_par.input_im_dir_suffix is required for saving prediction plots  
    end
    
    if beh_par.TRAIN_AND_PREDICT
        [Ypred, avg_pred_time, pred_loss_function] = train_and_predict(path_par, pred_par, beh_par);
    end

    if (beh_par.SAVE_PREDICTION_PLOT)||(beh_par.EVALUATE_PREDICTION)
        eval_results = pred_eval(beh_par, path_par, pred_par, disp_par, Ypred, avg_pred_time);
    end

    write_time_series_pred_log_file(path_par, beh_par, pred_par, eval_results);    
    
end