% Prediction of multi-dimensional time-series data using different methods: RNNs trained with UORO, RTRL, and SnAp-1, DNI, LMS and linear regression.
% The data provided consists of the 3D position of external markers on the chest used during the radiotherapy treatment to accurately deliver radiation.
%
% In this function, the hyperparameters relative to each prediction method are optimized using grid search
% 
% Matlab's parallel computing toolbox is needed to do parallel processing, which significantly reduces the computing time of grid search.
% To run the program without performing parallel computing, one can replace the "parfor" instructions with "for" instructions.
%
% Author : Pohl Michel
% Date : September 27th, 2021
% Version : v1.1
% License : 3-clause BSD License


clear all
close all
clc

% current directory
pwd_split_cell = strsplit(pwd, '\');
pwdir = string(pwd_split_cell(end));

if pwdir == "Future_frame_prediction"
    cd Time_series_forecasting
end

addpath(genpath('1. Main prediction functions'))    
addpath(genpath('2. RNN prediction functions'))    
addpath(genpath('3. Other prediction functions'))    
addpath(genpath('4. Auxiliary functions (calculus)')) 
addpath(genpath('5. Auxiliary functions (loading, saving files and parameters)'))    
addpath(genpath('6. Auxiliary functions (plotting)'))

%% PARAMETERS

% GPU computing or not
beh_par.GPU_COMPUTING = false;

% Directories 
path_par = load_sigpred_path_parameters();

% Display parameters
disp_par = load_sigpred_display_parameters(path_par);  
               
% Prediction methods to test
pred_meths = {'multivariate linear regression', 'LMS', 'UORO', 'SnAp-1', 'DNI', 'no prediction', 'fixed W', 'RTRL v2'};

nb_seq = length(path_par.time_series_dir_tab);
for seq_idx = 1:nb_seq

    % filename of the sequence being studied
    path_par.time_series_dir = path_par.time_series_dir_tab(seq_idx);
    path_par.input_seq_dir = sprintf('%s\\%s', path_par.parent_seq_dir, path_par.time_series_dir);
    path_par.time_series_data_filename = sprintf('%s\\%s', path_par.input_seq_dir, path_par.time_series_data_filename_suffix);    

    for pred_meth_idx = 1:length(pred_meths)
        pred_meth = pred_meths{pred_meth_idx};

        % Parameters concerning the prediction of the position of objects
        pred_par = load_pred_par(path_par, pred_meth);
        % Hyperparameters to optimize 
        hppars = load_hyperpar_cv_info(pred_par);
    
        %% PROGRAM
        [optim, best_par] = train_eval_predictor_mult_param(hppars, pred_par, path_par, disp_par, beh_par);
        par_influence = evaluate_par_influence_grid(hppars, pred_par, optim);
        write_hppar_optim_log_file(hppars, pred_par, path_par, optim, best_par, par_influence, beh_par);

    end
    
end