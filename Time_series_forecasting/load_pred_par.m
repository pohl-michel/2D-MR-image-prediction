function [pred_par] = load_pred_par(path_par, pred_meth)
% LOAD_PRED_PAR Loads prediction parameters from a specified file and sets additional parameters for different prediction methods.
%
% This function reads the prediction parameters stored in a file and returns them in the `pred_par` structure. 
% Depending on the prediction method selected (or passed as an argument), additional settings like normalization,
% gradient clipping, GPU computing, and other algorithm-specific parameters are set.
%
% INPUTS:
% - path_par (struct): Structure containing path parameters, particularly `path_par.pred_par_filename_suffix` and `path_par.input_seq_dir`, 
%   which are used to locate the file storing the prediction parameters.
% - pred_meth (string, optional): A string specifying the prediction method. If not provided, the method specific by the user in pred_par.pred_meth is used.
%   Accepted methods include:
%   - 'multivariate linear regression'
%   - 'RTRL' (Real-Time Recurrent Learning, "standard" implementation)
%   - 'RTRL v2' (another implementation of RTRL, based on Haykin's book)
%   - 'no prediction'
%   - 'LMS' (Least Mean Squares)
%   - 'UORO' (Unbiased Online Recurrent Optimization)
%   - 'univariate linear regression'
%   - 'SnAp-1' (default, Sparse 1-step Approximation)
%   - 'DNI' (Decoupled Neural Interfaces)
%   - 'fixed W' (RNN model with fixed hidden layer parameters)
%
% OUTPUT:
% - pred_par (struct): A structure containing the loaded and computed prediction parameters
%
% REMARKS:
% - GPU_COMPUTING was found to increase computation speed with "RTRL" (and "RTRL v2"). It requires an NVidia GPU processor and Matlab's parallel computing toolbox.
% - GPU_COMPUTING was also implemented for UORO but not for SnAp-1 and DNI (yet)
% - Parallel computing can cause execution to stop with test data if not enough memory on the GPU and prevents debugging. That can be solved by 
% setting the number of workers within the parfor loop accordingly: https://www.mathworks.com/help/parallel-computing/parfor.html
% - One can choose between "nRMSE" and "RMSE" for pred_par.cross_val_metric at the moment, although it's the same practically because the current model is sequence-specific.
% 
% Author : Pohl Michel
% License : 3-clause BSD License

    % Filename containing the prediciton parameters
    path_par.pred_par_filename = sprintf('%s\\%s', path_par.input_seq_dir, path_par.pred_par_filename_suffix);

    % Loading the prediction parameters contained in that file
    opts = detectImportOptions(path_par.pred_par_filename);
    opts = setvartype(opts,'double');
    opts.DataRange = '2:2';
    pred_par = table2struct(readtable(path_par.pred_par_filename, opts));
    
    % Default GPU computing flag (false) - can be changed inside switch case statement below based on the method - don't change it here!
    pred_par.GPU_COMPUTING = false;  

    % Enabling parallel computing to speed up certain computations.
    pred_par.PARALLEL_COMPUTING = true; 

    % Default cross-validation metric for evaluating the prediction model (normalized RMSE).
    pred_par.cross_val_metric = 'nRMSE';  % possible options: "nRMSE" or "RMSE"

    switch nargin
        case 1 % Manually choosing the prediction method in image_prediction_main.m (if OPTIMIZE_NB_PCA_CP == false) or signal_prediction_main.m
            pred_par.pred_meth = 'SnAp-1';
        case 2 % Prediction method specified in image_prediction_main.m (if OPTIMIZE_NB_PCA_CP == true) or sigpred_hyperparameter_optimization_main.m
            pred_par.pred_meth = pred_meth;
    end
    
    % Adjusting parameters based on the selected prediction method.
    switch(pred_par.pred_meth)
        case 'multivariate linear regression'
            pred_par.nb_runs = 1;  % not a stochastic method
            pred_par.NORMALIZE_DATA = false;
            pred_par.tmax_training = 160;  % MR data (ETH Zurich - CMIG paper)

            % pred_par.tmax_training = 180; % markers 3.33 Hz (CPMB paper)
            % pred_par.tmax_training = 540; % markers 10 Hz (CPMB paper)
            % pred_par.tmax_training = 1620; % markers 30 Hz (CPMB paper)

        case 'RTRL'
            % Real-Time Recurrent Learning (version described in Haykin's book).
            pred_par.NORMALIZE_DATA = true;
            pred_par.update_meth = 'stochastic gradient descent';
            pred_par.GRAD_CLIPPING = true;
            pred_par.grad_threshold = 100.0;    
            pred_par.Winit_std_dev = 0.02;
            pred_par.GPU_COMPUTING = true;  % experimentally faster

        case 'RTRL v2'
            % Another version of Real-Time Recurrent Learning (as described by Tallec et al.).
            pred_par.NORMALIZE_DATA = true;
            pred_par.update_meth = 'stochastic gradient descent'; 
            pred_par.GRAD_CLIPPING = true; 
            pred_par.grad_threshold = 100.0;  
			pred_par.Winit_std_dev = 0.02;
            pred_par.GPU_COMPUTING = true;
            pred_par.PARALLEL_COMPUTING = false;  % experimentally faster

        case 'no prediction'
            % No prediction is performed, instead using the latest acquired value.
            pred_par.nb_runs = 1;
            pred_par.NORMALIZE_DATA = false;
            pred_par.SHL = 1;  % last value used for prediction.

        case 'LMS'
            % Least Mean Squares (multivariate).
            pred_par.nb_runs = 1; % not a stochastic method
            pred_par.NORMALIZE_DATA = true;    
            pred_par.update_meth = 'stochastic gradient descent';
            pred_par.GRAD_CLIPPING = true;
            pred_par.grad_threshold = 100.0;

        case 'UORO'
            % Unbiased online recurrent optimization
            pred_par.NORMALIZE_DATA = true;
            pred_par.eps_tgt_fwd_prp = 0.0000001;
            pred_par.eps_normalizers = 0.0000001;
            pred_par.update_meth = 'stochastic gradient descent';
            pred_par.GRAD_CLIPPING = true;
			pred_par.grad_threshold = 100.0;  
			pred_par.Winit_std_dev = 0.02;	

        case 'univariate linear regression'
            pred_par.nb_runs = 1; % not a stochastic method
            pred_par.NORMALIZE_DATA = false;
            pred_par.tmax_training = 160; % cine-MR sequence prediction (CMIG paper)

            % pred_par.tmax_training = 180; % markers 3.33 Hz (CPMB paper)
            % pred_par.tmax_training = 540; % markers 10 Hz (CPMB paper)
            % pred_par.tmax_training = 1620; % markers 30 Hz (CPMB paper)

		case 'SnAp-1'
            % Sparse 1-step approximation
            pred_par.NORMALIZE_DATA = true;
            pred_par.update_meth = 'stochastic gradient descent'; 
            pred_par.GRAD_CLIPPING = true; 
            pred_par.grad_threshold = 100.0;  
			pred_par.Winit_std_dev = 0.02;

        case 'DNI'
            % Decoupled neural interfaces
            pred_par.NORMALIZE_DATA = true;
            pred_par.update_meth = 'stochastic gradient descent'; 
            pred_par.GRAD_CLIPPING = true; 
            pred_par.grad_threshold = 100.0;  
			pred_par.Winit_std_dev = 0.02;
            
            % parameters for estimating the matrix A such that c = x_tilde*A, where c is the credit assignment vector and x_tilde = [x, Ytrue(:,t).', 1]
            pred_par.learn_rate_A = 0.002;
            pred_par.GRAD_CLIPPING_A = false;
            pred_par.update_meth_A = 'stochastic gradient descent';

        case 'fixed W'
            % RNN model with a frozen layer
            pred_par.NORMALIZE_DATA = true;
            pred_par.update_meth = 'stochastic gradient descent'; 
            pred_par.GRAD_CLIPPING = true; 
            pred_par.grad_threshold = 100.0;
			pred_par.Winit_std_dev = 0.02;     

        case 'SVR'

            pred_par.nb_runs = 1;  % not a stochastic method
            pred_par.NORMALIZE_DATA = true;
            pred_par.tmax_training = 160;  % MR data (ETH Zurich - although not in the CMIG paper)

            % pred_par.tmax_training = 180; % markers 3.33 Hz (CPMB paper)
            % pred_par.tmax_training = 540; % markers 10 Hz (CPMB paper)
            % pred_par.tmax_training = 1620; % markers 30 Hz (CPMB paper)             

    end
    
    % Additional optimizer settings (if required)
    if isfield(pred_par, 'update_meth')
        switch(pred_par.update_meth)
            case 'stochastic gradient descent'
                % No additional settings needed for standard SGD.
            case 'ADAM'
                % Parameters for ADAM optimizer.
                pred_par.ADAM_beta1 = 0.9;
                pred_par.ADAM_beta2 = 0.999;
                pred_par.ADAM_eps = 10^-8;
        end    
    end
    
end