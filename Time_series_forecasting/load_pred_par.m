function [pred_par] = load_pred_par(path_par, pred_meth, horizon)
% LOAD_PRED_PAR Loads prediction parameters from a specified file and sets additional parameters for different prediction methods.
%
% This function reads the prediction parameters stored in a file and returns them in the `pred_par` structure. 
% Depending on the prediction method selected (or passed as an argument), additional settings like normalization,
% gradient clipping, GPU computing, and other algorithm-specific parameters are set.
% horizon used only for population transformer, as we need to load the shl from the trained model
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
%   - 'transformer' (transformer encoder with final 1-layer feedforward network)
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
            pred_par.pred_meth = 'population_transformer';
            horizon = pred_par.horizon;
        case 2 % Prediction method specified in image_prediction_main.m (if OPTIMIZE_NB_PCA_CP == true) or sigpred_hyperparameter_optimization_main.m
            pred_par.pred_meth = pred_meth;
            horizon = pred_par.horizon;
        case 3    
            pred_par.pred_meth = pred_meth;
    end
    
    % Adjusting parameters based on the selected prediction method.
    switch(pred_par.pred_meth)
        case 'multivariate linear regression'
            pred_par.nb_runs = 1;  % not a stochastic method
            pred_par.NORMALIZE_DATA = false;
            % pred_par.tmax_training = 160;  % MR data (ETH Zurich - CMIG paper)
            pred_par.tmax_training = 303;  % MR data (Magdeburg - CMIG paper)

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
            % pred_par.tmax_training = 160; % cine-MR sequence prediction (CMIG paper)
            pred_par.tmax_training = 303;  % MR data (Magdeburg - CMIG paper)

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
            % pred_par.tmax_training = 160;  % MR data (ETH Zurich - although not in the CMIG paper)
            pred_par.tmax_training = 303;  % MR data (Magdeburg - CMIG paper)

            % pred_par.tmax_training = 180; % markers 3.33 Hz (CPMB paper)
            % pred_par.tmax_training = 540; % markers 10 Hz (CPMB paper)
            % pred_par.tmax_training = 1620; % markers 30 Hz (CPMB paper)     

        case 'transformer'

            pred_par.NORMALIZE_DATA = true;
            % pred_par.tmax_training = 160;  % MR data (ETH Zurich - CMIG paper)
            pred_par.tmax_training = 303;  % MR data (Magdeburg - CMIG paper) 

            % default parameters specific to the transformer
            pred_par.batch_size = 32;
            pred_par.num_epochs = 50;        
            pred_par.d_model = 16;  % embedding dimension - should be divisible by nhead
            pred_par.nhead = 2;
            pred_par.num_layers = 2;
            pred_par.dim_feedforward = 0;  % Setting that value to zero sets the encoder MLP hidden layer as proportional to d_model 
            pred_par.final_layer_dim = 0;  % Setting that value to zero sets the final layer dim to geometric avg. of input and output
            pred_par.dropout = 0.5;
            pred_par.learn_rate = 0.0001;  % comment that to load from the Excel file instead
            pred_par.GPU_COMPUTING = true;  % experimentally faster but can be toggled off
            pred_par.print_every = 25;  % print the loss value every "print_every" step
            % pred_par.nb_runs = 2;  % for testing - normally loaded from pred_par.xlsx file

            % Add Python module directory if not already in sys.path
            moduleDir = get_python_transformers_module_dir();
            if count(py.sys.path, moduleDir) == 0
                insert(py.sys.path, int32(0), moduleDir);
            end
        
            % Check and set KMP_DUPLICATE_LIB_OK only if not already set
            currentVal = getenv("KMP_DUPLICATE_LIB_OK");
            if ~strcmp(currentVal, "TRUE")
                setenv("KMP_DUPLICATE_LIB_OK", "TRUE");
            end

        case 'population_transformer' % horizon from excel file - the SHl is loaded from config.json file corresponding to model

            % We are going to use the scaler from the population model instead
            pred_par.NORMALIZE_DATA = false;

            % The model has already been trained, but we do testing in the same way as online methods: tmax_training + 1 is the 1st time index for prediction
            % pred_par.tmax_training = 160;  % MR data (ETH Zurich - CMIG paper)
            pred_par.tmax_training = 303;  % MR data (Magdeburg - CMIG paper) 

            % Add Python module directory if not already in sys.path
            moduleDir = get_python_transformers_module_dir();
            if count(py.sys.path, moduleDir) == 0
                insert(py.sys.path, int32(0), moduleDir);
            end
        
            % Check and set KMP_DUPLICATE_LIB_OK only if not already set
            currentVal = getenv("KMP_DUPLICATE_LIB_OK");
            if ~strcmp(currentVal, "TRUE")
                setenv("KMP_DUPLICATE_LIB_OK", "TRUE");
            end

            % Loading the config file - We assume that the config for the first transformer config is similar to the others (could add check in further improvements)
            % config_path = sprintf('%s/horizon_%d/transformer_h2_model1', path_par.temp_var_dir, horizon); % uncomment to specify by hand (if no date provided)
            config_path = get_most_recent_model_config(path_par.temp_var_dir, horizon);
            config_str = fileread(config_path);
            config = jsondecode(config_str);

            % Setting the SHL (and possibly the number of runs)            
            pred_par.SHL = config.config.seq_length;

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


function config_path = get_most_recent_model_config(base_dir, horizon)
% GET_MOST_RECENT_MODEL_CONFIG Find the most recent transformer model config file
%
% INPUTS:
% - base_dir: Base directory containing horizon folders
% - horizon: Horizon value (e.g., 2 for horizon_2 folder)
%
% OUTPUT:
% - config_path: Full path to the most recent config file

    % Construct the horizon folder path
    horizon_folder = fullfile(base_dir, sprintf('horizon_%d', horizon));
    
    % Check if folder exists
    if ~exist(horizon_folder, 'dir')
        error('Horizon folder does not exist: %s', horizon_folder);
    end
    
    % Get all config files matching the pattern
    pattern = sprintf('transformer_h%d_model*_config.json', horizon);
    files = dir(fullfile(horizon_folder, pattern));
    
    if isempty(files)
        error('No config files found in %s matching pattern %s', horizon_folder, pattern);
    end
    
    % Extract date and time from filenames and find the most recent
    most_recent_datetime = 0;
    most_recent_file = '';
    
    for i = 1:length(files)
        filename = files(i).name;
        
        % Extract date and time using regular expression
        % Pattern: transformer_h2_model1_YYYYMMDD_HHMMSS_config.json
        tokens = regexp(filename, 'transformer_h\d+_model\d+_(\d{8})_(\d{6})_config\.json', 'tokens');
        
        if ~isempty(tokens)
            date_str = tokens{1}{1}; % YYYYMMDD
            time_str = tokens{1}{2}; % HHMMSS
            
            % Convert to datetime for comparison
            datetime_str = [date_str, time_str]; % YYYYMMDDHHMMSS
            current_datetime = str2double(datetime_str);
            
            if current_datetime > most_recent_datetime
                most_recent_datetime = current_datetime;
                most_recent_file = filename;
            end
        end
    end
    
    if isempty(most_recent_file)
        error('No valid config files found with expected naming pattern');
    end
    
    % Return full path to the most recent config file
    config_path = fullfile(horizon_folder, most_recent_file);
    
    fprintf('Found most recent model: %s\n', most_recent_file);
end