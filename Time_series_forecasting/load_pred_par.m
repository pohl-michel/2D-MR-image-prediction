function [pred_par] = load_pred_par(path_par, pred_meth)
% Load the parameters concerning prediction,
% which are initially stored in the file named path_par.pred_par_filename.
% RTRL -> version in Haykin's book / RTRL v2 -> "standard" version as described for instance in Tallec's paper about UORO
% 
% Author : Pohl Michel
% Date : September 27th, 2021
% Version : v1.1
% License : 3-clause BSD License

    path_par.pred_par_filename = sprintf('%s\\%s', path_par.input_seq_dir, path_par.pred_par_filename_suffix);

    opts = detectImportOptions(path_par.pred_par_filename);
    opts = setvartype(opts,'double');
    opts.DataRange = '2:2';
    pred_par = table2struct(readtable(path_par.pred_par_filename, opts));
    
    pred_par.GPU_COMPUTING = false;  
        % false is the default value - set to true manually inside switch case statement below
        % increases computation speed with RTRL (and RTRL v2)
        % NVidia GPU processor and Matlab's parallel computing toolbox is required
        % also implemented for UORO but not for SnAp-1 and DNI   
    pred_par.PARALLEL_COMPUTING = true; 
        % parallel computing can cause execution to stop with test data if not enough memory on the GPU and prevents debugging
        % another option would be to set the number of workers within the parfor loop: https://www.mathworks.com/help/parallel-computing/parfor.html  

    switch nargin
        case 1 % Choice of the prediction method by hand (most cases)
            pred_par.pred_meth = 'SnAp-1';
        case 2 % prediction method specified in image_prediction_main.m if beh_par.OPTIMIZE_NB_PCA_CP = true 
            pred_par.pred_meth = pred_meth;
    end
    
    switch(pred_par.pred_meth)

        case 'multivariate linear regression'

            pred_par.nb_runs = 1; % because it is not a stochastic method
            pred_par.NORMALIZE_DATA = false;
            pred_par.tmax_training = 160; % MR data (ETH Zurich - CMIG paper)

            % pred_par.tmax_training = 180; % markers 3.33 Hz (CPMB paper)
            % pred_par.tmax_training = 540; % markers 10 Hz (CPMB paper)
            % pred_par.tmax_training = 1620; % markers 30 Hz (CPMB paper)

        case 'RTRL'

            pred_par.NORMALIZE_DATA = true;
            pred_par.update_meth = 'stochastic gradient descent';
            pred_par.GRAD_CLIPPING = true;
            pred_par.grad_threshold = 100.0;    
            pred_par.Winit_std_dev = 0.02;
            pred_par.GPU_COMPUTING = true;

        case 'RTRL v2'

            pred_par.NORMALIZE_DATA = true;
            pred_par.update_meth = 'stochastic gradient descent'; 
            pred_par.GRAD_CLIPPING = true; 
            pred_par.grad_threshold = 100.0;  
			pred_par.Winit_std_dev = 0.02;
            pred_par.GPU_COMPUTING = true;
            pred_par.PARALLEL_COMPUTING = false; 

        case 'no prediction'

            pred_par.nb_runs = 1;
            pred_par.NORMALIZE_DATA = false;
            pred_par.SHL = 1; % The lastest acquired value is used instead of the predicted value

        case 'LMS' %multivariate least mean squares

            pred_par.nb_runs = 1; % not a stochastic method
            pred_par.NORMALIZE_DATA = true;    
            pred_par.update_meth = 'stochastic gradient descent';
            pred_par.GRAD_CLIPPING = true;
            pred_par.grad_threshold = 100.0;

        case 'UORO'

            pred_par.NORMALIZE_DATA = true;
            pred_par.eps_tgt_fwd_prp = 0.0000001;
            pred_par.eps_normalizers = 0.0000001;
            pred_par.update_meth = 'stochastic gradient descent';
            pred_par.GRAD_CLIPPING = true;
			pred_par.grad_threshold = 100.0;  
			pred_par.Winit_std_dev = 0.02;	

        case 'univariate linear regression'

            pred_par.nb_runs = 1; % because it is not a stochastic method
            pred_par.NORMALIZE_DATA = false;
            pred_par.tmax_training = 160;   

            % pred_par.tmax_training = 180; % markers 3.33 Hz (CPMB paper)
            % pred_par.tmax_training = 540; % markers 10 Hz (CPMB paper)
            % pred_par.tmax_training = 1620; % markers 30 Hz (CPMB paper)

		case 'SnAp-1'

            pred_par.NORMALIZE_DATA = true;
            pred_par.update_meth = 'stochastic gradient descent'; 
            pred_par.GRAD_CLIPPING = true; 
            pred_par.grad_threshold = 100.0;  
			pred_par.Winit_std_dev = 0.02;

        case 'DNI'

            pred_par.NORMALIZE_DATA = true;
            pred_par.update_meth = 'stochastic gradient descent'; 
            pred_par.GRAD_CLIPPING = true; 
            pred_par.grad_threshold = 100.0;  
			pred_par.Winit_std_dev = 0.02;
            
            % parameters for finding the matrix A such that c = x_tilde*A where c is the credit assignment vector and x_tilde = [x, Ytrue(:,t).', 1]
            pred_par.learn_rate_A = 0.002;
            pred_par.GRAD_CLIPPING_A = false;
            pred_par.update_meth_A = 'stochastic gradient descent';

        case 'fixed W'

            pred_par.NORMALIZE_DATA = true;
            pred_par.update_meth = 'stochastic gradient descent'; 
            pred_par.GRAD_CLIPPING = true; 
            pred_par.grad_threshold = 100.0; % 2022/11/14 not very sure here since Wa and Wb are fixed...  
			pred_par.Winit_std_dev = 0.02;            

    end
    
    if isfield(pred_par, 'update_meth')
        switch(pred_par.update_meth)
            case 'stochastic gradient descent'
                % do nothing
            case 'ADAM'
                pred_par.ADAM_beta1 = 0.9;
                pred_par.ADAM_beta2 = 0.999;
                pred_par.ADAM_eps = 10^-8;
        end    
    end
    
end