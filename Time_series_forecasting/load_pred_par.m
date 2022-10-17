function [pred_par] = load_pred_par(path_par)
% Load the parameters concerning prediction,
% which are initially stored in the file named path_par.pred_par_filename.
%
% Author : Pohl Michel
% Date : September 27th, 2021
% Version : v1.0
% License : 3-clause BSD License

    path_par.pred_par_filename = sprintf('%s\\%s', path_par.input_seq_dir, path_par.pred_par_filename_suffix);

    opts = detectImportOptions(path_par.pred_par_filename);
    opts = setvartype(opts,'double');
    opts.DataRange = '2:2';
    pred_par = table2struct(readtable(path_par.pred_par_filename, opts));
    
    % Choice of the prediction method
    pred_par.pred_meth = 'DNI';
    
    switch(pred_par.pred_meth)

        case 'multivariate linear regression'

            pred_par.nb_runs = 1; % because it is not a stochastic method
            pred_par.NORMALIZE_DATA = false;
            pred_par.tmax_training = 160;

        case 'RTRL'

            pred_par.NORMALIZE_DATA = true;
            pred_par.update_meth = 'stochastic gradient descent';
            pred_par.GRAD_CLIPPING = true;
            pred_par.grad_threshold = 100.0;    
            pred_par.Winit_std_dev = 0.02;

        case 'no prediction'

            pred_par.nb_runs = 1;
            pred_par.NORMALIZE_DATA = false;
            pred_par.SHL = 1; % The lastest acquired value is used instead of the predicted value

        case 'LMS' %multivariate least mean squares

            pred_par.nb_runs = 1; % not a stochastic method
            pred_par.NORMALIZE_DATA = true;    
            pred_par.update_meth = 'stochastic gradient descent';
            pred_par.GRAD_CLIPPING = true;
            pred_par.grad_threshold = 2.0;

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