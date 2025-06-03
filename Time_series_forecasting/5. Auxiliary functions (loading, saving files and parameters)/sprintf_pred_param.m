function pred_param_str = sprintf_pred_param(pred_par)
% Returns a character string which contains information concerning the prediction parameters for saving and loading temporary variables
% 
% Author : Pohl Michel
% Date : September 27th, 2021
% Version : v1.0
% License : 3-clause BSD License

    if pred_par.NORMALIZE_DATA
        nrm_data_str = string('nrlzed data');
    else
        nrm_data_str = string('no nrmztion');
    end

    switch(pred_par.pred_meth)
        case {'multivariate linear regression', 'univariate linear regression'}
            pred_param_str = sprintf('k=%d h=%d tmax_train=%d %s', pred_par.SHL, pred_par.horizon, pred_par.tmax_training, nrm_data_str);
                % tmax_training is recorded because linear regression is an offline method
        case {'transformer', 'population_transformer'} % I am copying/simplifying for the population transformer otherwise the string is very long
            pred_param_str = sprintf('k=%d h=%d tmax_train=%d %s nepochs=%d embd_dim=%d nhead=%d nlayers=%d dim_ff=%d final_dim=%d dropout=%g lr=%g', ...
                pred_par.SHL, pred_par.horizon, pred_par.tmax_training, nrm_data_str, ...
                pred_par.num_epochs, pred_par.d_model, pred_par.nhead, pred_par.num_layers, pred_par.dim_feedforward, pred_par.final_layer_dim, pred_par.dropout, pred_par.learn_rate);
        case 'SVR'
            pred_param_str = sprintf('k=%d h=%d tmax_train=%d sigma=%d C=%d epsilon=%d %s', pred_par.SHL, pred_par.horizon, pred_par.tmax_training, ...
                pred_par.svr_kernel_scale, pred_par.svr_box_constraint, pred_par.svr_epsilon, nrm_data_str);        
        case {'RTRL', 'UORO', 'SnAp-1', 'DNI', 'RTRL v2', 'fixed W'} % RNN
            pred_param_str = sprintf('k=%d q=%d eta=%g sg=%g grd_tshld=%g h=%d %s', pred_par.SHL, pred_par.rnn_state_space_dim, ...
                pred_par.learn_rate, pred_par.Winit_std_dev, pred_par.grad_threshold, pred_par.horizon, nrm_data_str);
            % k = nb of time steps for performing one prediction
            % q = nb of neurons in the hidden layer
            % eta = learning rate
            % sg = standard deviation of the gaussian distribution of the initial weights values
            % grd_tshld = clipping value
        case 'no prediction'
            pred_param_str = sprintf('h=%d %s', pred_par.horizon, nrm_data_str);
        case 'LMS'
            pred_param_str = sprintf('k=%d h=%d eta=%g sg=%g %s', pred_par.SHL, pred_par.horizon, pred_par.learn_rate, pred_par.Winit_std_dev, nrm_data_str);    
    end
        
end