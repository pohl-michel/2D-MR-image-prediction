function [optim, best_par] = train_eval_predictor_mult_param(hppars, pred_par, path_par, disp_par, beh_par)
% Performs the training and evaluation of the prediction method selected in load_pred_par
% using grid search with the hyperparameter grid selected in load_hyperpar_cv, for multiple horizon values
% Parallel processing is used to accelerate the speed of grid search.
%
% Rk: one could reduce the number of lines of code by looping over the keys of the different structures (e.g., best_par)
% Rk: I'm checking if the method is the population transformer to load the correct config file, but there could be a better code structure,
% where the prediction parameters are loaded from scratch inside or just before the for loop around perform_cv_once. 
%
% Author : Pohl Michel
% Date : August 27, 2022
% Version : v1.1
% License : 3-clause BSD License


    fprintf('Optimization of the parameters of the prediction algorithm \n\n');
        
    beh_par.SAVE_PREDICTION_PLOT = false;
    beh_par.SAVE_PRED_RESULTS = false;  
    
        
    %% EVALUATION ON ALL THE PARAMETERS
 
    % we need the amount of data for training, for cross validation and for testing
    pred_par.t_eval_start = 1 + pred_par.tmax_training;
    pred_par.nb_predictions = pred_par.tmax_cv - pred_par.t_eval_start + 1;
    pred_par.tmax_pred = pred_par.tmax_cv;
    pred_par.nb_runs = hppars.nb_runs_cv;    
    
    size_other_hyppr_tab = [];
    for hppar_idx = 1:hppars.nb_additional_params
        size_other_hyppr_tab = [size_other_hyppr_tab, hppars.other(hppar_idx).nb_val];
    end
    if (hppars.nb_additional_params == 1)
        size_other_hyppr_tab = [size_other_hyppr_tab, 1];
    end
    
    optim = struct();
    for hrz_idx = 1:hppars.nb_hrz_val
        optim(hrz_idx).rms_error_tab = zeros(size_other_hyppr_tab);
        optim(hrz_idx).rms_error_confidence_half_range_tab = zeros(size_other_hyppr_tab);
        optim(hrz_idx).nb_xplosion_tab = zeros(size_other_hyppr_tab);
        optim(hrz_idx).nrmse_tab = zeros(size_other_hyppr_tab);
        optim(hrz_idx).pred_time_tab = zeros(size_other_hyppr_tab);           
    end
    
    if pred_par.PARALLEL_COMPUTING
        parfor hrz_idx = 1:hppars.nb_hrz_val        
            optim_hrz_idx = optim(hrz_idx);
            optim(hrz_idx) = perform_grid_train_and_eval(optim_hrz_idx, hrz_idx, size_other_hyppr_tab, pred_par, hppars, path_par, beh_par, disp_par); 
        end
    else
        for hrz_idx = 1:hppars.nb_hrz_val
            optim(hrz_idx) = perform_grid_train_and_eval(optim(hrz_idx), hrz_idx, size_other_hyppr_tab, pred_par, hppars, path_par, beh_par, disp_par); 
        end
    end
    
    %% SEARCH FOR THE BEST PARAMETERS
     
    fprintf('Eval on test data');

    beh_par.SAVE_PREDICTION_PLOT = false;
        
    best_par.rms_cv_error_tab = zeros(hppars.nb_hrz_val, 1);
    best_par.nb_expl_cv_tab = zeros(hppars.nb_hrz_val, 1);
    
    best_par.other_hyppar_tab = zeros(hppars.nb_hrz_val, hppars.nb_additional_params);
    
    best_par.rms_err_test_set_tab = zeros(hppars.nb_hrz_val, 1);
    best_par.cf_half_range_rms_err_test_set_tab = zeros(hppars.nb_hrz_val, 1);
    best_par.nrmse_test_set_tab = zeros(hppars.nb_hrz_val, 1);
    best_par.cf_half_range_nrmse_test_set_tab = zeros(hppars.nb_hrz_val, 1);    
    best_par.mean_pt_pos_pred_time_tab = zeros(hppars.nb_hrz_val, 1);
    best_par.nb_xplosions_tab = zeros(hppars.nb_hrz_val, 1);

    % Useful when predicting the position of objects:
    best_par.mean_err_test_set_tab = zeros(hppars.nb_hrz_val, 1);
    best_par.cf_half_range_mean_err_test_set_tab = zeros(hppars.nb_hrz_val, 1);    
    best_par.max_err_test_set_tab = zeros(hppars.nb_hrz_val, 1);
    best_par.cf_half_range_max_err_test_set_tab = zeros(hppars.nb_hrz_val, 1);
    best_par.jitter_test_set_tab = zeros(hppars.nb_hrz_val, 1);
    best_par.cf_half_range_jitter_test_set_tab = zeros(hppars.nb_hrz_val, 1);

    pred_par_cell = cell(hppars.nb_hrz_val, 1);
    	% eval_results_best_par is a cell which contains structures
    pred_meth = pred_par.pred_meth; % necessary otherwise pred_par_cell{hrz_idx} in the parfor loop below may have a different pred_meth field (image prediction)
    
    for hrz_idx = 1:hppars.nb_hrz_val
        
        % error_aux_tab contains the RMSE / nRMSE values of the prediction algorithm for the considered hyperparameters and the current horizon value
        switch(pred_par.cross_val_metric)
            case 'nRMSE'
                error_aux_tab = optim(hrz_idx).nrmse_tab;
            case 'RMSE'
                error_aux_tab = optim(hrz_idx).rms_error_tab;
        end        
        
        nb_xplosion_tab_temp = optim(hrz_idx).nb_xplosion_tab;
        
        min_expl = my_min(nb_xplosion_tab_temp);
        nb_xplosion_tab_temp = nb_xplosion_tab_temp - min_expl*ones(size(nb_xplosion_tab_temp));
        expl_idx_tab = nb_xplosion_tab_temp~=0;
        error_aux_tab(expl_idx_tab) = Inf;
            % parameters for which there are numerical errors are not taken into account
        best_par.rms_cv_error_tab(hrz_idx) = my_min(error_aux_tab);
        lin_idx_min = find(error_aux_tab == best_par.rms_cv_error_tab(hrz_idx));
        
        idx_vec = my_ind2sub(size(error_aux_tab), lin_idx_min(1));
        
        best_par.nb_expl_cv_tab(hrz_idx) = min_expl;
        for hppar_idx = 1:hppars.nb_additional_params
            best_par.other_hyppar_tab(hrz_idx, hppar_idx) = hppars.other(hppar_idx).val(idx_vec(hppar_idx));
        end

        % evaluation on the test set with the optimal parameters
        pred_par_cell{hrz_idx} = load_pred_par(path_par, pred_meth, hppars.horizon_tab(hrz_idx)); %we find tmax_pred - hrz_idx only has an effect for population transformer
            % in the case of linear regression, the value of pred_par.tmax_training is already modified inside the function load_pred_par.m
        pred_par_cell{hrz_idx}.t_eval_start = 1 + pred_par_cell{hrz_idx}.tmax_cv; % because evaluation on the test set
        pred_par_cell{hrz_idx}.nb_predictions = pred_par_cell{hrz_idx}.tmax_pred - pred_par_cell{hrz_idx}.t_eval_start + 1;
        pred_par_cell{hrz_idx}.nb_runs = hppars.nb_runs_eval_test;    
        
        pred_par_cell{hrz_idx}.horizon = hppars.horizon_tab(hrz_idx);
        for hppar_idx = 1:hppars.nb_additional_params
            pred_par_cell{hrz_idx}.(hppars.other(hppar_idx).name) = best_par.other_hyppar_tab(hrz_idx, hppar_idx);
        end
 
    end
    
    if pred_par.PARALLEL_COMPUTING
        parfor hrz_idx = 1:hppars.nb_hrz_val   
            fprintf('\n \n');
            fprintf('Eval on test data for h = %d \n', hppars.horizon_tab(hrz_idx))         
            [Ypred, avg_pred_time, ~] = train_and_predict(path_par, pred_par_cell{hrz_idx}, beh_par);
            eval_results_best_par(hrz_idx) = pred_eval(beh_par, path_par, pred_par_cell{hrz_idx}, disp_par, Ypred, avg_pred_time); % eval_results_best_par is a structure array  
        end
    else
        for hrz_idx = 1:hppars.nb_hrz_val   
            fprintf('\n \n');
            fprintf('Eval on test data for h = %d \n', hppars.horizon_tab(hrz_idx))         
            [Ypred, avg_pred_time, ~] = train_and_predict(path_par, pred_par_cell{hrz_idx}, beh_par);
            eval_results_best_par(hrz_idx) = pred_eval(beh_par, path_par, pred_par_cell{hrz_idx}, disp_par, Ypred, avg_pred_time); % eval_results_best_par is a structure array  
        end
    end
        
    for hrz_idx = 1:hppars.nb_hrz_val    
        
        best_par.rms_err_test_set_tab(hrz_idx) = eval_results_best_par(hrz_idx).mean_rms_err;
        best_par.cf_half_range_rms_err_test_set_tab(hrz_idx) = eval_results_best_par(hrz_idx).confidence_half_range_rms_err;
        best_par.nrmse_test_set_tab(hrz_idx) = eval_results_best_par(hrz_idx).mean_nrmse;
        best_par.cf_half_range_nrmse_test_set_tab(hrz_idx) = eval_results_best_par(hrz_idx).confidence_half_range_nrmse;
        best_par.mean_pt_pos_pred_time_tab(hrz_idx) = eval_results_best_par(hrz_idx).mean_pt_pos_pred_time;
        best_par.nb_xplosions_tab(hrz_idx) = eval_results_best_par(hrz_idx).nb_xplosion;

        % For the prediction of 3D objects
        best_par.mean_err_test_set_tab(hrz_idx) = eval_results_best_par(hrz_idx).mean_mean_err;
        best_par.cf_half_range_mean_err_test_set_tab(hrz_idx) = eval_results_best_par(hrz_idx).confidence_half_range_mean_err;
        best_par.max_err_test_set_tab(hrz_idx) = eval_results_best_par(hrz_idx).mean_max_err;
        best_par.cf_half_range_max_err_test_set_tab(hrz_idx) = eval_results_best_par(hrz_idx).confidence_half_range_max_err;
        best_par.jitter_test_set_tab(hrz_idx) = eval_results_best_par(hrz_idx).mean_jitter;
        best_par.cf_half_range_jitter_test_set_tab(hrz_idx) = eval_results_best_par(hrz_idx).confidence_half_range_jitter;
        
    end

end


function optim_hrz_idx = perform_grid_train_and_eval(optim_hrz_idx, hrz_idx, size_other_hyppr_tab, pred_par, hppars, path_par, beh_par, disp_par)
% Perform training and evaluation using grid search with the grid defined in hppars for a specific horizon, and storing the results in optim_hrz_idx

    pred_par_h = get_pred_par_h(pred_par, hppars, hrz_idx, path_par);
    v_h = ones(1, hppars.nb_additional_params);
    nb_calc_crt = 1;     
    
    ready = false;
    optim_hrz_idx = perform_cv_once( v_h, optim_hrz_idx, nb_calc_crt, hppars, pred_par_h, path_par, beh_par, disp_par);
    while ~ready
        % Update the index vector:
        ready = true;
        for k = hppars.nb_additional_params:-1:1
            v_h(k) = v_h(k) + 1;
            if v_h(k) <= size_other_hyppr_tab(k)

                ready = false;
                nb_calc_crt = nb_calc_crt +1;
                optim_hrz_idx = perform_cv_once( v_h, optim_hrz_idx, nb_calc_crt, hppars, pred_par_h, path_par, beh_par, disp_par);                            
                
                break;  % v(k) increased successfully, leave the "for k" loop

            end
            v_h(k) = 1;  % v(k) reached the limit, reset it and iterate v(k-1)
        end
    end   

end