function [eval_results] = pred_eval(beh_par, path_par, pred_par, disp_par, Ypred, avg_pred_time)
% Evaluation of the prediction performance of the trained RNN.
% This function :
%   - computes the root-mean-square error (RMSE), the normalized RMSE, the associated 95% mean confidence intervals, the number of numerical errors,
%       and the prediction time, which are stored in the structure eval_results.
%   - displays and saves plots comparing the predicted and ground-truth signals
%   - displays and saves plots of the instantaneous error function.
% 
% Author : Pohl Michel
% Date : September 27th, 2021
% Version : v1.0
% License : 3-clause BSD License


    pt_pos_pred_time = zeros(pred_par.nb_runs, 1);
        % time for predicting the position of the markers at each time step
    eval_results = struct();
    eval_results.rms_err = zeros(pred_par.nb_runs, 1);
    eval_results.nrmse = zeros(pred_par.nb_runs, 1);
    eval_results.jitter = zeros(pred_par.nb_runs, 1);

    % Loading the ground-truth signal
    load(path_par.time_series_data_filename, 'org_data');
    org_data = org_data(:,1:pred_par.tmax_pred);
    [data_dim, ~] = size(org_data);

    [~, M, ~] = size(Ypred);
    tmax = pred_par.tmax_pred;
    pred_data = zeros(data_dim, tmax); 
    loss_function_tab = zeros(tmax, pred_par.nb_runs);
    
    for run_idx=1:pred_par.nb_runs

        % RMSE computation
        pred_data(:, (tmax-M+1):tmax) = Ypred(:,:,run_idx);
        delta = org_data - pred_data;
        eval_results.rms_err(run_idx) = (1/sqrt(data_dim*pred_par.nb_predictions))*sqrt(sum(sum(delta(:, pred_par.t_eval_start:pred_par.tmax_pred).^2)));

        % nRMSE computation
        mu_true = mean(org_data(:,pred_par.t_eval_start:pred_par.tmax_pred), 2);
        nrmse_denominator = sqrt(sum(sum((mu_true*ones(1, pred_par.nb_predictions)-org_data(:, pred_par.t_eval_start:pred_par.tmax_pred)).^2)));
        eval_results.nrmse(run_idx) = (1/nrmse_denominator)*sqrt(sum(sum(delta(:, pred_par.t_eval_start:pred_par.tmax_pred).^2)));

        % One can insert a breakpoint here and look at pred_data and org_data
        if (beh_par.SAVE_PREDICTION_PLOT)&&(run_idx == 1) 
            
            plot_pred_signal( pred_data, org_data, pred_par, path_par, disp_par);

            % plotting the loss function as a function of time
            loss_function_tab(:, run_idx) = sum(delta.^2, 1);
            loss_function_tab(1:pred_par.tmax_training, run_idx) = 0;
            title_str = sprintf('Prediction loss function (run %d)', run_idx);
            filename_suffix = sprintf('%s pred loss function %s %d-th run', path_par.time_series_dir, sprintf_pred_param(pred_par), run_idx);
            plot_pred_error( loss_function_tab(:, run_idx), disp_par, pred_par, path_par, filename_suffix, title_str, true );     
            
        end

        % Calculation time
        pt_pos_pred_time(run_idx) = avg_pred_time(run_idx);
    
    end
    
    % counts the number of times when the prediction fails due to a numerical error (typically gradient explosion) 
    num_error_idx_vec = any(isnan(eval_results.rms_err),2);
    eval_results.nb_xplosion = sum(num_error_idx_vec); 
    eval_results.nb_correct_runs = pred_par.nb_runs - eval_results.nb_xplosion;
    
    % plot of the mean error function
 
    meanloss = mean(loss_function_tab, 2);

    if (beh_par.SAVE_PREDICTION_PLOT)
        SAVE_ONLY = false;     
        filename_suffix = sprintf('%s pred mean loss function %s', path_par.time_series_dir, sprintf_pred_param(pred_par));
        title_str = sprintf('Prediction mean loss function');
        plot_pred_error( meanloss, disp_par, pred_par, path_par, filename_suffix, title_str, SAVE_ONLY )
    end

    % removing the NaN rows (numerical error) in the evaluation process
    pred_time_temp_mat = pt_pos_pred_time;
    pred_time_temp_mat(num_error_idx_vec) = [];
    pred_rms_err_temp_mat = eval_results.rms_err;
    pred_rms_err_temp_mat(num_error_idx_vec) = [];  
    pred_nrmse_temp_mat = eval_results.nrmse;
    pred_nrmse_temp_mat(num_error_idx_vec) = [];      
    
    % calculating evaluation statistics  
    eval_results.mean_pt_pos_pred_time = mean(pred_time_temp_mat); 
    eval_results.mean_rms_err = mean(pred_rms_err_temp_mat);
    eval_results.confidence_half_range_rms_err = 1.96*std(pred_rms_err_temp_mat)/sqrt(eval_results.nb_correct_runs); 
    eval_results.mean_nrmse = mean(pred_nrmse_temp_mat);    
    eval_results.confidence_half_range_nrmse = 1.96*std(pred_nrmse_temp_mat)/sqrt(eval_results.nb_correct_runs); 
    
end

