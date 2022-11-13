function [eval_results] = pred_eval(beh_par, path_par, pred_par, disp_par, Ypred, avg_pred_time)
% Evaluation of the prediction performance of the trained RNN.
% This function :
%   - computes the root-mean-square error (RMSE), the normalized RMSE, the associated 95% mean confidence intervals, the number of numerical errors,
%       and the prediction time, which are stored in the structure eval_results.
%   - displays and saves plots comparing the predicted and ground-truth signals
%   - displays and saves plots of the instantaneous error function.
% 
% To do to improve the code: 
% - use loops over the keys of eval_results / redefine the structures at the end to reduce the nb of lines of code 
%   (sth like: for metric in metrics: eval_results.metric = ...)
%
% Author : Pohl Michel
% Date : September 9, 2022
% Version : v2.0
% License : 3-clause BSD License


    eval_results = struct();
    eval_results.rms_err = zeros(pred_par.nb_runs, 1);
    eval_results.nrmse = zeros(pred_par.nb_runs, 1);
    % The following is used in the prediction of the position of markers:
    eval_results.jitter = zeros(pred_par.nb_runs, 1);
    eval_results.mean_err = zeros(pred_par.nb_runs, 1);
    eval_results.max_err = zeros(pred_par.nb_runs, 1);   

    % Loading the ground-truth signal
    load(path_par.time_series_data_filename, 'org_data');
    org_data = org_data(:,1:pred_par.tmax_pred);
    [data_dim, ~] = size(org_data);
       
    switch pred_par.data_type
        case 1 % prediction of the 3D position of markers 
            pred_par.dim_per_obj = 3; 
            pred_par.nb_obj = data_dim/3; % number of points - because we are predicting the 3D position of points.
        case 2 % prediction of other signals (for instance weights of the PCA of the DVF) 
            pred_par.dim_per_obj = 1;
            pred_par.nb_obj = data_dim; 
                % we can regard each signal as corresponding to one object
                % essentially it is for using consistent notation with what follows
    end

    pred_data = zeros(pred_par.nb_obj*pred_par.dim_per_obj, pred_par.tmax_pred); 
    loss_function_tab = zeros(pred_par.tmax_pred, pred_par.nb_runs);     

    for run_idx=1:pred_par.nb_runs

        if (pred_par.nb_runs>1)&&(mod(run_idx,10) == 1)
            fprintf('Evaluation for the %d-th test (random initialization) \n', run_idx);
        end

        delta = zeros(pred_par.nb_obj, pred_par.tmax_pred); % think about transposition later maybe
        pred_data(:, (pred_par.SHL+pred_par.horizon):pred_par.tmax_pred) = Ypred(:,:,run_idx);
            % pred_data(obj_idx + (crt_dim-1)*pred_par.dim_per_obj, t) is the predicted position of coordinate crt_dim of obj_idx at time t

        switch pred_par.data_type
            case 1 % prediction of the 3D position of markers 
                crdwise_delta = org_data - pred_data;
                for obj_idx = 1:pred_par.nb_obj
                    for crt_dim = 1:pred_par.dim_per_obj
                        delta(obj_idx, :) = delta(obj_idx, :) + crdwise_delta(obj_idx + (crt_dim-1)*pred_par.dim_per_obj, :).^2;
                    end
                end
                delta = sqrt(delta);
            case 2 % prediction of other signals (for instance weights of the PCA of the DVF)
                delta = org_data - pred_data;
        end

        % loss function
        loss_function_tab(:, run_idx) = sum(delta.^2, 1);
        loss_function_tab(1:pred_par.SHL+pred_par.horizon-1, :) = 0;

        delta_test = delta(:,pred_par.t_eval_start:pred_par.tmax_pred);
        eval_results.mean_err(run_idx) = (1/(pred_par.nb_obj*pred_par.nb_predictions))*sum(sum(delta_test));
        eval_results.max_err(run_idx) = max(max(delta_test));
        eval_results.rms_err(run_idx) = (1/sqrt(pred_par.nb_obj*pred_par.nb_predictions))*sqrt(sum(sum(delta_test.^2)));  

        % nRMSE calculation
        mu_true = mean(org_data(:,pred_par.t_eval_start:pred_par.tmax_pred), 2);
        nrmse_denominator_tab = mu_true*ones(1, pred_par.nb_predictions)-org_data(:, pred_par.t_eval_start:pred_par.tmax_pred);
        nrmse_denominator = sqrt(sum(sum(nrmse_denominator_tab.^2)));
        eval_results.nrmse(run_idx) = (1/nrmse_denominator)*sqrt(sum(sum(delta_test.^2)));

        % Jitter calculation
        switch pred_par.data_type
            case 1 % prediction of the 3D position of markers 
                pred_pts_pos = zeros(pred_par.nb_obj, pred_par.tmax_pred, 3);
                    % Note: here I use another array pred_pts_pos but I could also use the programming style when computing delta above
                pred_pts_pos(:, (pred_par.SHL+pred_par.horizon):pred_par.tmax_pred, 1) = Ypred(1:pred_par.nb_obj,               :,run_idx);             %x coordinate
                pred_pts_pos(:, (pred_par.SHL+pred_par.horizon):pred_par.tmax_pred, 2) = Ypred((pred_par.nb_obj+1):(2*pred_par.nb_obj),  :,run_idx);    %y coordinate 
                pred_pts_pos(:, (pred_par.SHL+pred_par.horizon):pred_par.tmax_pred, 3) = Ypred((2*pred_par.nb_obj+1):(3*pred_par.nb_obj),:,run_idx);    %z coordinate
                instant_jitter = sqrt(sum((pred_pts_pos(:, (pred_par.t_eval_start+1):pred_par.tmax_pred, :) ...
                                            - pred_pts_pos(:, pred_par.t_eval_start:(pred_par.tmax_pred-1), :)).^2,3));
            case 2 % prediction of other signals (for instance weights of the PCA of the DVF)
                instant_jitter = pred_data(:,(pred_par.t_eval_start+1):pred_par.tmax_pred) ...
                                    - pred_data(:,pred_par.t_eval_start:(pred_par.tmax_pred-1));
        end
        eval_results.jitter(run_idx) = mean(mean(instant_jitter));

        SAVE_ONLY = (run_idx ~=1);
        % only the plots from the first test remain displayed on the screen.

        if (beh_par.SAVE_PREDICTION_PLOT)&&(run_idx <= disp_par.nb_pred_runs_saved)

            % Plotting / saving the predicted signals corresponding to run_idx
            plot_pred_signal( pred_data, org_data, pred_par, path_par, disp_par, SAVE_ONLY)

            % Plotting / saving the loss function corresponding to run_idx
            title_str = sprintf('Prediction loss function (run %d)', run_idx);
            filename_suffix = sprintf('%s pred loss function %s %d-th run', path_par.time_series_dir, sprintf_pred_param(pred_par), run_idx);
            plot_pred_error( loss_function_tab(:, run_idx), disp_par, pred_par, path_par, filename_suffix, title_str, SAVE_ONLY );
        end
  
    end
    
    % counts the number of times when the prediction fails due to a numerical error (typically gradient explosion) 
    num_error_idx_vec = any(isnan(eval_results.rms_err),2);
    eval_results.nb_xplosion = sum(num_error_idx_vec); 
    eval_results.nb_correct_runs = pred_par.nb_runs - eval_results.nb_xplosion;
    loss_function_tab(:, num_error_idx_vec) = [];
    
    % plot of the mean error function
    if ismember(pred_par.pred_meth, {'RTRL', 'UORO', 'SnAp-1', 'DNI', 'RTRL v2', 'fixed W'})
        meanloss = mean(loss_function_tab, 2);
        if (beh_par.SAVE_PREDICTION_PLOT)
            SAVE_ONLY = false;     
            filename_suffix = sprintf('%s pred mean loss function %s', path_par.time_series_dir, sprintf_pred_param(pred_par));
            title_str = sprintf('Prediction mean loss function');
            plot_pred_error(meanloss, disp_par, pred_par, path_par, filename_suffix, title_str, SAVE_ONLY)
        end
    end

    % removing the NaN rows (numerical error) in the evaluation process
    pred_time_temp_mat = avg_pred_time;
    pred_time_temp_mat(num_error_idx_vec) = [];
    pred_rms_err_temp_mat = eval_results.rms_err;
    pred_rms_err_temp_mat(num_error_idx_vec) = [];  
    pred_nrmse_temp_mat = eval_results.nrmse;
    pred_nrmse_temp_mat(num_error_idx_vec) = [];  
    
    % useful when predicting the position of external markers
    pred_mean_err_temp_mat = eval_results.mean_err;
    pred_mean_err_temp_mat(num_error_idx_vec) = [];
    pred_max_err_temp_mat = eval_results.max_err;
    pred_max_err_temp_mat(num_error_idx_vec) = [];  
    pred_jitter_temp_mat = eval_results.jitter;
    pred_jitter_temp_mat(num_error_idx_vec) = [];       
    
    % calculating evaluation statistics  
    eval_results.mean_pt_pos_pred_time = mean(pred_time_temp_mat); 
    eval_results.mean_rms_err = mean(pred_rms_err_temp_mat);
    eval_results.confidence_half_range_rms_err = 1.96*std(pred_rms_err_temp_mat)/sqrt(eval_results.nb_correct_runs); 
    eval_results.mean_nrmse = mean(pred_nrmse_temp_mat);    
    eval_results.confidence_half_range_nrmse = 1.96*std(pred_nrmse_temp_mat)/sqrt(eval_results.nb_correct_runs); 

    % useful when predicting the position of external markers    
    eval_results.mean_mean_err = mean(pred_mean_err_temp_mat);
    eval_results.confidence_half_range_mean_err = 1.96*std(pred_mean_err_temp_mat)/sqrt(eval_results.nb_correct_runs);
    eval_results.mean_max_err = mean(pred_max_err_temp_mat);
    eval_results.confidence_half_range_max_err = 1.96*std(pred_max_err_temp_mat)/sqrt(eval_results.nb_correct_runs);    
    eval_results.mean_jitter = mean(pred_jitter_temp_mat);
    eval_results.confidence_half_range_jitter = 1.96*std(pred_jitter_temp_mat)/sqrt(eval_results.nb_correct_runs);

end
