function eval_results = compute_registration_error_cv_set(dvf_type, im_par, OF_par, path_par, warp_par, pred_par, br_model_par, beh_par, eval_results, time_signal_pred_results)
% This function:
%  - computes the predicted DVF u_t between t=1 and t for t in [t_eval_start, tmax_pred] (using load_computeOF_for_warp)
%  - computes statistics to evaluate the DVF prediction accuracy and stores them in eval_results
%
% Rk 1: prediction is random so we use different runs and take them into account when computing statistics
% Rk 2: this function is called when evaluating the prediction of the c.v. set (when selecting the nb. of PCA components)
% Rk 3: the statistics reflect the accuracy of the predicted DVF u_t without warping the original image (this helps save computation time in contrast to eval_of_warp_corr).
%
% Author : Pohl Michel
% Date : Nov 15th, 2022
% Version : v1.1
% License : 3-clause BSD License

    
    % chargement de l'image à t=1
    t_init = 1; crop_flag = false; filter_flag = false;
    I = load_crop_filter2D(t_init, crop_flag, filter_flag, 0, im_par, path_par.input_im_dir);   

    % I comment these lines but this function could be used to evaluate the initial DVF and the DVF from PCA as well, so I assume " dvf_type = 'predicted DVF' "
    % if ismember(dvf_type, {'initial DVF', 'DVF from PCA'}) || beh_par.NO_PRED_AT_ALL % evaluation of warping with the initial OF or evaluation of warping after PCA (no prediction)
    %      time_signal_pred_results.nb_correct_runs = 1;
    % end

    rmse_array = zeros(pred_par.nb_predictions, time_signal_pred_results.nb_correct_runs);   
    nrmse_array = zeros(pred_par.nb_predictions, time_signal_pred_results.nb_correct_runs);    
    
    for t=pred_par.t_eval_start:pred_par.tmax_pred

        % Loading/computing the (predicted or not) DVF u_t between t=1 and t
        [u_t, ~] = load_computeOF_for_warp(dvf_type, t, OF_par, path_par, im_par, br_model_par, pred_par, beh_par, warp_par);

        % chargement de l'image à t
        fprintf('Computing the registration error at t=%d (cross validation) \n', t);
        J = load_crop_filter2D(t, false, false, 0, im_par, path_par.input_im_dir);

        for run_idx = 1:time_signal_pred_results.nb_correct_runs
            J_translated = translate2DIm(J, u_t(:, :, :, run_idx));
            rmse_array(t - pred_par.t_eval_start + 1, run_idx) = my_rmse(J_translated, I, beh_par.EVALUATE_IN_ROI, im_par); 
            nrmse_array(t - pred_par.t_eval_start + 1, run_idx) = my_nrmse(J_translated, I, beh_par.EVALUATE_IN_ROI, im_par);
        end

    end

    eval_results.mean_of_rmse = mean(mean(rmse_array)); % maybe something like sqrt( (1/(nb_im-1)) * sum(rms_t.^2)) is better - I will do here as in the CPMB paper
    eval_results.confidence_half_range_of_rmse = 1.96*std(mean(rmse_array))/sqrt(time_signal_pred_results.nb_correct_runs);

    eval_results.mean_of_nrmse = mean(mean(nrmse_array));
    eval_results.confidence_half_range_of_nrmse = 1.96*std(mean(nrmse_array))/sqrt(time_signal_pred_results.nb_correct_runs);
    
end
