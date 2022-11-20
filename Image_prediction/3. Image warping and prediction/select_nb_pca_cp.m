function [eval_results, best_pred_par_struct, best_pca_cp_tab] = select_nb_pca_cp(beh_par, disp_par, OF_par, im_par, path_par, br_model_par, eval_results, warp_par)
% This function selects the optimal number of PCA components by predicting images and computing the error between the predicted and ground-truth images 
% on the cross-validation set for each combination of hyper-parameters and horizon value considered, as defined in the "load_hyperpar_cv_info" function.
% The maximum number of principal components considered is br_model_par.nb_pca_cp.
%
% Output:
% - eval_results: the evaluation results, which is updated by this function to include cross_cor_pca_optim_tab, the array of the maximum c.c. between the
% ground-truth and predicted images of the test set for each couple (hrz_idx, nb_pca_cp) considered.
% - best_pred_par_struct: the array containing the optimal prediction parameter for every possible number of pca components
% - best_pca_cp_tab: the array containing the optimal number of PCA component for each horizon value
%
% To do later : I can consider computing the error between the predicted and ground-truth deformation field instead, so that computation becomes faster.
% Hints: 
% - take inspiration from the "evalOF" function: https://github.com/pohl-michel/Lucas-Kanade-pyramidal-optical-flow-for-3D-image-sequences
% - use a switch statement over a field of beh_par.
%
% Author : Pohl Michel
% Date : Nov. 15th, 2022
% Version : v1.2
% License : 3-clause BSD License


    % Parameters concerning the prediction of the position of objects
    pred_par = load_pred_par(path_par);
    % Hyperparameters to optimize 
    hppars = load_hyperpar_cv_info(pred_par);

    % We redefine nb_pca_cp_max to be the maximum number of PCA components considered, and nb_pca_cp becomes the current value in the loop.
    nb_pca_cp_max = br_model_par.nb_pca_cp;
    br_model_par.nb_pca_cp_max = nb_pca_cp_max;
    nb_runs_for_cc_eval = warp_par.nb_runs_for_cc_eval;

    pca_perf_optim_tab = zeros(hppars.nb_hrz_val, nb_pca_cp_max);

    for nb_pca_cp=1:nb_pca_cp_max

        br_model_par.nb_pca_cp = nb_pca_cp;

        % Computation of PCA 
        % [I returned W & F for debugging purposes - the data is saved in the last lines of compute_PCA_of_DVF and loaded later when needed]
        [W, F, Xmean, ~] = compute_PCA_of_DVF(beh_par, disp_par, OF_par, im_par, path_par, pred_par, br_model_par, eval_results);

        pred_par.t_eval_start = 1 + pred_par.tmax_training;
        pred_par.nb_predictions = pred_par.tmax_cv - pred_par.t_eval_start + 1;
        pred_par.tmax_pred = pred_par.tmax_cv;
        pred_par.nb_runs = hppars.nb_runs_cv;               

        %% 1) We optimize the hyper-parameters for all values of h using the PCA weights of the cross-validation set
        % [I return optim for debugging purposes]
        [optim, best_par] = train_eval_predictor_mult_param(hppars, pred_par, path_par, disp_par, beh_par);

        best_pred_par_struct(nb_pca_cp) = best_par;

        beh_par.SAVE_PREDICTION_PLOT = false; % We do not need to plot and save figures here, as we want to do cross validation time fast.
        beh_par.SAVE_PRED_RESULTS = true; % so that eval_of_warp_cor can correctly load the predicted DVF

        %% 2) we calculate the c.c. between the predicted and ground-truth images of the c-v set to determine the best nb. of PCA coeffs for each horizon value h            
        parfor hrz_idx = 1:hppars.nb_hrz_val

            pred_par_h = pred_par;
            warp_par_h = warp_par;

            pred_par_h.horizon = hppars.horizon_tab(hrz_idx); % We use the best parameters for performing prediction on the cv set
            for hppar_idx = 1:hppars.nb_additional_params
                pred_par_h.(hppars.other(hppar_idx).name) = best_par.other_hyppar_tab(hrz_idx, hppar_idx);
            end

            [Ypred, avg_pred_time, ~] = train_and_predict(path_par, pred_par_h, beh_par);
            time_signal_pred_results = pred_eval(beh_par, path_par, pred_par_h, disp_par, Ypred, avg_pred_time);

            dvf_type = 'predicted DVF'; % warping with the predicted optical flow on the cv set
            warp_par_h.nb_runs_for_cc_eval = min(nb_runs_for_cc_eval, time_signal_pred_results.nb_correct_runs);
            time_signal_pred_results.nb_correct_runs = warp_par_h.nb_runs_for_cc_eval;

            % Rk: I can consider changing this line if I want to use the error from the comparison between the predicted and original optical flow directly
            eval_results_pca_optim = struct();

            if beh_par.REGISTRATION_ERROR_CV
                eval_results_pca_optim = compute_registration_error_cv_set(dvf_type, im_par, OF_par, path_par, warp_par_h, pred_par_h, br_model_par, ...
                                                                                        beh_par, eval_results_pca_optim, time_signal_pred_results)
                pca_perf_optim_tab(hrz_idx, nb_pca_cp) = eval_results_pca_optim.mean_of_nrmse;
            else
                eval_results_pca_optim = eval_of_warp_corr(dvf_type, im_par, OF_par, path_par, warp_par_h, pred_par_h, br_model_par, disp_par, ...
                                                                                        beh_par, eval_results_pca_optim, time_signal_pred_results);
                pca_perf_optim_tab(hrz_idx, nb_pca_cp) = eval_results_pca_optim.mean_corr_im_pred;
            end

        end

    end

    eval_results.cross_cor_pca_optim_tab = pca_perf_optim_tab;
    
    % We find the optimal number of PCA components
    if beh_par.REGISTRATION_ERROR_CV
        [~, best_pca_cp_tab] = min(pca_perf_optim_tab, [], 2); % minimum of the nrmse
    else    
        [~, best_pca_cp_tab] = max(pca_perf_optim_tab, [], 2); % minimum of the cross correlation
    end

end