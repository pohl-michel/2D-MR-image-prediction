function eval_im_pred_best_par(eval_results, best_pred_par_struct, best_pca_cp_tab, beh_par, disp_par, OF_par, im_par, path_par, br_model_par, warp_par)
% Evaluates the accuracy of image prediction of the test set
% using the optimal hyperparameters selected beforehand by "select_nb_pca_cp"
%
% Rk: parfor may not work well as "write_im_pred_log_file" attempts to write txt files, in that case, replace "parfor" with "for"
% If that happens, put write_im_pred_log_file inside its own for loop (after the parfor loop, not inside) and use cell arrays for its arguments 
%
% Author : Pohl Michel
% Date : Sept 18th, 2022
% Version : v2.0
% License : 3-clause BSD License


    beh_par.SAVE_WARPED_IM = false; % because display takes time and this function needs to run fast 
    beh_par.SAVE_PREDICTION_PLOT = false; % idem
    
    pred_par = load_pred_par(path_par); %we need to know tmax_pred corresponding to the test set (in particular)
        % in the case of linear regression, the value of pred_par.tmax_training is already modified inside the function "load_pred_par"
    pred_par.t_eval_start = 1 + pred_par.tmax_cv; % evaluation on the test set
    pred_par.nb_predictions = pred_par.tmax_pred - pred_par.t_eval_start + 1;
    hppars = load_hyperpar_cv_info(pred_par); 
    pred_par.nb_runs = hppars.nb_runs_eval_test;
    nb_runs_for_cc_eval = warp_par.nb_runs_for_cc_eval;

    log_txt_filename_suffix_temp = path_par.log_txt_filename(1:end-4); % we suppress the extension .txt

    dvf_type = 'initial DVF'; % warping with the initial optical flow
    my_empty_struct = struct();
    eval_results = eval_of_warp_corr(dvf_type, im_par, OF_par, path_par, warp_par, pred_par, br_model_par, disp_par, beh_par, eval_results, my_empty_struct);                   

    beh_par.EVAL_INIT_OF_WARP = true;            
    beh_par.IM_PREDICTION = true;
    beh_par.EVAL_PCA_RECONSTRUCT = true;  

    eval_results_cell = cell(1, hppars.nb_hrz_val);

    % Computation of PCA inside a for loop otherwise different threads may attempt to write the same file, leading to a "corrupt file" error 
    for hrz_idx = 1:hppars.nb_hrz_val

        pred_par.horizon = hppars.horizon_tab(hrz_idx);
        br_model_par_h.nb_pca_cp = best_pca_cp_tab(hrz_idx);

        for hppar_idx = 1:hppars.nb_additional_params
            pred_par.(hppars.other(hppar_idx).name) = best_pred_par_struct(br_model_par_h.nb_pca_cp).other_hyppar_tab(hrz_idx, hppar_idx);
        end                

        [W, F, Xmean, eval_results_cell{1, hrz_idx}] = compute_PCA_of_DVF(beh_par, disp_par, OF_par, im_par, path_par, pred_par, br_model_par_h, eval_results); 

    end

    parfor hrz_idx = 1:hppars.nb_hrz_val

        pred_par_h = pred_par;
        br_model_par_h = br_model_par;
        path_par_h = path_par;
        warp_par_h = warp_par;
        eval_results_h = eval_results_cell{1, hrz_idx};

        pred_par_h.horizon = hppars.horizon_tab(hrz_idx);
        br_model_par_h.nb_pca_cp_max = br_model_par.nb_pca_cp; % used in write_im_pred_log_file below
        br_model_par_h.nb_pca_cp = best_pca_cp_tab(hrz_idx);

        for hppar_idx = 1:hppars.nb_additional_params
            pred_par_h.(hppars.other(hppar_idx).name) = best_pred_par_struct(br_model_par_h.nb_pca_cp).other_hyppar_tab(hrz_idx, hppar_idx);
        end                

        dvf_type = 'DVF from PCA'; % warping with the signal reconstructed from PCA on the test set
        my_empty_struct = struct();
        eval_results_h = eval_of_warp_corr(dvf_type, im_par, OF_par, path_par_h, warp_par_h, pred_par_h, br_model_par_h, disp_par, beh_par, eval_results_h, my_empty_struct); 

        path_par_h.time_series_data_filename = write_PCAweights_mat_filename(OF_par, path_par, br_model_par_h);
        [Ypred, avg_pred_time, ~] = train_and_predict(path_par_h, pred_par_h, beh_par, br_model_par_h);
        time_signal_pred_results = pred_eval(beh_par, path_par_h, pred_par_h, disp_par, Ypred, avg_pred_time);       

        dvf_type = 'predicted DVF'; % warping with the predicted optical flow on the test set
        warp_par_h.nb_runs_for_cc_eval = min(nb_runs_for_cc_eval, time_signal_pred_results.nb_correct_runs);
        time_signal_pred_results.nb_correct_runs = warp_par_h.nb_runs_for_cc_eval;            
        eval_results_h = eval_of_warp_corr(dvf_type, im_par, OF_par, path_par_h, warp_par_h, pred_par_h, br_model_par_h, disp_par, beh_par, eval_results_h, time_signal_pred_results);

        path_par_h.log_txt_filename = sprintf('%s %s hrz=%d.txt', log_txt_filename_suffix_temp, pred_par.pred_meth, pred_par_h.horizon);
        write_im_pred_log_file(path_par_h, beh_par, im_par, OF_par, hppars, pred_par_h, br_model_par_h, warp_par_h, eval_results_h);

    end

end