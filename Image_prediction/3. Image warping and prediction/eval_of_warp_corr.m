function eval_results = eval_of_warp_corr(dvf_type, im_par, OF_par, path_par, warp_par, pred_par, br_model_par, disp_par, beh_par, eval_results, time_signal_pred_results)
% This function performs several actions in the following order:
%  - Computes/loads the DVF (initial DVF, DVF from PCA, or predicted DVF) u_t between t=1 and t for t in [t_eval_start, tmax_pred] using load_computeOF_for_warp
%  - Computes I_warped, the image warped from I_init using u_t, and compares I_warped with the ground-truth image J at t 
%  - Computes statistics to evaluate the reconstruction/prediction accuracy and stores them in eval_results
%  - Saves the predicted/reconstructed images 
%  - Saves (instantaneous and mean) difference thermal images when we perform prediction (i.e., when u_t corresponds to a predicted DVF):
%
% Rk 1: prediction is random so we use different runs and take them into account when computing statistics and the difference images
% Rk 2: this function is called when evaluating the prediction of the test set (when evaluating the whole algorithm) or c.v. set (when selecting the nb. of PCA
% components)
% Rk 3: the code could be slightly improved by fixing a maximum for the error in the legend when saving the thermal instantanous error images 
%
% Author : Pohl Michel
% Date : Sept 18th, 2022
% Version : v1.1
% License : 3-clause BSD License


    % Loading the image à t=1
    t_init = 1; crop_flag = false; filter_flag = false; sigma_init = 'whatever';
    I_init = load_crop_filter2D(t_init, crop_flag, filter_flag, sigma_init, im_par, path_par.input_im_dir);
    
    % Initializing various arrays
    im_warp_calc_time_array = zeros(pred_par.nb_predictions, 1);
    OFcalc_time_array = zeros(pred_par.nb_predictions, 1);

    if ismember(dvf_type, {'initial DVF', 'DVF from PCA'}) || beh_par.NO_PRED_AT_ALL % evaluation of warping with the initial OF or evaluation of warping after PCA (no prediction)
        time_signal_pred_results.nb_correct_runs = 1;
    end

    im_correlation_array = zeros(pred_par.nb_predictions, time_signal_pred_results.nb_correct_runs);   
    mssim_array = zeros(pred_par.nb_predictions, time_signal_pred_results.nb_correct_runs);   
    nrmse_array = zeros(pred_par.nb_predictions, time_signal_pred_results.nb_correct_runs);    
    
    if strcmp(dvf_type, 'predicted DVF') && beh_par.SAVE_WARPED_IM % difference color image only if prediction
        diff_im_tensor = zeros(im_par.W, im_par.L, pred_par.nb_predictions);
        temp_runs_tensor = zeros(im_par.W, im_par.L, time_signal_pred_results.nb_correct_runs); % contains the images I(:,:,t) - Iwarped(:,:,t,run_idx) for t fixed
    end
    
    for t=pred_par.t_eval_start:pred_par.tmax_pred

        % Loading/computing the (predicted or not) DVF u_t between t=1 and t and computing I_warped, which results from warping I_init with u_t.
        if beh_par.NO_PRED_AT_ALL        
            I_warped = load_crop_filter2D(t-pred_par.horizon, crop_flag, filter_flag, 0.0, im_par, path_par.input_im_dir);      
            im_warp_calc_time_t = 0.0; %arbitrary
        else
            [u_t, OFcalc_time_t] = load_computeOF_for_warp(dvf_type, t, OF_par, path_par, im_par, br_model_par, pred_par, beh_par, warp_par);
            OFcalc_time_array(t - pred_par.t_eval_start + 1) = OFcalc_time_t;

            fprintf('Forward warping the optical flow at t=%d \n', t);
            [I_warped, im_warp_calc_time_t] = forward_warp2D(I_init, u_t, warp_par);
            % The last dimension of Iwarped is the run index, i.e. if
            % [~,~,nb_runs] = size(Iwarped), then nb_runs == time_signal_pred_results.nb_correct_runs (nb of runs without NaN predictions)
        end
        
        im_warp_calc_time_array(t - pred_par.t_eval_start + 1) = im_warp_calc_time_t;

        for run_idx = 1:time_signal_pred_results.nb_correct_runs
            
            % Loading the ground-truth image J at time t
            crop_flag = false; filter_flag = false;
            J = load_crop_filter2D(t, crop_flag, filter_flag, 0.0, im_par, path_par.input_im_dir);

            % Computing the statistics corresponding to the difference between J and I_warped
            im_correlation_array(t - pred_par.t_eval_start + 1, run_idx) = corr_two_im2d( I_warped(:,:,run_idx), J, beh_par.EVALUATE_IN_ROI, im_par);
            mssim_array(t - pred_par.t_eval_start + 1, run_idx) = my_ssim(I_warped(:,:,run_idx), J, beh_par.EVALUATE_IN_ROI, im_par);
            nrmse_array(t - pred_par.t_eval_start + 1, run_idx) = my_nrmse(I_warped(:,:,run_idx), J, beh_par.EVALUATE_IN_ROI, im_par);
            
            if (run_idx <= 1) && beh_par.SAVE_WARPED_IM
                
                % Saving the warped image I_warped (at time t since this is in the for loop, and run index "run_idx_for_save")
                warped_im_name = write_warp_2Dim_filename(dvf_type, t, path_par, OF_par, warp_par, pred_par, br_model_par );
                enhance_flag = true;
                run_idx_for_save = 1;
                save_crop_enhance_2Dim_jpg(I_warped(:,:,run_idx_for_save), warped_im_name, beh_par.CROP_FOR_DISP_SAVE, enhance_flag, ...
                    disp_par, path_par, im_par.x_m, im_par.x_M, im_par.y_m, im_par.y_M, t);
            
                % Saving the image difference J-I_warped as a thermal color image (at time t since this is in the for loop, and run index "run_idx_for_save")
                if strcmp(dvf_type, 'predicted DVF') % difference color image only if prediction at the moment (temporarily)
                    pred_param_str = sprintf_pred_param(pred_par);
                    difference_im_name = sprintf('prediction error image %s t = %d %s %s %d cpts', path_par.input_im_dir_suffix, ...
                        t, pred_par.pred_meth, pred_param_str, br_model_par.nb_pca_cp); 
                    save_crop_thermal_2Dim_jpg_fig(abs(J-I_warped(:,:,run_idx_for_save)), difference_im_name, crop_flag, disp_par, ...
                        path_par, im_par.x_m, im_par.x_M, im_par.y_m, im_par.y_M);
                end
                
            end
            
            % Computing the image difference J-I_warped at time t and run "run_idx"
            if strcmp(dvf_type, 'predicted DVF') && beh_par.SAVE_WARPED_IM % difference color image only if prediction
                temp_runs_tensor(:,:,run_idx) = abs(J-I_warped(:,:,run_idx));
            end
            
        end
        
        % Computing the image difference J-I_warped at time t over all the runs
        if strcmp(dvf_type, 'predicted DVF') && beh_par.SAVE_WARPED_IM % difference color image only if prediction
            diff_im_tensor(:,:,t - pred_par.t_eval_start + 1) = mean(temp_runs_tensor,3);
        end
        
    end
    
    eval_results = update_OFwarp_results(dvf_type, eval_results, im_correlation_array, mssim_array, nrmse_array, im_warp_calc_time_array, ...
                                            OFcalc_time_array, time_signal_pred_results);
    
    % Computing the image difference J-I_warped over t and all the run indexes, and saves it as a thermal image
    if strcmp(dvf_type, 'predicted DVF') && beh_par.SAVE_WARPED_IM % difference color image only if prediction
        pred_param_str = sprintf_pred_param(pred_par);
        difference_im_name = sprintf('mean prediction error im test set %s %s %s %d cpts', path_par.input_im_dir_suffix, pred_par.pred_meth, pred_param_str, br_model_par.nb_pca_cp); 
        mean_pred_error_test_im = mean(diff_im_tensor, 3);
        save_crop_thermal_2Dim_jpg_fig(mean_pred_error_test_im, difference_im_name, crop_flag, disp_par, path_par, im_par.x_m, im_par.x_M, im_par.y_m, im_par.y_M);
    end
    
end

