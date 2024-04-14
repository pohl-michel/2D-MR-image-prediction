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

    pred_param_str = sprintf_pred_param(pred_par); % auxiliary variable used when making file names

    % Loading the image à t=1
    t_init = 1; crop_flag = false; filter_flag = false; sigma_init = 'whatever';
    I_init = load_crop_filter2D(t_init, crop_flag, filter_flag, sigma_init, im_par, path_par.input_im_dir);
    
    % Initializing various variables
    im_warp_calc_time_array = zeros(pred_par.nb_predictions, 1);
    OFcalc_time_array = zeros(pred_par.nb_predictions, 1);
    
    acc_metrics = struct(); 
    acc_metrics.whole_im = struct(); 
    if beh_par.EVALUATE_IN_ROI
        acc_metrics.roi = struct(); 
    end

    if ismember(dvf_type, {'initial DVF', 'DVF from PCA'}) || beh_par.NO_PRED_AT_ALL % evaluation of warping with the initial OF or evaluation of warping after PCA (no prediction)
        time_signal_pred_results.nb_correct_runs = 1;
    end

    % Storing prediction metrics regarding image pixel intensities
    acc_metrics.whole_im.im_correlation_array = zeros(pred_par.nb_predictions, time_signal_pred_results.nb_correct_runs);   
    acc_metrics.whole_im.ssim_array = zeros(pred_par.nb_predictions, time_signal_pred_results.nb_correct_runs);   
    acc_metrics.whole_im.nrmse_array = zeros(pred_par.nb_predictions, time_signal_pred_results.nb_correct_runs);    

    if beh_par.EVALUATE_IN_ROI
        acc_metrics.roi.ssim_array = zeros(pred_par.nb_predictions, time_signal_pred_results.nb_correct_runs);  
        acc_metrics.roi.im_correlation_array = zeros(pred_par.nb_predictions, time_signal_pred_results.nb_correct_runs);   
        acc_metrics.roi.nrmse_array = zeros(pred_par.nb_predictions, time_signal_pred_results.nb_correct_runs);    
    end

    % Storing prediction metrics regarding geometrical deformation error (assuming that the original OF/DVF is the ground-truth)
    acc_metrics.whole_im.dvf_mean_error_array = zeros(pred_par.nb_predictions, time_signal_pred_results.nb_correct_runs);  
    acc_metrics.whole_im.dvf_max_error_array = zeros(pred_par.nb_predictions, time_signal_pred_results.nb_correct_runs);  
    if beh_par.EVALUATE_IN_ROI
        acc_metrics.roi.dvf_mean_error_array = zeros(pred_par.nb_predictions, time_signal_pred_results.nb_correct_runs);  
        acc_metrics.roi.dvf_max_error_array = zeros(pred_par.nb_predictions, time_signal_pred_results.nb_correct_runs);  
    end     
    
    if strcmp(dvf_type, 'predicted DVF') && beh_par.SAVE_WARPED_IM % difference color image only if prediction (at the moment)
        diff_im_tensor = zeros(im_par.W, im_par.L, pred_par.nb_predictions);
        diff_dvf_tensor = zeros(im_par.W, im_par.L, pred_par.nb_predictions);
        temp_runs_tensor = zeros(im_par.W, im_par.L, time_signal_pred_results.nb_correct_runs); % contains the images I(:,:,t) - Iwarped(:,:,t,run_idx) for t fixed
        temp_runs_dvf_tensor = zeros(im_par.W, im_par.L, time_signal_pred_results.nb_correct_runs); % contains the images "norm(u_t(:,:,t) - u_t_pred(:,:,t,run_idx))" for t fixed
    end
    
    for t=pred_par.t_eval_start:pred_par.tmax_pred

        % Loading the original DVF u_t_org between t=1 and t
        [u_t_org, ~] = load_computeOF_for_warp('initial DVF', t, OF_par, path_par, im_par, br_model_par, pred_par, beh_par, warp_par); % only (t, OF_par, path_par) are used as arguments

        % Loading/computing the (predicted or not) DVF u_t between t=1 and t and computing I_warped, which results from warping I_init with u_t.
        if strcmp(dvf_type, 'no prediction')   
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

            % Image I warped by the push-forward DVF u_t at time t with the current run run_idx
            I_warped_crt_run = I_warped(:,:,run_idx);

            % Computing the statistics corresponding to the difference between J and I_warped
            eval_in_roi = false;
            acc_metrics.whole_im.nrmse_array(t - pred_par.t_eval_start + 1, run_idx) = my_nrmse(I_warped_crt_run, J, eval_in_roi, im_par);
            acc_metrics.whole_im.ssim_array(t - pred_par.t_eval_start + 1, run_idx) = my_ssim(I_warped_crt_run, J, eval_in_roi, im_par);
            acc_metrics.whole_im.im_correlation_array(t - pred_par.t_eval_start + 1, run_idx) = corr_two_im2d(I_warped_crt_run, J, eval_in_roi, im_par);

            if beh_par.EVALUATE_IN_ROI
                eval_in_roi = true;
                acc_metrics.roi.ssim_array(t - pred_par.t_eval_start + 1, run_idx) = my_ssim(I_warped_crt_run, J, eval_in_roi, im_par);
                acc_metrics.roi.im_correlation_array(t - pred_par.t_eval_start + 1, run_idx) = corr_two_im2d(I_warped_crt_run, J, eval_in_roi, im_par);
                acc_metrics.roi.nrmse_array(t - pred_par.t_eval_start + 1, run_idx) = my_nrmse(I_warped_crt_run, J, eval_in_roi, im_par);
            end

            % Computing statistics corresponding to the difference between u_t_org and u_t
            if strcmp(dvf_type, 'predicted DVF')
                u_t_diff = sqrt((u_t_org(:,:,1) - u_t(:,:,1, run_idx)).^2+(u_t_org(:,:,2) - u_t(:,:,2, run_idx)).^2); % pixel-wise euclidean norm of the difference
                flattened_u_t_diff = u_t_diff(:);
                acc_metrics.whole_im.dvf_mean_error_array(t - pred_par.t_eval_start + 1, run_idx) = mean(flattened_u_t_diff);
                acc_metrics.whole_im.dvf_max_error_array(t - pred_par.t_eval_start + 1, run_idx) = max(flattened_u_t_diff);                
                if beh_par.EVALUATE_IN_ROI
                    u_t_roi_diff = u_t_diff(im_par.y_m:im_par.y_M, im_par.x_m:im_par.x_M);
                    flattened_u_t_roi_diff = u_t_roi_diff(:);
                    acc_metrics.roi.dvf_mean_error_array(t - pred_par.t_eval_start + 1, run_idx) = mean(flattened_u_t_roi_diff);
                    acc_metrics.roi.dvf_max_error_array(t - pred_par.t_eval_start + 1, run_idx) = max(flattened_u_t_roi_diff);       
                end 
            end
            
            if (run_idx <= 1) && beh_par.SAVE_WARPED_IM
                
                % Saving the warped image I_warped (at time t since this is in the for loop, and run index "run_idx_for_save")
                warped_im_name = write_warp_2Dim_filename(dvf_type, t, path_par, OF_par, warp_par, pred_par, br_model_par );
                enhance_flag = true;
                run_idx_for_save = 1;
                save_crop_enhance_2Dim_jpg(I_warped(:,:,run_idx_for_save), warped_im_name, beh_par.CROP_FOR_DISP_SAVE, enhance_flag, ...
                    disp_par, path_par, im_par.x_m, im_par.x_M, im_par.y_m, im_par.y_M, t);
            
                % Saving the image difference J-I_warped and DVF difference as thermal color images (at time t since this is in the for loop, and run index "run_idx_for_save")
                if strcmp(dvf_type, 'predicted DVF') % difference color image only if prediction at the moment (temporarily)
                    
                    difference_im_name = sprintf('prediction error image %s t = %d %s %s %d cpts', path_par.input_im_dir_suffix, ...
                        t, pred_par.pred_meth, pred_param_str, br_model_par.nb_pca_cp); 
                    difference_dvf_name = sprintf('predicted dvf error %s t = %d %s %s %d cpts', path_par.input_im_dir_suffix, ...
                        t, pred_par.pred_meth, pred_param_str, br_model_par.nb_pca_cp);
                    
                    save_crop_thermal_2Dim_jpg_fig(abs(double(J)-I_warped(:,:,run_idx_for_save)), difference_im_name, crop_flag, disp_par, ...
                        path_par, im_par.x_m, im_par.x_M, im_par.y_m, im_par.y_M);
                    save_crop_thermal_2Dim_jpg_fig(u_t_diff, difference_dvf_name, crop_flag, disp_par, ...
                        path_par, im_par.x_m, im_par.x_M, im_par.y_m, im_par.y_M);
                end
                
            end

            % Computing the image difference J-I_warped at time t and run "run_idx" - and storing as well the difference between original and predicted DVFs
            if strcmp(dvf_type, 'predicted DVF') && beh_par.SAVE_WARPED_IM % difference color image only if prediction
                temp_runs_tensor(:,:,run_idx) = abs(double(J)-I_warped_crt_run);
                temp_runs_dvf_tensor(:,:,run_idx) = u_t_diff;
            end
            
        end
        
        % Computing the mean image difference J-I_warped and ean DVF difference norm(pred_DVF-org_DVF) at time t over all the runs
        if strcmp(dvf_type, 'predicted DVF') && beh_par.SAVE_WARPED_IM % difference color image only if prediction
            diff_im_tensor(:,:,t - pred_par.t_eval_start + 1) = mean(temp_runs_tensor,3);
            diff_dvf_tensor(:,:,t - pred_par.t_eval_start + 1) = mean(temp_runs_dvf_tensor,3);
        end
        
    end
    
    eval_results.whole_im = update_OFwarp_results(eval_results.whole_im, dvf_type, acc_metrics.whole_im);
    if beh_par.EVALUATE_IN_ROI
        eval_results.roi = update_OFwarp_results(eval_results.roi, dvf_type, acc_metrics.roi);
    end
    
    % average calculation time for one image
    eval_results.im_warp_calc_time_avg = mean(im_warp_calc_time_array);
    eval_results.OFrec_calc_time_avg = mean(OFcalc_time_array); % OF reconstruction from PCA

    
    % Computing the image difference J-I_warped and the DVF difference norm(pred_DVF-org_DVF) over t and all the run indexes, and saves it as a thermal image
    if strcmp(dvf_type, 'predicted DVF') && beh_par.SAVE_WARPED_IM % difference color image only if prediction

        mean_pred_error_test_im = mean(diff_im_tensor, 3);
        difference_im_name = sprintf('mean prediction error im test set %s %s %s %d cpts', path_par.input_im_dir_suffix, pred_par.pred_meth, pred_param_str, br_model_par.nb_pca_cp); 
        save_crop_thermal_2Dim_jpg_fig(mean_pred_error_test_im, difference_im_name, crop_flag, disp_par, path_par, im_par.x_m, im_par.x_M, im_par.y_m, im_par.y_M);

        mean_pred_dvf_error_test = mean(diff_dvf_tensor, 3);
        difference_dvf_name = sprintf('mean test dvf error im %s %s %s %d cpts', path_par.input_im_dir_suffix, pred_par.pred_meth, pred_param_str, br_model_par.nb_pca_cp); 
        save_crop_thermal_2Dim_jpg_fig(mean_pred_dvf_error_test, difference_dvf_name, crop_flag, disp_par, path_par, im_par.x_m, im_par.x_M, im_par.y_m, im_par.y_M);        

    end
    
end

