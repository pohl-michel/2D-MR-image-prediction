function [ eval_results ] = eval_of_warp_corr( of_type_idx, im_par, OF_par, path_par, warp_par, pred_par, br_model_par, disp_par, beh_par, eval_results, time_signal_pred_results)
%UNTITLED17 Summary of this function goes here
%   Detailed explanation goes here

% Note : in the case of_type_idx = 4 (extrapolated optical flow for 4D sequence generation),
% "pred_par" actually means "extrapolation parameters" 
% In order to keep the number of arguments to a minimum, we use pred_par instead of extrl_par.

    % chargement de l'image à t=1
    t_init = 1; crop_flag = false; filter_flag = false; sigma_init = 'whatever';
    I_init = load_crop_filter2D(t_init, crop_flag, filter_flag, sigma_init, im_par, path_par.input_im_dir);
    
    im_warp_calc_time_array = zeros(pred_par.nb_predictions, 1);
    OFcalc_time_array = zeros(pred_par.nb_predictions, 1);
    
    if (of_type_idx == 1)||(of_type_idx == 2)||beh_par.NO_PRED_AT_ALL  
        % evaluation of the warp with the initial OF or evaluation of the warp after PCA (no prediction)
        time_signal_pred_results.nb_correct_runs = 1;
    end
    
    im_correlation_array = zeros(pred_par.nb_predictions, time_signal_pred_results.nb_correct_runs);   
    mssim_array = zeros(pred_par.nb_predictions, time_signal_pred_results.nb_correct_runs);   
    nrmse_array = zeros(pred_par.nb_predictions, time_signal_pred_results.nb_correct_runs);    
    
    if (of_type_idx == 3)&&(beh_par.SAVE_WARPED_IM) % difference color image only if prediction
        diff_im_tensor = zeros(im_par.W, im_par.L, pred_par.nb_predictions);
        temp_runs_tensor = zeros(im_par.W, im_par.L, time_signal_pred_results.nb_correct_runs); % contains the images I(:,:,t) - Iwarped(:,:,t,run_idx) for t fixed
    end
    
    for t=pred_par.t_eval_start:pred_par.tmax_pred

        if beh_par.NO_PRED_AT_ALL        

            I_warped = load_crop_filter2D(t-pred_par.horizon, crop_flag, filter_flag, 0.0, im_par, path_par.input_im_dir);      
            im_warp_calc_time_t = 0.0; %arbitrary
        
        else
            
            [ u_t, OFcalc_time_t ] = load_computeOF_for_warp( of_type_idx, t, OF_par, path_par, im_par, br_model_par, pred_par, beh_par, warp_par);
            OFcalc_time_array(t - pred_par.t_eval_start + 1) = OFcalc_time_t;

            % Pas de GPU ou de code C pour le moment
            fprintf('Forward warping the optical flow at t=%d \n', t);
            [I_warped, im_warp_calc_time_t] = forward_warp2D(I_init, u_t, warp_par);
            % Ici Iwarped va avoir une dernière dimension qui correspond au numéro du run
            % ie [~,~,nb_runs] = size(Iwarped) où nb_runs == time_signal_pred_results.nb_correct_runs (ie après avoir supprimé les prédictions avec NaN
            
        end
        
        im_warp_calc_time_array(t - pred_par.t_eval_start + 1) = im_warp_calc_time_t;

        for run_idx = 1:time_signal_pred_results.nb_correct_runs
            
            % chargement de l'image à t
            crop_flag = false; filter_flag = false;
            J = load_crop_filter2D(t, crop_flag, filter_flag, 0.0, im_par, path_par.input_im_dir);
            im_correlation_array(t - pred_par.t_eval_start + 1, run_idx) = corr_two_im2d( I_warped(:,:,run_idx), J, beh_par.EVALUATE_IN_ROI, im_par);
            mssim_array(t - pred_par.t_eval_start + 1, run_idx) = my_ssim(I_warped(:,:,run_idx), J, beh_par.EVALUATE_IN_ROI, im_par);
            nrmse_array(t - pred_par.t_eval_start + 1, run_idx) = my_nrmse(I_warped(:,:,run_idx), J, beh_par.EVALUATE_IN_ROI, im_par);
            
            if (run_idx <= 1)&&(beh_par.SAVE_WARPED_IM)
                % saving the warped images
                warped_im_name = write_warp_2Dim_filename( of_type_idx, t, path_par, OF_par, warp_par, pred_par, br_model_par );
                enhance_flag = true;
                run_idx_for_save = 1;
                save_crop_enhance_2Dim_jpg(I_warped(:,:,run_idx_for_save), warped_im_name, beh_par.CROP_FOR_DISP_SAVE, enhance_flag, disp_par, path_par, im_par.x_m, im_par.x_M, im_par.y_m, im_par.y_M, t);
            
                % saving the image difference as a thermal color image
                if (of_type_idx == 3) % difference color image only if prediction at the moment (temporarily)
                    pred_param_str = sprintf_pred_param(pred_par); %function in the folder "Time series forecasting"
                    difference_im_name = sprintf('prediction error image %s t = %d %s %s %d cpts', path_par.input_im_dir_suffix, t, pred_par.pred_meth_str, pred_param_str, br_model_par.nb_pca_cp); 
                    save_crop_thermal_2Dim_jpg_fig(abs(J-I_warped(:,:,run_idx_for_save)), difference_im_name, crop_flag, disp_par, path_par, im_par.x_m, im_par.x_M, im_par.y_m, im_par.y_M);
                end
                
            end
            
            if (of_type_idx == 3)&&(beh_par.SAVE_WARPED_IM) % difference color image only if prediction
                temp_runs_tensor(:,:,run_idx) = abs(J-I_warped(:,:,run_idx));
            end
            
        end
        
        if (of_type_idx == 3)&&(beh_par.SAVE_WARPED_IM) % difference color image only if prediction
            diff_im_tensor(:,:,t - pred_par.t_eval_start + 1) = mean(temp_runs_tensor,3);
        end
        
    end
    
    eval_results = update_OFwarp_results( of_type_idx, eval_results, im_correlation_array, mssim_array, nrmse_array, im_warp_calc_time_array, OFcalc_time_array, time_signal_pred_results);
    
    if (of_type_idx == 3)&&(beh_par.SAVE_WARPED_IM) % difference color image only if prediction
        pred_param_str = sprintf_pred_param(pred_par); %function in the folder "Time series forecasting"
        difference_im_name = sprintf('mean prediction error im test set %s %s %s %d cpts', path_par.input_im_dir_suffix, pred_par.pred_meth_str, pred_param_str, br_model_par.nb_pca_cp); 
        mean_pred_error_test_im = mean(diff_im_tensor, 3);
        save_crop_thermal_2Dim_jpg_fig(mean_pred_error_test_im, difference_im_name, crop_flag, disp_par, path_par, im_par.x_m, im_par.x_M, im_par.y_m, im_par.y_M);
    end
    
end

