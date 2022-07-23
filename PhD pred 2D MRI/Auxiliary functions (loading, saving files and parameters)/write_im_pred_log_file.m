function write_im_pred_log_file(path_par, beh_par, im_par, OF_par, hppars, pred_par, br_model_par, warp_par, eval_results)
% Saves the parameters used and some of the numerical results in a txt file created in the Matlab workspace.

    log_file_complete_filename = sprintf('%s\\%s %s', path_par.txt_file_dir, path_par.input_im_dir_suffix, path_par.log_txt_filename);
    fid = fopen(log_file_complete_filename,'wt');
        
        fprintf(fid, '%s \n',path_par.date_and_time);
 
        % I] Writing down the calculation paremeters        
        
        fprintf(fid, 'image sequence %s \n\n',path_par.input_im_dir_suffix);

        if beh_par.EVALUATE_IN_ROI
            fprintf(fid, 'evaluation in the region of interest \n');
            fprintf(fid, 'x_m = %d \n', im_par.x_m);
            fprintf(fid, 'x_M = %d \n', im_par.x_M);        
            fprintf(fid, 'y_m = %d \n', im_par.y_m);        
            fprintf(fid, 'y_M = %d \n', im_par.y_M); 
            fprintf(fid, 'z_m = %d \n', im_par.z_m);        
            fprintf(fid, 'z_M = %d \n', im_par.z_M);
        else
            fprintf(fid, 'evaluation in the entire image \n');
        end
        fprintf(fid, '\n');
        
        fprintf(fid, 'Optical flow calculation parameters \n');
        fprintfOFpar( fid, OF_par );
        fprintf(fid, '\n');
        
        fprintf(fid, 'Reconstruction : %s \n', sprintf_warp_param( warp_par ));
        switch(warp_par.kernel_appl_meth)
            case 1
                fprintf(fid, 'Kernel calculation method : matrix computation \n');
            case 2
                fprintf(fid, 'Kernel calculation method : pointwise computation \n');
        end        
        fprintf(fid, '\n');
        if (pred_par.pred_meth_idx == 2)||(pred_par.pred_meth_idx == 5) % stochastic method such as RTRL or UORO
            fprintf(fid, 'Image cross correlation calculated using %d runs \n', warp_par.nb_runs_for_cc_eval);
        end
        
        % Optimization of the prediction method
        if beh_par.OPTIMIZE_NB_PCA_CP
            fprintf(fid, 'The number of principal components has been optimized on the cross validation set \n');
            fprintf(fid, 'Cross validation with number of principal components from 1 to %d \n', br_model_par.nb_pca_cp_max);
            fprintf(fid, '\n');
            
            fprintf(fid, 'Optimization of the prediction: \n');
            fprintf(fid, 'Prediction method : %s \n', pred_par.pred_meth_str);
            fprintf(fid, 'Training between t = 1 and t = %d \n', pred_par.tmax_training);
            fprintf(fid, 'Cross-validation between t = %d and t = %d \n', 1+pred_par.tmax_training, pred_par.tmax_cv);
            fprintf(fid, 'Number of runs on the cross validation set due to random weights initialization nb_runs_cv = %d \n', hppars.nb_runs_cv);

            fprintf(fid, 'Range of parameters tested \n');   
            for hppar_idx = 1:hppars.nb_additional_params
                fprintf(fid, '%s : \n', hppars.other(hppar_idx).name);
                for par_val_idx = 1:hppars.other(hppar_idx).nb_val
                    fprintf(fid, '%g \t', hppars.other(hppar_idx).val(par_val_idx)); 
                end
                fprintf(fid, '\n');
            end
            
        end
        fprintf(fid, '\n');
        
        
        % Prediction paremeters  
        if beh_par.OPTIMIZE_NB_PCA_CP  fprintf(fid, 'Parameters selected after cross validation for the horizon h=%d \n', pred_par.horizon); end
        if beh_par.TRAIN_EVAL_PREDICTOR || beh_par.IM_PREDICTION || beh_par.OPTIMIZE_NB_PCA_CP || beh_par.EVAL_PCA_RECONSTRUCT
            fprintf(fid, 'Number of principal components used for prediction/evaluation (test set) :%d \n', br_model_par.nb_pca_cp);        
            if beh_par.TRAIN_EVAL_PREDICTOR || beh_par.IM_PREDICTION || beh_par.OPTIMIZE_NB_PCA_CP
            fprintfpred_par( fid, pred_par, beh_par );
            end
            fprintf(fid, '\n');
        end
        
        % II] Writing down the evaluation results
        
        if beh_par.OPTIMIZE_NB_PCA_CP
            fprintf(fid, 'Horizon values tested: \n');
            for hrz_idx = 1:hppars.nb_hrz_val
                fprintf(fid, '%d \t', hppars.horizon_tab(hrz_idx)); 
            end
            fprintf(fid, '\n');
            
            fprintf(fid, 'Effect of the number of PCA components and horizon on the c.c. of the c.v. set: \n');
            fprintf(fid, '(Lines: horizon value / columns: nb of PCA components) \n');
            for hrz_idx = 1:hppars.nb_hrz_val
                for nb_pca_cp = 1:br_model_par.nb_pca_cp_max
                    fprintf(fid, '%g \t', eval_results.cross_cor_pca_optim_tab(hrz_idx, nb_pca_cp)); 
                end
                fprintf(fid, '\n');
            end
            fprintf(fid, '\n');
        end
        
        fprintf(fid, 'Calculation time \n');
        if beh_par.COMPUTE_OPTICAL_FLOW fprintf(fid, 'Computation of the DVF with pyramidal iterative LK (for one image or time step) : %f s\n', eval_results.OF_calc_time); end
        if beh_par.PCA_OF_DVF fprintf(fid, 'PCA weights computation (in real time) for one time step : %f s\n', eval_results.PCA_time_weights_calc_time); end
        if beh_par.IM_PREDICTION  fprintf(fid, 'Optical flow reconstruction from PCA (for one image or time step) : %f s\n', eval_results.OFrec_calc_time_avg); end
        if beh_par.IM_PREDICTION  fprintf(fid, 'Optical flow warping (for one image or time step) : %f s\n', eval_results.im_warp_calc_time_avg); end    
        fprintf(fid, '\n');
        
        fprintf(fid, 'Evaluation results \n');
        if beh_par.EVAL_INIT_OF_WARP 
            fprintf(fid, 'Mean cross-correlation between original images and the 1st image warped with initOF: %f \n', eval_results.mean_corr_initOF_warp); 
            fprintf(fid, 'Mean nRMSE between original images and the 1st image warped with initOF: %f \n', eval_results.mean_nrmse_initOF_warp);
            fprintf(fid, 'Mean SSIM between original images and the 1st image warped with initOF: %f \n', eval_results.mean_ssim_initOF_warp);
            fprintf(fid, 'Same results but for copy-pasting in Excel : \n');
            fprintf(fid, '%f\n', eval_results.mean_corr_initOF_warp);
            fprintf(fid, '%f\n', eval_results.mean_nrmse_initOF_warp);
            fprintf(fid, '%f\n', eval_results.mean_ssim_initOF_warp); 
        end 
        if beh_par.EVAL_PCA_RECONSTRUCT fprintf(fid, 'Mean cross-cor between orgnal images and the 1st image warped with OF reconstructed from PCA : %f \n', eval_results.mean_corr_warp_from_PCA); end        
        if beh_par.IM_PREDICTION  
            fprintf(fid, 'Mean cross-correlation between predicted and original images : %f \n', eval_results.mean_corr_im_pred);
            fprintf(fid, 'Confidence half-range of the mean cross-correlation between predicted and original images : %f \n', eval_results.confidence_half_range_corr_im_pred);
            fprintf(fid, 'Mean nmrse between predicted and original images : %f \n', eval_results.mean_nrmse);
            fprintf(fid, 'Confidence half-range of the mean nrmse between predicted and original images : %f \n', eval_results.confidence_half_range_nrmse_im_pred);
            fprintf(fid, 'Mean ssim between predicted and original images : %f \n', eval_results.mean_ssim);
            fprintf(fid, 'Confidence half-range of the mean ssim between predicted and original images : %f \n', eval_results.confidence_half_range_ssim_im_pred);
            
            fprintf(fid, 'Same results but for copy-pasting in Excel : \n');
            switch(pred_par.pred_meth_idx)
                case {2, 5, 7} %RNN
                    fprintf(fid, '%f\n', eval_results.mean_corr_im_pred);
                    fprintf(fid, '%f\n', eval_results.confidence_half_range_corr_im_pred);
                    fprintf(fid, '%f\n', eval_results.mean_nrmse);
                    fprintf(fid, '%f\n', eval_results.confidence_half_range_nrmse_im_pred);
                    fprintf(fid, '%f\n', eval_results.mean_ssim);
                    fprintf(fid, '%f\n', eval_results.confidence_half_range_ssim_im_pred);
                otherwise
                    fprintf(fid, '%f\n', eval_results.mean_corr_im_pred);
                    fprintf(fid, '%f\n', eval_results.mean_nrmse);
                    fprintf(fid, '%f\n', eval_results.mean_ssim);               
            end
        end        
        
        if beh_par.NO_PRED_AT_ALL 
            fprintf(fid, 'Case when no prediction is performed \n'); 
            fprintf(fid, 'Evaluation between t = %d and t = %d \n', pred_par.t_eval_start, pred_par.tmax_pred);            
            fprintf(fid, 'Mean cross-correlation between images at t and images at t-%d: %f \n', pred_par.horizon, eval_results.mean_corr_im_pred); 
            fprintf(fid, 'Mean nmrse between images at t and images at t-%d: %f \n', pred_par.horizon, eval_results.mean_nrmse);
            fprintf(fid, 'Mean ssim between images at t and images at t-%d : %f \n', pred_par.horizon, eval_results.mean_ssim);
            fprintf(fid, 'Same results but for copy-pasting in Excel : \n');
            fprintf(fid, '%f\n', eval_results.mean_corr_im_pred);
            fprintf(fid, '%f\n', eval_results.mean_nrmse);
            fprintf(fid, '%f\n', eval_results.mean_ssim); 
        end               
        
        fprintf(fid, '\n');
        
    fclose(fid);

end