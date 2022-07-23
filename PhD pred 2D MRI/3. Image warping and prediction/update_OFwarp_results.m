function [ eval_results ] = update_OFwarp_results( of_type_idx, eval_results, im_correlation_array, mssim_array, nrmse_array, im_warp_calc_time_array, OFcalc_time_array, time_signal_pred_results)
%UNTITLED25 Summary of this function goes here
%   Detailed explanation goes here

    switch of_type_idx
        
        case 1 % initial optical flow
            
            % average and half range correlation
            eval_results.mean_corr_initOF_warp = mean(im_correlation_array);
            eval_results.mean_nrmse_initOF_warp = mean(nrmse_array);
            eval_results.mean_ssim_initOF_warp = mean(mssim_array);
            
        case 2 % optical flow from several points with regression
            
            eval_results.mean_corr_warp_from_PCA = mean(im_correlation_array);
            
        case 3 % optical flow prediction
            
            % average and half range correlation
            eval_results.mean_corr_im_pred = mean(mean(im_correlation_array));
            eval_results.confidence_half_range_corr_im_pred = 1.96*std(mean(im_correlation_array))/sqrt(time_signal_pred_results.nb_correct_runs);
                % the mean is taken only once, so it is the mean along the time dimension
            
            % average and half range SSIM
            eval_results.mean_ssim = mean(mean(mssim_array));
            eval_results.confidence_half_range_ssim_im_pred = 1.96*std(mean(mssim_array))/sqrt(time_signal_pred_results.nb_correct_runs);
                
            eval_results.mean_nrmse = mean(mean(nrmse_array));
            eval_results.confidence_half_range_nrmse_im_pred = 1.96*std(mean(nrmse_array))/sqrt(time_signal_pred_results.nb_correct_runs);
            
            % average calculation time for one image
            eval_results.im_warp_calc_time_avg = mean(im_warp_calc_time_array);
            eval_results.OFrec_calc_time_avg = mean(OFcalc_time_array); % OF reconstruction from PCA
            
    end
    
    
end

