function eval_results = update_OFwarp_results(dvf_type, eval_results, im_correlation_array, mssim_array, nrmse_array, im_warp_calc_time_array, ...
                                                                                                                    OFcalc_time_array, time_signal_pred_results)
% "update_OFwarp_results" computes mean statistics concerning the image reconstruction accuracy (of the test set or cross-validation set): 
% image correlation, nrmse, ssim, and calculation time.
% "update_OFwarp_results" is called at the end of "eval_of_warp_corr" after warping the image at t=1 by the concerned DVF.
% The statistics are averaged over the number of images in the test set or cross-validation set, and also over the number of runs when predicting images
%
% Author : Pohl Michel
% Date : Sept 23rd, 2022
% Version : v1.1
% License : 3-clause BSD License


    switch dvf_type
        
        case 'initial DVF'
            
            % average and half range correlation
            eval_results.mean_corr_initOF_warp = mean(im_correlation_array);
            eval_results.mean_nrmse_initOF_warp = mean(nrmse_array);
            eval_results.mean_ssim_initOF_warp = mean(mssim_array);
            
        case 'DVF from PCA'
            
            eval_results.mean_corr_warp_from_PCA = mean(im_correlation_array);
            
        case 'predicted DVF'
            
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