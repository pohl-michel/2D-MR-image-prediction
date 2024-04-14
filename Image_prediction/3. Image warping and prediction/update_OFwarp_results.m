function eval_results = update_OFwarp_results(eval_results, dvf_type, acc_metrics)
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
            
            % average correlation
            eval_results.mean_corr_initOF_warp = mean(acc_metrics.im_correlation_array);
            eval_results.mean_nrmse_initOF_warp = mean(acc_metrics.nrmse_array);
            eval_results.mean_ssim_initOF_warp = mean(acc_metrics.ssim_array);
            
        case 'DVF from PCA'
            
            eval_results.mean_corr_warp_from_PCA = mean(acc_metrics.im_correlation_array);
            
        case 'predicted DVF'
            
            % average and half range correlation
            [eval_results.mean_corr_im_pred, eval_results.confidence_half_range_corr_im_pred] = get_mean_and_confidence(acc_metrics.im_correlation_array);
            
            % average and half range SSIM
            [eval_results.mean_ssim, eval_results.confidence_half_range_ssim_im_pred] = get_mean_and_confidence(acc_metrics.ssim_array);
                
            % average and half range nRMSE
            [eval_results.mean_nrmse, eval_results.confidence_half_range_nrmse_im_pred] = get_mean_and_confidence(acc_metrics.nrmse_array);

            % average deformation error (averaged over the whole cycle)
            [eval_results.mean_pred_dvf_error, eval_results.confidence_half_range_mean_pred_dvf_err] = get_mean_and_confidence(acc_metrics.dvf_mean_error_array);

            % average maximum error (averaged over the whole cycle) - I could take maximum over the cycle, which would lead to a higher error value but the
            % choice is arbitrary
            [eval_results.max_pred_dvf_error, eval_results.confidence_half_range_max_pred_dvf_err] = get_mean_and_confidence(acc_metrics.dvf_max_error_array);

        case 'no prediction'

            [eval_results.mean_corr_no_pred, ~] = get_mean_and_confidence(acc_metrics.im_correlation_array);
            [eval_results.mean_ssim_no_pred, ~] = get_mean_and_confidence(acc_metrics.ssim_array);
            [eval_results.mean_nrmse_no_pred, ~] = get_mean_and_confidence(acc_metrics.nrmse_array);            

    end    
    
end


function [my_mean, my_confidence_half_range] = get_mean_and_confidence(my_array)
    [~, nb_runs] = size(my_array);
    my_mean = mean(mean(my_array));
    my_confidence_half_range = 1.96*std(mean(my_array))/sqrt(nb_runs); % the mean is taken only once, so it is the mean along the time dimension
end