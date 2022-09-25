function [ warped_im_name] = write_warp_2Dim_filename( of_type_idx, t, path_par, OF_par, warp_par, pred_par, br_model_par )
% Returns the name of the warped images to be saved
%
% Author : Pohl Michel
% Date : Sept 18th, 2022
% Version : v1.0
% License : 3-clause BSD License


    %warp_par_str  = sprintf_warp_param( warp_par );
    OF_param_str = sprintf_OF_param(OF_par);

    switch of_type_idx
        case 1 % warping with the initial optical flow
            warped_im_name = sprintf('initOF warp %s t = %d %s', path_par.input_im_dir_suffix, t, OF_param_str);            

        case 2 % warping with the DVF reconstructed from PCA
            warped_im_name = sprintf('PCA warp %s t = %d %s %d components', path_par.input_im_dir_suffix, t, OF_param_str, br_model_par.nb_pca_cp);  
            
        case 3 % warping with the predicted optical flow    
            pred_param_str = sprintf_pred_param(pred_par); %function in the "Time_series_forecasting" folder
            warped_im_name = sprintf('PCA pred warp %s t = %d %s %s %d cpts', path_par.input_im_dir_suffix, t, pred_par.pred_meth_str, pred_param_str, br_model_par.nb_pca_cp); 
           
    end

end

