function [ warped_im_name] = write_warp_2Dim_filename( of_type_idx, t, path_par, OF_par, warp_par, pred_par, br_model_par )
%UNTITLED19 Summary of this function goes here
%   Detailed explanation goes here

    warp_par_str  = sprintf_warp_param( warp_par );
    OF_param_str = sprintf_OF_param(OF_par);

    switch of_type_idx % UTILISER UNE FONCTION AUXILIAIRE
        case 1 % initial optical flow
            warped_im_name = sprintf('initOF warp %s t = %d %s', path_par.input_im_dir_suffix, t, OF_param_str);            

        case 2 % optical flow from several points with regression
            warped_im_name = sprintf('PCA warp %s t = %d %s %d components', path_par.input_im_dir_suffix, t, OF_param_str, br_model_par.nb_pca_cp);  
            
        case 3 % optical flow prediction             
            pred_param_str = sprintf_pred_param(pred_par); %function in the folder "Time series forecasting"
            warped_im_name = sprintf('PCA pred warp %s t = %d %s %s %d cpts', path_par.input_im_dir_suffix, t, pred_par.pred_meth_str, pred_param_str, br_model_par.nb_pca_cp); 
           
    end

end

