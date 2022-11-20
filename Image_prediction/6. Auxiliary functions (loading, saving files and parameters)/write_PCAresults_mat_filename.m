function [ filename ] = write_PCAresults_mat_filename( beh_par, OF_par, path_par, br_model_par )
% Returns the filename containing the results of the PCA calculation from the deformation vector field data
% That file will be in path_par.temp_var_dir
%
% Author : Pohl Michel
% Date : Sept 18th, 2022
% Version : v1.1
% License : 3-clause BSD License


    OF_param_str = sprintf_OF_param(OF_par);
    filename = sprintf('%s\\PCA of DVF %s - %s %d pca cpts', path_par.temp_var_dir, path_par.input_im_dir_suffix, OF_param_str, br_model_par.nb_pca_cp);
    if beh_par.CROP_FOR_DISP_SAVE
        filename = sprintf('%s ROI', filename);
    end
    filename = sprintf('%s.mat', filename);

end