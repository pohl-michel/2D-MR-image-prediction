function time_series_data_filename = write_PCAweights_mat_filename(OF_par, path_par, br_model_par)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    % PCA weights filename
    OF_param_str = sprintf_OF_param(OF_par);
    time_series_data_filename = sprintf('%s\\PCA weights %s %s %d cpts.mat', path_par.temp_var_dir, path_par.input_im_dir_suffix, OF_param_str, br_model_par.nb_pca_cp);   

end
