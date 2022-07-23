function [ filename ] = write_PCAresults_mat_filename( beh_par, OF_par, path_par )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

OF_param_str = sprintf_OF_param(OF_par);
filename = sprintf('%s\\PCA of DVF %s - %s', path_par.temp_var_dir, path_par.input_im_dir_suffix, OF_param_str);
if beh_par.CROP_FOR_DISP_SAVE
    filename = sprintf('%s ROI', filename);
end
filename = sprintf('%s.mat', filename);