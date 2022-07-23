function [ filename ] = write_DVF_component_jpg_filename( beh_par, OF_par, path_par, cpt_idx, disp_par )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

OF_param_str = sprintf_OF_param(OF_par);
filename = sprintf('%s\\DVF %d-th component scale_factor=%d %s - %s', path_par.temp_im_dir, cpt_idx, disp_par.PCA_cp_scale_factor, path_par.input_im_dir_suffix, OF_param_str);
if beh_par.CROP_FOR_DISP_SAVE
    filename = sprintf('%s ROI', filename);
end
filename = sprintf('%s.jpg', filename);