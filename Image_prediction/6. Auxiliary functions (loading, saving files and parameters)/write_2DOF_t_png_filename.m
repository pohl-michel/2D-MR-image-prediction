function [ filename ] = write_2DOF_t_png_filename( beh_par, OF_par, path_par, t )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

OF_param_str = sprintf_OF_param(OF_par);
filename = sprintf('%s\\2DOF %s t=1 t=%d - %s', path_par.temp_im_dir, path_par.input_im_dir_suffix, t, OF_param_str);
if beh_par.CROP_FOR_DISP_SAVE
    filename = sprintf('%s ROI', filename);
end
filename = sprintf('%s.jpg', filename);


end

