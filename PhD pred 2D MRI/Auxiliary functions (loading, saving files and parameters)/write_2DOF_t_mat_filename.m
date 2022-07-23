function [ OF_t_filename ] = write_2DOF_t_mat_filename( OF_par, path_par, t )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

OF_param_str = sprintf_OF_param(OF_par);
OF_t_filename = sprintf('%s\\2DOF %s t=1 t=%d - %s.mat', path_par.temp_var_dir, path_par.input_im_dir_suffix, t , OF_param_str);

end

