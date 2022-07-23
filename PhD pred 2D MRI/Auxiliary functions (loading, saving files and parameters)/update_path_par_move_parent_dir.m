function [path_par] = update_path_par_move_parent_dir(path_par)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

    path_par.time_series_data_filename = sprintf('PhD pred 2D MRI\\%s', path_par.time_series_data_filename);
    path_par.temp_fig_dir = sprintf('PhD pred 2D MRI\\%s', path_par.temp_fig_dir);
    path_par.temp_im_dir = sprintf('PhD pred 2D MRI\\%s', path_par.temp_im_dir);
    path_par.temp_var_dir = sprintf('PhD pred 2D MRI\\%s', path_par.temp_var_dir);  
    path_par.txt_file_dir = sprintf('PhD pred 2D MRI\\%s', path_par.txt_file_dir);  
    path_par.input_seq_dir = path_par.input_im_dir_suffix;

end

