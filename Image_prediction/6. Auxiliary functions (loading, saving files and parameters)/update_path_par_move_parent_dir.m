function [path_par] = update_path_par_move_parent_dir(path_par)
% Updates variables in path_par when moving in the parent directory of "Image_prediction" ("Future_frame_prediction") to use the functions in the folder "Time_series_forecasting"
% in the script image_prediction_main.
%
% Author : Pohl Michel
% Date : Sept 18th, 2022
% Version : v1.0
% License : 3-clause BSD License


    path_par.temp_fig_dir = sprintf('%s\\%s', path_par.im_pred_dir, path_par.temp_fig_dir);
    path_par.temp_im_dir = sprintf('%s\\%s', path_par.im_pred_dir, path_par.temp_im_dir);
    path_par.temp_var_dir = sprintf('%s\\%s', path_par.im_pred_dir, path_par.temp_var_dir);  
    path_par.txt_file_dir = sprintf('%s\\%s', path_par.im_pred_dir, path_par.txt_file_dir);  
    path_par.input_seq_dir = path_par.input_im_dir_suffix;

end

