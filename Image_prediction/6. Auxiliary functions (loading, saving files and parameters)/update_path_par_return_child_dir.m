function [path_par] = update_path_par_return_child_dir(path_par)
% Updates variables in path_par when moving back to the child directory of "Future_frame_prediction" ("Image_prediction")
% in the script image_prediction_main.
%
% Author : Pohl Michel
% Date : Sept 18th, 2022
% Version : v1.0
% License : 3-clause BSD License

    path_par.temp_fig_dir = 'b. Temporarily saved figures';
    path_par.temp_im_dir = 'c. Temporarily saved images';
    path_par.temp_var_dir = 'd. Temporarily saved variables';  
    path_par.txt_file_dir = 'e. Temporarily saved txt files'; 
        
end

