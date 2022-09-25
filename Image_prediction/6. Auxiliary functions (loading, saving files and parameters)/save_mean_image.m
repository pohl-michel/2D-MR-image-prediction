function save_mean_image(im_par, path_par, disp_par, pred_par)
% Saves the mean image of the test set
%
% Author : Pohl Michel
% Date : Sept 18th, 2022
% Version : v1.0
% License : 3-clause BSD License


    enhance_flag = false;
    % to change if necessary
    
    T_eval = pred_par.tmax_pred - pred_par.t_eval_start +1;
    im_tensor = zeros(im_par.W, im_par.L, T_eval);
    
    crop_flag = false; initial_filtering_flag = false; 
    for t=pred_par.t_eval_start:pred_par.tmax_pred
        im_tensor(:,:,t-pred_par.t_eval_start+1) = load_crop_filter2D(t, crop_flag, initial_filtering_flag, 0, im_par, path_par.input_im_dir);
    end
    
    mean_im_test_set = mean(im_tensor, 3);

    im_filename_suffix = sprintf('%s mean im test set', path_par.input_im_dir_suffix);
    t = 0; disp_par.write_im_time = false; 
    save_crop_enhance_2Dim_jpg(mean_im_test_set, im_filename_suffix, crop_flag, enhance_flag, disp_par, path_par, im_par.x_m, im_par.x_M, im_par.y_m, im_par.y_M, t)
    

end
