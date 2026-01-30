function save_main_DVF_jpg(beh_par, path_par, disp_par, OF_par, im_par, br_model_par, W, Xtrain_mean)
% Displays the mean DVF over time and its principal components on top of the initial image at t=1.
% The DVF images are saved in the folder path_par.im_par_filename.
%
% Author : Pohl Michel
% Date : September 25th, 2022
% Version : v2.0
% License : 3-clause BSD License

    
    % loading image at time t=1
    PERFORM_ENHANCEMENT = false;
    im_t1 = enhance_2Dim(load_crop_filter2D(1, beh_par.CROP_FOR_DISP_SAVE, false, 0, im_par, path_par.input_im_dir), PERFORM_ENHANCEMENT);    
        % the image is cropped if we select a specific region where the optical flow should be displayed (beh_par.CROP_FOR_DISPLAY = 1)
        % The image is not filtered because it is used to display the optical flow here (FILTER = false)
    
    G = return_mask_for_DVF_display(im_t1, disp_par);
    SAVE_OF_JPG = true;

    % We plot the temporal mean of the DVF
    u = reshape_DVF(im_par, Xtrain_mean);
    DVFmean_jpg_filename = write_DVF_jpg_filename(0, beh_par, OF_par, path_par, disp_par);
    display_save_DVF(u, im_t1, G, DVFmean_jpg_filename, im_par, beh_par, disp_par, SAVE_OF_JPG);
    
    % Then we plot each component
    for cpt_idx=1:br_model_par.nb_pca_cp
        u = reshape_DVF(im_par, W(:, cpt_idx));
        DVF_component_jpg_filename = write_DVF_jpg_filename(cpt_idx, beh_par, OF_par, path_par, disp_par);
        display_save_DVF(u, im_t1, disp_par.PCA_cp_scale_factor*G, DVF_component_jpg_filename, im_par, beh_par, disp_par, SAVE_OF_JPG);        
    end
    
end


function filename = write_DVF_jpg_filename(pca_cpt_idx, beh_par, OF_par, path_par, disp_par)
% Write the name of the image file containing the pca_cpt_idx-th principal deformation field.
% By convention, if pca_cpt_idx == 0, this is the file containing the mean DVF over time.
%
% Author : Pohl Michel
% Date : September 25th, 2022
% Version : v2.0
% License : 3-clause BSD License


    OF_param_str = sprintf_OF_param(OF_par);

    if pca_cpt_idx == 0 % By convention this is the mean DVF over time
        filename = sprintf('%s\\DVF temporal mean %s - %s', path_par.temp_im_dir, path_par.input_im_dir_suffix, OF_param_str);
    else
        filename = sprintf('%s\\DVF %d-th component scale_factor=%d %s - %s', path_par.temp_im_dir, pca_cpt_idx, disp_par.PCA_cp_scale_factor, ...
                                                                                                            path_par.input_im_dir_suffix, OF_param_str);
    end
    
    if beh_par.CROP_FOR_DISP_SAVE
        filename = sprintf('%s ROI', filename);
    end

    filename = sprintf('%s.jpg', filename);

end