function save2DOFjpg(beh_par, path_par, disp_par, OF_par, im_par)
% Loads the optical flow previously calculated by compute_2Dof and displays it on top of the initial image at t=1.
% The DVF images are saved in the folder path_par.im_par_filename if beh_par.SAVE_OF_JPG is set to true
%
% Author : Pohl Michel
% Date : September 25th, 2022
% Version : v2.0
% License : 3-clause BSD License

    
    % loading image at time t=1
    im_t1 = enhance_2Dim(load_crop_filter2D(1, beh_par.CROP_FOR_DISP_SAVE, false, 0, im_par, path_par.input_im_dir), true);    
        % the image is cropped if we select a specific region where the optical flow should be displayed (beh_par.CROP_FOR_DISPLAY = 1)
        % The image is not filtered because it is used to display the optical flow here (FILTER = false)
    
    G = return_mask_for_DVF_display(im_t1, disp_par);

    for t=2:im_par.nb_im
        OF_t_filename = write_2DOF_t_mat_filename( OF_par, path_par, t );
        load(OF_t_filename, 'u_t');
        OF_t_png_filename = write_2DOF_t_png_filename(beh_par, OF_par, path_par, t);
        display_save_DVF(u_t, im_t1, G, OF_t_png_filename, im_par, beh_par, disp_par, beh_par.SAVE_OF_JPG);    
    end
    
end


function [ filename ] = write_2DOF_t_png_filename( beh_par, OF_par, path_par, t )
% Returns the name of the png/jpg file containing the optical flow between t=1 and t computed with the parameters specified in OF_par
%
% Author : Pohl Michel
% Date : July 16th, 2020
% Version : v1.0
% License : 3-clause BSD License


    OF_param_str = sprintf_OF_param(OF_par);
    filename = sprintf('%s\\2DOF %s t=1 t=%d - %s', path_par.temp_im_dir, path_par.input_im_dir_suffix, t, OF_param_str);
    if beh_par.CROP_FOR_DISP_SAVE
        filename = sprintf('%s ROI', filename);
    end
    filename = sprintf('%s.jpg', filename);

end