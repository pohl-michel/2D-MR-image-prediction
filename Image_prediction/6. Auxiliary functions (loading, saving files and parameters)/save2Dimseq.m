function save2Dimseq(im_par, path_par, disp_par, beh_par)
% Saves jpg images from the image sequence in path_par.input_im_dir
%
% Author : Pohl Michel
% Date : Sept 18th, 2022
% Version : v1.0
% License : 3-clause BSD License


    enhance_flag = false;  

    for t=1:im_par.nb_im

        % chargement de l'image à t
        crop_flag = false; filter_flag = false;
        sag_slice_t = load_crop_filter2D(t, crop_flag, filter_flag, 0.0, im_par, path_par.input_im_dir);
        
        % saving jpg images of the cross sections
        crop_flag = beh_par.CROP_FOR_DISP_SAVE;
        im_filename_suffix = sprintf('%s image%d', path_par.input_im_dir_suffix, t);
        save_crop_enhance_2Dim_jpg(sag_slice_t, im_filename_suffix, crop_flag, enhance_flag, disp_par, path_par, im_par.x_m, im_par.x_M, im_par.y_m, im_par.y_M, t);
  
    end
    
end
