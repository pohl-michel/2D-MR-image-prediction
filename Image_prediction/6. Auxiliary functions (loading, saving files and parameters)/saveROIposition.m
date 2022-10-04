function saveROIposition(im_par, path_par, disp_par)
% Saves the first image of the 2D sequence with a red rectangle around the region of interest (ROI)
% with resolution disp_par.wrp_im_res (even though it does not correspond to a warped image)
%
% Author : Pohl Michel
% Date : Oct. 5, 2022
% Version : v1.0
% License : 3-clause BSD License

    t_init = 1;
    crop_flag = false; initial_filtering_flag = false; 
    Iinit = load_crop_filter2D(t_init, crop_flag, initial_filtering_flag, 0, im_par, path_par.input_im_dir);

    f = figure;
    imshow(Iinit, []);
    set(gcf, 'units','normalized','outerposition',[0 0 1 1]);

    % draws rectangle on cropped area
    ROIx_size = im_par.x_M - im_par.x_m + 1;
    ROIy_size = im_par.y_M - im_par.y_m + 1;    
    rectangle('Position',[im_par.x_m, im_par.y_m, ROIx_size, ROIy_size],'EdgeColor','r','LineWidth', disp_par.ROI_rectangle_linew)

    fprintf('saving position of ROI image \n');
    filename = sprintf('%s\\%s %s', path_par.temp_im_dir, path_par.input_im_dir_suffix, 'ROI_position_tinit');
    set(gca,'position',[0 0 1 1],'units','normalized');
    print(filename, '-dpng', disp_par.wrp_im_res);
    close(f);


end

