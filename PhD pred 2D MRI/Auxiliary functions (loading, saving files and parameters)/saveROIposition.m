function saveROIposition(im_par, path_par, disp_par)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

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
    filename = sprintf('%s\\%s', path_par.temp_im_dir, 'ROI_position_tinit');
    set(gca,'position',[0 0 1 1],'units','normalized');
    print(filename, '-dpng', '-r300');
    close(f);


end

