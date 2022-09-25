function save_crop_enhance_2Dim_jpg(im, im_filename_suffix, crop_flag, enhance_flag, disp_par, path_par, x_m, x_M, y_m, y_M, t)
% Saves the 2D image "im" after enhancing it and cropping it if specified.
%
% Author : Pohl Michel
% Date : Sept 18th, 2022
% Version : v1.1
% License : 3-clause BSD License


    f = figure;
    im = enhance_2Dim( im, enhance_flag);
    
    if crop_flag
        imshow(im(y_m:y_M, x_m:x_M), []);
        im_filename_suffix = sprintf('%s ROI', im_filename_suffix);
    else
        imshow(im, []);
    end
    
    if disp_par.write_im_time
        text(4, 4, num2str(t), 'Color', 'w', 'FontSize', disp_par.im_time_fontsize, 'FontWeight', 'bold') %revoir pt_idx_fontsize
    end
        
    im_filename = sprintf('%s//%s.jpg', path_par.temp_im_dir, im_filename_suffix);
    
    set(gca,'position',[0 0 1 1],'units','normalized');
    print(im_filename, '-djpeg', disp_par.wrp_im_res);
    close(f);
        
end

