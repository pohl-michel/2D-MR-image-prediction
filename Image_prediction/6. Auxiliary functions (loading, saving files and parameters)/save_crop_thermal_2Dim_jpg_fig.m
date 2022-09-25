function save_crop_thermal_2Dim_jpg_fig(im, im_filename_suffix, crop_flag, disp_par, path_par, x_m, x_M, y_m, y_M)
% Saves the image im as a thermal image
% The size of the bar on the side of the image can be specified in disp_par.thermaldiff_fontsize
% If crop_flag is set to true, the image is cropped before saving.
%
% Author : Pohl Michel
% Date : Sept 18th, 2022
% Version : v1.0
% License : 3-clause BSD License


    f = figure;
        
    if crop_flag
        im = im(y_m:y_M, x_m:x_M);
        im_filename_suffix = sprintf('%s ROI', im_filename_suffix);
    end

    pcolor(flipud(im))
    shading interp
    colormap(jet)
	
	if isfield(disp_par, 'thermaldiff_fontsize')
		cH = colorbar;
		set(cH,'FontSize', disp_par.thermaldiff_fontsize);
	else
		colorbar
	end
    
    fig_filename = sprintf('%s\\%s.fig', path_par.temp_fig_dir, im_filename_suffix);
    savefig(f, fig_filename);
    set(gca,'XTick',[], 'YTick', [])
    png_filename = sprintf('%s\\%s.png', path_par.temp_im_dir, im_filename_suffix);
    print(png_filename, '-dpng', disp_par.wrp_im_res);    
        % here we could change the resolution
    
    close(f);    
    
end

