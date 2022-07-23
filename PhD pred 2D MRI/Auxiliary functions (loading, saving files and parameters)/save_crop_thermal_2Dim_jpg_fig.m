function save_crop_thermal_2Dim_jpg_fig(im, im_filename_suffix, crop_flag, disp_par, path_par, x_m, x_M, y_m, y_M)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

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

