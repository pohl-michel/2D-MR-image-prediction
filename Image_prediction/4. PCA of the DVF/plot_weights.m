function plot_weights( F, OF_par, path_par, disp_par)
% Plots the weights associated with the kth principal DVF (or principal components
% 
% Author : Pohl Michel
% Date : August 12th, 2020
% Version : v1.0
% License : 3-clause BSD License

    [tmax, nb_cp] = size(F);

    for cp_idx = 1:nb_cp
    
        current_weights = F(:, cp_idx);

        f = figure;
        %f = figure('units','normalized','outerposition',[0 0 1 1]); % for saving figures full screen
        plot(1:tmax, current_weights, 'Color', 'k')
        ylabel(sprintf('Weight associated with the %d-th principal DVF', cp_idx));
        xlabel('time index');

        filename_suffix = sprintf('%s %d-th weights %s', path_par.input_im_dir_suffix, cp_idx, sprintf_OF_param(OF_par));
        fig_filename = sprintf('%s\\%s.fig', path_par.temp_fig_dir, filename_suffix);
        savefig(f, fig_filename);
        jpg_filename = sprintf('%s\\%s.jpg', path_par.temp_im_dir, filename_suffix);
        print(jpg_filename, '-djpeg', disp_par.pred_plot_res);

        close(f); 

    end
    
end

