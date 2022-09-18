function plot_pred_error( RNNloss, disp_par, pred_par, path_par, filename_suffix, title_str, SAVE_ONLY )
% Plots the instantaneous error function as a function of time.
% 
% Author : Pohl Michel
% Date : August 12th, 2020
% Version : v1.1
% License : 3-clause BSD License

    switch pred_par.data_type
        case 1 % prediction of the 3D position of markers 
            my_ylabel = 'Loss function (mm^2)';
        case 2 % prediction of other signals (for instance weights of the PCA of the DVF)
            my_ylabel = 'Loss function (no unit)';
    end


    f = figure; %f = figure('units','normalized','outerposition',[0 0 1 1]); % for saving figures full screen
    plot(RNNloss, 'Color', 'k')
    
    % Uncomment the following code to display a vertical line corresponding to pred_par.t_eval_start
        %t_start_pred = pred_par.t_eval_start;
        %line([t_start_pred, t_start_pred], get(gca, 'ylim'), 'Color', [1 0 0], 'linewidth', disp_par.pred_start_linewidth);
        %yl = ylim();
        %text(t_start_pred, yl(2)*0.9, {'Test data'})
    
    title(title_str); % I can suppress the title later if necessary
    ylabel(my_ylabel);
    xlabel('Time index');
    
    fig_filename = sprintf('%s\\%s.fig', path_par.temp_fig_dir, filename_suffix);
    savefig(f, fig_filename);
    png_filename = sprintf('%s\\%s.png', path_par.temp_im_dir, filename_suffix);
    print(png_filename, '-dpng', disp_par.pred_plot_res);
    
    if SAVE_ONLY
       close(f); 
    end
    
end
