function plot_pred_signal( pred_data, org_data, pred_par, path_par, disp_par)
% Plots the weights associated with the k-th principal component/ principal deformation field
% 
% Author : Pohl Michel
% Date : September 27, 2021
% Version : v1.0
% License : 3-clause BSD License


    [nb_cp, tmax] = size(org_data);

    for cp_idx = 1:nb_cp
    
        current_weights = org_data(cp_idx, :);
        tpred_plot_start = 1 + pred_par.tmax_training;
        pred_weights = pred_data(cp_idx, tpred_plot_start:tmax);

        f = figure;
        %f = figure('units','normalized','outerposition',[0 0 1 1]); % for saving figures full screen
        
        plot(tpred_plot_start:tmax, pred_weights, 'x', 'Color', 'k')
        hold on
        plot(1:tmax, current_weights, 'Color', 'k')
        
        ylabel(sprintf('Coefficient associated with the %d-th principal component', cp_idx));
        xlabel('time index');
        legend({'predicted signal','original signal'},'Location','southwest')

        filename_suffix = sprintf('%s %d-th weight prediction %s %s', path_par.input_im_dir_suffix, cp_idx, pred_par.pred_meth_str, sprintf_pred_param(pred_par));
        fig_filename = sprintf('%s\\%s.fig', path_par.temp_fig_dir, filename_suffix);
        savefig(f, fig_filename);
        jpg_filename = sprintf('%s\\%s.jpg', path_par.temp_im_dir, filename_suffix);
        print(jpg_filename, '-djpeg', disp_par.pred_plot_res);

        close(f); 

    end
    
end

