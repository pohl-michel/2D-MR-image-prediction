function plot_pred_signal( pred_data, org_data, pred_par, path_par, disp_par, SAVE_ONLY)
% Plots the predicted signals (either predicted positions of objects or predicted PCA signals) 
% 
% Author : Pohl Michel
% Date : September 11, 2022
% Version : v2.0
% License : 3-clause BSD License


    [nb_cp, ~] = size(org_data);

    switch pred_par.data_type
        case 1 % prediction of the 3D position of markers 
            my_legend.pred = 'predicted coordinate';
            my_legend.org = 'original coordinate';
        case 2 % prediction of other signals (for instance weights of the PCA of the DVF)
            my_legend.pred = 'predicted signal';
            my_legend.org = 'original signal';
    end

    for cp_idx = 1:nb_cp

        switch pred_par.data_type
            case 1 % prediction of the 3D position of markers 
                obj_idx = mod(cp_idx, pred_par.dim_per_obj); 
                crt_dim = 1 + floor(cp_idx/pred_par.dim_per_obj);
                switch crt_dim 
                    case 1
                        dir_char = 'x';
                    case 2
                        dir_char = 'y';
                    case 3
                        dir_char = 'z';
                end
                my_ylabel = sprintf('%s coordinate', dir_char);
                filename_suffix = sprintf('%s pred %s %d-th pt %s %d-th run', path_par.time_series_dir, dir_char, obj_idx, sprintf_pred_param(pred_par));
            case 2 % prediction of other signals (for instance weights of the PCA of the DVF)
                my_ylabel = sprintf('Coefficient associated with the %d-th principal component', cp_idx);
                filename_suffix = sprintf('%s %d-th weight prediction %s %s', path_par.input_im_dir_suffix, cp_idx, pred_par.pred_meth, sprintf_pred_param(pred_par));
        end
        
        org_signal = org_data(cp_idx, :);
        tpred_plot_start = 1 + pred_par.SHL + pred_par.horizon; % also possible: tpred_plot_start = 1 + pred_par.tmax_training;
        pred_signal = pred_data(cp_idx, tpred_plot_start:pred_par.tmax_pred);
    
        f = figure; % to save full screen: f = figure('units','normalized','outerposition',[0 0 1 1]); 
    
        plot(tpred_plot_start:pred_par.tmax_pred, pred_signal, 'x', 'Color', 'k')
        hold on
        plot(1:pred_par.tmax_pred, org_signal, 'Color', 'k')
        
        ylabel(my_ylabel);
        xlabel('time index');
        legend({my_legend.pred,my_legend.org},'Location','southwest')
    
        fig_filename = sprintf('%s\\%s.fig', path_par.temp_fig_dir, filename_suffix);
        savefig(f, fig_filename);
        jpg_filename = sprintf('%s\\%s.jpg', path_par.temp_im_dir, filename_suffix);
        print(jpg_filename, '-djpeg', disp_par.pred_plot_res);
    
        if SAVE_ONLY
           close(f); 
        end

    end
    
end