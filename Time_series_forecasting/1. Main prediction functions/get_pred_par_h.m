function pred_par_h = get_pred_par_h(pred_par, hppars, hrz_idx, path_par)
% Get an updated pred_par structure where the horizon corresponds to hrz_idx in hppars.horizon_tab and the SHL corresponds to the correct 
% transformer model. This function is used mainly when iterating over different horizons to assess the performance of forecasting algorithms

    pred_par_h = pred_par;
    crt_horizon = hppars.horizon_tab(hrz_idx);
    pred_par_h.horizon = crt_horizon;
    if strcmp(pred_par_h.pred_meth, "population_transformer")
        % updating pred_par_h to load the SHL in the transformer config (so that data is loaded correctly in load_pred_data_XY())
        pred_par_h = update_pred_par_with_transformer_config(path_par, pred_par_h, crt_horizon);
    end

end