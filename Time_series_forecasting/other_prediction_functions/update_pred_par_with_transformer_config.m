function pred_par = update_pred_par_with_transformer_config(path_par, pred_par, horizon)
% Loads the most recent config.json file for the transformer model corresponding to a specific horizon and run index
% The prediction parameters in pred_par are then updated using that configuration file (this is used to load data proprely with the correct SHL)
% Rk: I could have run_idx = 1 as default when not specified

    % there will always be a run number 1, regardless of the maximum number of runs, so we load that config 
    run_idx = 1; 
    
    config_path = get_most_recent_transformer_model_config(path_par, pred_par, horizon, run_idx);
    config_str = fileread(config_path);
    config = jsondecode(config_str);
    
    % Get all field names from config.config
    field_names = fieldnames(config.config);
    
    % Loop through each field and copy to pred_par
    % we need shl to load input data correctly, the nb of runs, 
    % and other params for logging, e.g., with sprintfpred_par()
    for i = 1:length(field_names)
        field_name = field_names{i};
        pred_par.(field_name) = config.config.(field_name);
    end
    
end