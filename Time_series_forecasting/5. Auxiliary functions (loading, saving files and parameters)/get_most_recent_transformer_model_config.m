function config_path = get_most_recent_transformer_model_config(path_par, pred_par, horizon, run_idx)
% GET_MOST_RECENT_MODEL_CONFIG Find the most recent transformer model config file
%
% INPUTS:
% - path_par: path parameters (not needed anymore but I keep it just in case it's needed later and because I'm lazy to clean up) 
% - pred_par: prediction parameters
% - horizon: Horizon value (e.g., 2 for horizon_2 folder)
% - run_idx: run index of the transformer model whose configuration we are loading
%
% OUTPUT:
% - config_path: Full path to the most recent config file

    % Construct the horizon folder path
    horizon_folder = fullfile(pred_par.models_dir, sprintf('horizon_%d', horizon));
    
    % If we are in the in the MRI_prediction folder, add the time series forecasting base folder
    if any(strcmp({dir(pwd).name}, 'Time_series_forecasting'))
        horizon_folder = fullfile('Time_series_forecasting', horizon_folder);
    end

    % Check if folder exists
    if ~exist(horizon_folder, 'dir')
        error('Horizon folder does not exist: %s', horizon_folder);
    end
    
    % Get all config files matching the pattern
    pattern = sprintf('transformer_h%d_model%d*_config.json', horizon, run_idx);
    files = dir(fullfile(horizon_folder, pattern));
    
    if isempty(files)
        error('No config files found in %s matching pattern %s', horizon_folder, pattern);
    end
    
    % Extract date and time from filenames and find the most recent
    most_recent_datetime = 0;
    most_recent_file = '';
    
    for i = 1:length(files)
        filename = files(i).name;
        
        % Extract date and time using regular expression
        % Pattern: transformer_h2_model1_YYYYMMDD_HHMMSS_config.json
        pattern = sprintf('transformer_h%d_model\\d+_(\\d{8})_(\\d{6})_config\\.json', horizon);
        tokens = regexp(filename, pattern, 'tokens');        

        if ~isempty(tokens)
            date_str = tokens{1}{1}; % YYYYMMDD
            time_str = tokens{1}{2}; % HHMMSS
            
            % Convert to datetime for comparison
            datetime_str = [date_str, time_str]; % YYYYMMDDHHMMSS
            current_datetime = str2double(datetime_str);
            
            if current_datetime > most_recent_datetime
                most_recent_datetime = current_datetime;
                most_recent_file = filename;
            end
        end
    end
    
    if isempty(most_recent_file)
        error('No valid config files found with expected naming pattern');
    end
    
    % Return full path to the most recent config file
    config_path = fullfile(horizon_folder, most_recent_file);
    
    fprintf('Found most recent model: %s\n', most_recent_file);
end