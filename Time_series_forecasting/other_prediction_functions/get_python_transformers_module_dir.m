function moduleDir = get_python_transformers_module_dir()

    cwd = pwd;
    if endsWith(cwd, 'Time_series_forecasting')
        moduleDir = fullfile(cwd, 'other_prediction_functions');
    else % we are calling from the root project (2D-MR-image-prediction or other name if it was renamed)
        moduleDir = fullfile(cwd, 'Time_series_forecasting', 'other_prediction_functions');
    end

end