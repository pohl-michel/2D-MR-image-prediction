function moduleDir = get_python_transformers_module_dir()

    cwd = pwd;
    if endsWith(cwd, 'MRI_prediction')
        moduleDir = fullfile(cwd, 'Time_series_forecasting', 'other_prediction_functions');
    else
        moduleDir = fullfile(cwd, 'other_prediction_functions');
    end

end