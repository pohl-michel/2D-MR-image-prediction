function [Ypred, avg_pred_time, pred_loss_function] = transformer_pred(path_par, pred_par, X, Y, run_idx)
% prediction with a tranformer encoder model followed by a feed-forward layer.
% This function is a wrapper around the train_and_predict and population_model_predict functions in transformers_forecasting.py
% X: past data matrix 
% Y: future data matrix
% Note: the signal history length / time lag hyperparameter is already taken into account in the construction
% of the X and Y matrices inside load_pred_data_XY()

    % Not an online method so in that case the variable M in load_pred_data_XY does not represent the number of predictions
    idx_max = pred_par.tmax_training-pred_par.SHL-pred_par.horizon+1; % similar to M+k+h-1 = pred_par.tmax_pred in load_pred_data_XY.m
    [n_features, M] = size(Y); % M is the number of samples

    if pred_par.GPU_COMPUTING
        pred_par.selected_device = "cuda";
    else
        pred_par.selected_device = "cpu";
    end    

    if strcmp(pred_par.pred_meth, "population_transformer")
        % Get the configuration corresponding to the horizon and run index of interest
        config_path = get_most_recent_transformer_model_config(path_par.temp_var_dir, pred_par.horizon, run_idx);
        
        % set the configuration path in pred_par, so that Python knows which transformer model to load
        pred_par.config_path = config_path;  % Rk: the transformer config. (e.g., nb. of layers) is the same for all runs and horizons
    end

    % Converting the structure containing the transformer parameters to a Python dictionary.
    pred_par_py = convert_to_py_dict(pred_par);

    % Reshaping the data matrix for use by the Python transformer code
    X_reshaped = reshape_data_matrix(X, n_features, pred_par.SHL, M);

    % Getting the input test data and converting it to a Python format
    X_test = X_reshaped(idx_max+1:end, :, :);
    X_test_py = py.numpy.array(X_test);

    switch(pred_par.pred_meth)

        case 'transformer'  % sequence-specific transformer

            % Separating data for training and testing (or cross-validation)
            X_train = X_reshaped(1:idx_max, :, :); 
            Y_train = Y(:, 1:idx_max)';
            Y_test = Y(:, idx_max+1:end)';
        
            % Converting the training and test data into a numpy array
            X_train_py = py.numpy.array(X_train);
            Y_train_py = py.numpy.array(Y_train);
            Y_test_py = py.numpy.array(Y_test);
        
            % Performing training and inference using a sequence-specific transformer encoder, using pytorch
            results = py.transformers_forecasting.train_and_predict(pred_par_py, X_train_py, Y_train_py, X_test_py, Y_test_py);

        case 'population_transformer'  % population transformer    

            % Performing inference using a transformer encoder (loading the model trained offline on a population dataset)
            results = py.transformers_forecasting.population_model_predict(pred_par_py, X_test_py, int32(run_idx));            
    end

    % Converting back the results into Matlab types
    [Ypred, avg_pred_time] = unpack_python_prediction(results);

    % Transposing the predicted values and computing the loss (data still normalized if NORMALIZE_DATA was set to true)
    Ypred = Ypred';
    pred_loss_function = transpose(sum((Ypred - Y(:,(1 + idx_max):end)).^2, 1));

end

function [Y_pred, avg_time] = unpack_python_prediction(results)
% Converting the results from the train_and_predict() Python function into Matlab objects
    py_preds = results{1};
    py_avg_time = results{2};

    Y_pred = double(py_preds);
    avg_time = double(py_avg_time);
end

function X_reshaped = reshape_data_matrix(X, n_features, shl, samples)
% Converts a data matrix of shape (1+shl * n_features, samples), with an initial row of ones,
% into a matrix of shape (samples, shl, n_features) as expected by the Python code for transformers 

    % Remove the first row (the bias or constant 1s row)
    X_no_bias = X(2:end, :);  % the shape is now [seq_length * n_features, samples]
    
    % Reshape into (n_features, shl, samples)
    X_reshaped = reshape(X_no_bias, [n_features, shl, samples]);
    
    % Permute to get (samples, shl, n_features)
    X_reshaped = permute(X_reshaped, [3, 2, 1]); 

end

function pred_par_py = convert_to_py_dict(pred_par)
% Converts pred_par into a Python dictionary, which includes
% casting pred_par values to types that can be understood in Python

    % List of fields whose values should be converted to int32 or double
    int_fields = ["batch_size", "num_epochs", "SHL", "d_model", ...
                  "nhead", "num_layers", "dim_feedforward", "final_layer_dim", ...
                  "horizon", "seq_length", "n_features"];
    float_fields = ["dropout", "learn_rate"];
    string_fields = ["selected_device", "config_path"];  % Add string fields

    % Perform value type conversion to int32 or double
    fields = fieldnames(pred_par);
    for i = 1:numel(fields)
        f = fields{i};
        if ismember(f, int_fields)
            % Handle zero/empty values by converting to Python None
            if isempty(pred_par.(f)) || pred_par.(f) == 0
                pred_par.(f) = py.None;  % Convert to Python None
            else
                pred_par.(f) = int32(pred_par.(f));
            end
        elseif ismember(f, float_fields)
            pred_par.(f) = double(pred_par.(f));
        elseif ismember(f, string_fields)
            pred_par.(f) = string(pred_par.(f));
        end
    end

    % Converts the Matlab structure to a Python dictionary
    pred_par_py = py.dict(pred_par);

end