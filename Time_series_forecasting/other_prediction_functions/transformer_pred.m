function [Ypred, avg_pred_time, pred_loss_function] = transformer_pred(pred_par, X, Y)
% prediction with a tranformer encoder model followed by a feed-forward layer.
% This function is merely a wrapper around the train_and_predict function in transformers_forecasting.py
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
    
    % Converting the structure containing the transformer parameters to a Python dictionary.
    pred_par_py = convert_to_py_dict(pred_par);

    % Reshaping the data matrix for use by the Python transformer code
    X_reshaped = reshape_data_matrix(X, n_features, pred_par.SHL, M);

    % Separating data for training and testing (or cross-validation)
    X_train = X_reshaped(1:idx_max, :, :); 
    Y_train = Y(:, 1:idx_max)';
    X_test = X_reshaped(idx_max+1:end, :, :);
    Y_test = Y(:, idx_max+1:end)';

    % Converting the training and test data into a numpy array
    X_train_py = py.numpy.array(X_train);
    Y_train_py = py.numpy.array(Y_train);
    X_test_py = py.numpy.array(X_test);
    Y_test_py = py.numpy.array(Y_test);

    % Performing prediction using a transformer encoder, using pytorch, and converting back the results into Matlab types
    results = py.transformers_forecasting.train_and_predict(pred_par_py, X_train_py, Y_train_py, X_test_py, Y_test_py);
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
                  "nhead", "num_layers", "dim_feedforward", "final_layer_dim"];
    float_fields = ["dropout", "learn_rate"];

    % Perform value type conversion to int32 or double
    fields = fieldnames(pred_par);
    for i = 1:numel(fields)
        f = fields{i};
        if ismember(f, int_fields)
            pred_par.(f) = int32(pred_par.(f));
        elseif ismember(f, float_fields)
            pred_par.(f) = double(pred_par.(f));
        end
    end

    % Converts the Matlab structure to a Python dictionary
    pred_par_py = py.dict(pred_par);

end