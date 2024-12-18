function [Ypred, avg_pred_time, pred_loss_function] = svr_pred(pred_par, X, Y)
% prediction with support vector regression (one model per output vector)
% X: past data matrix 
% Y: future data matrix
% Note: the signal history length / time lag hyperparameter is already taken into account in the construction
% of the X and Y matrices inside load_pred_data_XY()

    % SVR is not an online method so in that case the variable M in load_pred_data_XY does not represent the number of predictions
    idx_max = pred_par.tmax_training-pred_par.SHL-pred_par.horizon+1; % similar to M+k+h-1 = pred_par.tmax_pred in load_pred_data_XY.m
    [~, M] = size(X);
    nb_predictions = M-idx_max; % also equal to pred_par.tmax_pred - pred_par.tmax_training

    % Transposing matrices because 
    X_train = X(:, 1:idx_max)'; 
    Y_train = Y(:, 1:idx_max)';
    X_test = X(:, idx_max+1:end)';
    Y_test = Y(:, idx_max+1:end)';

    % Train kernel SVR models for each dimension
    svrModels = cell(1, size(Y_train, 2));
    for dim = 1:size(Y_train, 2)
        svrModels{dim} = fitrsvm(X_train, Y_train(:, dim), ...
            'KernelFunction', 'rbf', ...
            'KernelScale', pred_par.svr_kernel_scale, ...
            'BoxConstraint', pred_par.svr_box_constraint, ...
            'Epsilon', pred_par.svr_epsilon,  ...
            'Standardize', false);
    end

    % Forecast the next steps
    Ypred = zeros(size(Y_test));

    tic
    for dim = 1:size(Y_test, 2)
        Ypred(:, dim) = predict(svrModels{dim}, X_test);
    end
    ttl_pred_time = toc;

    % for debugging - finding quickly a first set of parameter that can work
    % mse = mean((Y_test - Ypred).^2, 'all');
    % disp(['Mean Squared Error: ', num2str(mse)]);

    Ypred = Ypred';
    avg_pred_time = ttl_pred_time/nb_predictions;
    pred_loss_function = transpose(sum((Ypred - Y(:,(1 + idx_max):end)).^2, 1));

end