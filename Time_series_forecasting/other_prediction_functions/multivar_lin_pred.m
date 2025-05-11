function [Ypred, avg_pred_time, pred_loss_function] = multivar_lin_pred(pred_par, X, Y)
% prediction with multivariate linear prediction
% X: past data matrix 
% Y: future data matrix
% 
% Author : Pohl Michel
% Date : September 11th, 2021
% Version : v1.0
% License : 3-clause BSD License


    % Linear regression is not an online method so in that case the variable M in load_pred_data_XY does not represent the number of predictions
    idx_max = pred_par.tmax_training-pred_par.SHL-pred_par.horizon+1; % similar to M+k+h-1 = pred_par.tmax_pred in load_pred_data_XY.m
    [~, M] = size(X);
    nb_predictions = M-idx_max; % also equal to pred_par.tmax_pred - pred_par.tmax_training
    
    Xtrain = X(:,1:idx_max);
    Ytrain = Y(:,1:idx_max);
    
    A = Ytrain*(Xtrain.')*pinv(Xtrain*(Xtrain.'));
    
    tic
    Ypred = A*X(:,(1 + idx_max):end); % size (~, nb_predictions) 
    ttl_pred_time = toc;
    avg_pred_time = ttl_pred_time/nb_predictions;
    
    pred_loss_function = transpose(sum((Ypred - Y(:,(1 + idx_max):end)).^2, 1));

end