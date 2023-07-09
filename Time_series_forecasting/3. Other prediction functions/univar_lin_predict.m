function [Ypred, avg_pred_time, pred_loss_function] = univar_lin_predict(pred_par, X, Y)
% prediction with univariate linear prediction
% X: past data matrix 
% Y: future data matrix
% 
% Author : Pohl Michel
% Date : September 27, 2021
% Version : v1.0
% License : 3-clause BSD License

    [p, M] = size(Y);
    idx_max = pred_par.tmax_training-pred_par.SHL-pred_par.horizon+1;
    nb_predictions = M-idx_max;

    pred_time_tab = zeros(nb_predictions,1);
    Ypred = zeros(p, nb_predictions);

    Xi = zeros(1+pred_par.SHL, M);

    for i=1:p
        % loop over each variable to be predicted
        
        for j=1:pred_par.SHL
            Xi(1+j,:) = X(1+i+(j-1)*p, :);
        end
        
        Yi = Y(i,:); 
        
        Xitrain = Xi(:,1:idx_max);
        Yitrain = Yi(:,1:idx_max);
        
        Ai = Yitrain*(Xitrain.')*pinv(Xitrain*(Xitrain.'));
            
        tic
        Ypred(i,:) = Ai*Xi(:,(1 + idx_max):end);
        pred_time_tab(i) = toc;
        
        Xi(:) = 0;
        
    end
    
    avg_pred_time = sum(pred_time_tab, 1)/nb_predictions;
    pred_loss_function = transpose(sum((Ypred - Y(:,(1 + idx_max):end)).^2, 1));

end