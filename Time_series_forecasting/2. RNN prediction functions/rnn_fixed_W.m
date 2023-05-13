function [myRNN] = rnn_fixed_W(myRNN, pred_par, Xdata, Ydata)
% rnn_RTRL performs the training and prediction of a recurrent neural network (RNN) whose weights Wa and Wb are fixed (only Wc is learned online).
% This is a degenerate case which helps provide a lower bound on the performance of RNN learning algorithms.
%
% Input variables :
%   - myRNN : RNN structure previously initialized by the function "initialize_rnn"
%   - pred_par is the structure containing the parameters used for prediction
%   - Xdata is the matrix of the "past data" and Ydata the matrix of the "future" data given by the function "load_pred_data_XY".
% Output variables :
%   - myRNN is the updated RNN structure containing in particular :
%       - the predicted time series "myRNN.Ypred"
%       - the values of the loss function "myRNN.pred_loss_function"
%       - the array containing the time for making each prediction "myRNN.pred_time_array"
%
% Author : Pohl Michel
% Date : November 14, 2022
% Version : v1.0
% License : 3-clause BSD License	

    [~, M] = size(Xdata);
    
    for t=1:M

        tic
        
        % Forward propagation (prediction) and calculation of the instantaneous prediction error        
        u = Xdata(:,t); % input vector of size m+1
        [~, new_x] = RNN_state_fwd_prop(myRNN, u, myRNN.x);
            % we need z to compute myRNN.dtheta
        myRNN.Ypred(:,t) = myRNN.Wc*new_x; 
            % here we do not use new_x. myRNN.x is updated at the end of the loop
        e = Ydata(:,t) - myRNN.Ypred(:,t); 

        % gradient with respect to the output parameters dL/dWc
        dWc = -e*(new_x.');

        % line vector containing the concatenation of Wa, Wb and Wc
        Wc_new = update_param_optim(myRNN.Wc, dWc, pred_par, myRNN.grad_moments, t);
        myRNN.Wc = Wc_new;

        % States update
        myRNN.x = new_x;
        
        myRNN.pred_time_array(t) = toc;
        myRNN.pred_loss_function(t) = 0.5*(e.')*e;  % error evaluation so it is performed after time performance evaluation
        
    end  

end