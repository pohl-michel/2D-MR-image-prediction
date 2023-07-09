function [myRNN] = rnn_RTRLv2(myRNN, pred_par, Xdata, Ydata)
% rnn_RTRL performs the training and prediction of a recurrent neural network (RNN) trained with real-time recurrent learning (RTRL) and gradient clipping.
% This formulation of RTRL is the "standard" formulation coming directly from the two expressions of dL/dtheta and dx/dtheta (influence matrix).
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
% Date : November 1st, 2022
% Version : v1.0
% License : 3-clause BSD License	

    [~, M] = size(Xdata);
    m = myRNN.input_space_dim;
    q = myRNN.state_space_dim;
    p = myRNN.output_space_dim;
    nb_weights = myRNN.nb_weights; %(size of the theta vector)    

    size_Wa = q*q;
    size_Wb = q*(m+1);
    size_Wc = p*q;
   
    idx_min_Wc = size_Wa + size_Wb + 1;    

    if pred_par.GPU_COMPUTING
        Jind = 1:(q*(q + m + 1));
        Iind = mod(Jind - 1, q) + 1;
    end
    
    for t=1:M

        tic
        
        % Forward propagation (prediction) and calculation of the instantaneous prediction error        
        u = Xdata(:,t); % input vector of size m+1
        [z, new_x] = RNN_state_fwd_prop(myRNN, u, myRNN.x);
            % we need z to compute myRNN.dtheta
        myRNN.Ypred(:,t) = myRNN.Wc*new_x; 
            % here we do not use new_x. myRNN.x is updated at the end of the loop
        e = Ydata(:,t) - myRNN.Ypred(:,t); 

        % gradient with respect to the output parameters dL/dWc
        myRNN.dtheta(:, idx_min_Wc:nb_weights) = reshape(-e*(new_x.'), [1, p*q]); 

        % computation of Dt = dFst/dx (dynamic matrix)
        phi_prime_z = myRNN.phi_prime(z);
        Dt = phi_prime_z.*myRNN.Wa; 
        
        % computation of It = dFst/dtheta (immediate jacobian)
        It_compact = phi_prime_z*[(myRNN.x).', u.'];
        
        if pred_par.GPU_COMPUTING
            myRNN.It = full(sparse(Iind, Jind, It_compact));
            % https://www.mathworks.com/matlabcentral/answers/1840693-sparse-matrix-from-the-columns-of-an-initial-square-matrix
            % use myRNN.It = full(sparse(Iind, Jind, double(It_compact))); if there the following error occurs: "double" to solve "Sparse gpuArrays supports only double precision data"
        else
            % the method above without a for loop also work well here without GPU
            for k = 1:(q+m+1)
                myRNN.It(:, ((k-1)*q + 1):k*q) = diag(It_compact(:, k));
            end
        end
        
        % update of the influence matrix Jt = dx/dtheta (with theta from the elements in [Wa, Wb] here)
        myRNN.Jt = myRNN.It + Dt*myRNN.Jt;
        
        % gradient of the loss with respect to the states dL/dx = dL/dy * Fout/dx
        dx = -(e.') * myRNN.Wc; 

        % gradient of the loss function with respect to the weights Wa and Wb
        myRNN.dtheta(:, 1:(size_Wa+size_Wb)) = dx*myRNN.Jt;
        
        % Weight updates using optimization methods such as SGD
        theta_vec =[reshape(myRNN.Wa, [1, size_Wa]), reshape(myRNN.Wb, [1, size_Wb]), reshape(myRNN.Wc, [1, size_Wc])];
            % line vector containing the concatenation of Wa, Wb and Wc
        new_theta = update_param_optim(theta_vec, myRNN.dtheta, pred_par, myRNN.grad_moments, t);
        myRNN.Wa = reshape(new_theta(:,1:size_Wa), [q, q]);
        myRNN.Wb = reshape(new_theta(:,(1 + size_Wa):(size_Wa + size_Wb)), [q, m+1]);
        myRNN.Wc = reshape(new_theta(:,idx_min_Wc:nb_weights), [p, q]);

        % States update
        myRNN.x = new_x;
        
        myRNN.pred_time_array(t) = toc;
        myRNN.pred_loss_function(t) = 0.5*(e.')*e;  % error evaluation so it is performed after time performance evaluation
        
    end  

    if pred_par.GPU_COMPUTING
        myRNN.Ypred = gather(myRNN.Ypred);
        myRNN.pred_time_array = gather(myRNN.pred_time_array);
        myRNN.pred_loss_function = gather(myRNN.pred_loss_function);
    end  

end