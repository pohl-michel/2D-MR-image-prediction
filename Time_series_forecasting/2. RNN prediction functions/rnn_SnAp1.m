function [myRNN] = rnn_SnAp1(myRNN, pred_par, Xdata, Ydata)
% rnn_SnAp1 performs the training of a recurrent neural network (RNN) trained with SnAp-1 (in the case where the influence matrix is initialized to 0) and gradient clipping.
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
% Date : October 31st, 2022
% Version : v1.1
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
    
    for t=1:M

        tic
        
        % Forward propagation (prediction) and calculation of the instantaneous prediction error        
        u = Xdata(:,t); % input vector of size m+1
        [z, new_x] = RNN_state_fwd_prop(myRNN, u, myRNN.x);
            % we need z to compute myRNN.dtheta
        myRNN.Ypred(:,t) = myRNN.Wc*new_x; 
            % here we do not use new_x. myRNN.x is updated at the end of the loop
        e = Ydata(:,t) - myRNN.Ypred(:,t); 

        % gradient with respect to the output parameters
        myRNN.dtheta(:, idx_min_Wc:nb_weights) = reshape(-e*(new_x.'), [1, p*q]); 
        
        
        % Recursive calculation of dx/dtheta (DeepMind article)
        % or dh/dtheta with their notations (here x refers to the states)
        
        % computation of Dt (dynamic matrix)
        phi_prime_z = myRNN.phi_prime(z);
        Dt_diag = phi_prime_z.*diag(myRNN.Wa); 
            % the second factor is the column vector of the diagonal elements of Wa
            % Dt_diag : column vector of the diagonal elements of the dyanmics matrix Dt
            % we only need the diagonal elements in the SnAp1 approximation
        
        % computation of It (immediate jacobian)
        myRNN.It(:,1:q) = phi_prime_z*transpose(myRNN.x);
        myRNN.It(:,(q+1):(q+m+1)) = phi_prime_z*transpose(u);
        
        % update of the influence matrix and the gradient with respect to Wa and Wb
        myRNN.Jt = myRNN.It + Dt_diag.*myRNN.Jt;
        
        % gradient of the loss with respect to the states - je renomme ds (Tallec) en dx (Haykin)
        dx_transpose = -transpose(myRNN.Wc)*e; 

        myRNN.dtheta(:, 1:(size_Wa+size_Wb)) = reshape(dx_transpose.*myRNN.Jt, [1, size_Wa + size_Wb]);
        
        
        % Updates
        
        % Weight updates
        theta_vec =[reshape(myRNN.Wa, [1, size_Wa]), reshape(myRNN.Wb, [1, size_Wb]), reshape(myRNN.Wc, [1, size_Wc])];
            % line vector containing the concatenation of Wa, Wb and Wc
        new_theta = update_param_optim(theta_vec, myRNN.dtheta, pred_par, myRNN.grad_moments, t);
        myRNN.Wa = reshape(new_theta(:,1:size_Wa), [q, q]);
        myRNN.Wb = reshape(new_theta(:,(1 + size_Wa):(size_Wa + size_Wb)), [q, m+1]);
        myRNN.Wc = reshape(new_theta(:,idx_min_Wc:(size_Wa+size_Wb+size_Wc)), [p, q]);

        % States update
        myRNN.x = new_x;
        
        myRNN.pred_time_array(t) = toc;
        myRNN.pred_loss_function(t) = 0.5*(e.')*e;  % error evaluation so it is performed after time performance evaluation
        
    end  