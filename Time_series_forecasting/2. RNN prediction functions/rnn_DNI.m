function myRNN = rnn_DNI(myRNN, pred_par, X, Ytrue)
% rnn_DNI performs the training of a recurrent neural network (RNN) trained with decoupled neural interfaces (DNI) and gradient clipping.
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
% Remark:
%   One can comment the expression of dnormfA and uncomment that just below to perform an ablation study.
%   We argue that the proposed additional term (x_tilde_next.')*(fA*(D.')) is an improvement and leads to better performance (cf  	
%   https://doi.org/10.48550/arXiv.2403.01607) compared to the version in Marschall, et al., "A unified framework of online learning 
%   algorithms for training recurrent neural networks." Journal of machine learning research 21.135 (2020): 1-34.
%
% Author : Pohl Michel
% Date : March 12th, 2025
% Version : v1.1
% License : 3-clause BSD License
   
    [~, M] = size(X);
    m = myRNN.input_space_dim;
    q = myRNN.state_space_dim;
    p = myRNN.output_space_dim;
    nb_weights = myRNN.nb_weights; % size of the theta vector  

    size_Wa = q*q;
    size_Wb = q*(m+1);
    size_Wc = p*q;

    idx_min_Wc = size_Wa + size_Wb + 1;    
    
    for t=1:M

        tic
        
        % Forward propagation (prediction) and calculation of the instantaneous prediction error
        u = X(:, t); % input vector of size m+1
        [z, new_x] = RNN_state_fwd_prop(myRNN, u, myRNN.x);
        myRNN.Ypred(:, t) = myRNN.Wc*new_x;
        e = Ytrue(:, t) - myRNN.Ypred(:, t);
              
        % loss gradient with respect to the output weights Wc
        myRNN.dtheta(:, idx_min_Wc:nb_weights) = reshape(-e*(new_x.'), [1, p*q]); 

        % Dynamic matrix - myRNN.Wa is used instead of Diag(myRNN.Wa) in contrast to SnAp-1
        phi_prime_z = myRNN.phi_prime(z);
        D = phi_prime_z.*myRNN.Wa;

        % gradient of the loss with respect to the states
        dx = -(e.')*myRNN.Wc; 

        % vector x_tilde such that c = x_tilde*A where c is the "credit assignment" vector
        x_tilde_next = [new_x.', Ytrue(:,t).', 1]; % Rk: x_tilde is initialized in "iniitalize_rnn.m"

        % credit assignment vector temporary estimations (these are not true values as A is not estimated properly at this step)
        c_temp = myRNN.x_tilde*myRNN.A;
        c_next_temp = x_tilde_next*myRNN.A;

        % function whose squared norm we want to minimize to find A such that c = x_tilde*A
        fA = c_temp - dx - c_next_temp*D; % c_next is computed first to keep a computational complexity O(q^2)        

        % derivative of the squared norm of f with respect to A
        dnormfA = (myRNN.x_tilde.')*fA - (x_tilde_next.')*(fA*(D.')); % fA*(D.') is computed first to keep a computational complexity O(q^2)
        % dnormfA = (myRNN.x_tilde.')*fA; % ablation experiment

        % finding A which minimizes normfA
        myRNN.A = update_param_optim(myRNN.A, dnormfA, pred_par.optim_par_A);

        % credit assignment vector (this time the "correct" value after optimization of A):
        c = myRNN.x_tilde*myRNN.A;

        % loss gradient with respect to Wa and Wb
        myRNN.dtheta(:,1:(size_Wa + size_Wb)) = reshape(((c.').*phi_prime_z)*[myRNN.x.', u.'], [1, q*(q+m+1)]);
        
        % Weight updates
        theta_vec =[reshape(myRNN.Wa, [1, size_Wa]), reshape(myRNN.Wb, [1, size_Wb]), reshape(myRNN.Wc, [1, size_Wc])];
            % line vector containing the concatenation of Wa, Wb and Wc
        new_theta = update_param_optim(theta_vec, myRNN.dtheta, pred_par, myRNN.grad_moments, t);
        myRNN.Wa = reshape(new_theta(:,1:size_Wa), [q, q]);
        myRNN.Wb = reshape(new_theta(:,(1 + size_Wa):(size_Wa + size_Wb)), [q, m+1]);
        myRNN.Wc = reshape(new_theta(:,idx_min_Wc:nb_weights), [p, q]);

        % Update of the states, x_tilde, recording computation time and loss function value
        myRNN.x = new_x;
        myRNN.x_tilde = x_tilde_next;

        myRNN.pred_time_array(t) = toc; 
        myRNN.pred_loss_function(t) = 0.5*(e.')*e; % error evaluation so it is performed after time performance evaluation
        
    end   
    
    if pred_par.GPU_COMPUTING
        myRNN.Ypred = gather(myRNN.Ypred);
        myRNN.pred_time_array = gather(myRNN.pred_time_array);
        myRNN.pred_loss_function = gather(myRNN.pred_loss_function);
    end    

end