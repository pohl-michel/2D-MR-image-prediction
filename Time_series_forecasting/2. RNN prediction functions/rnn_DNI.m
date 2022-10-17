function myRNN = rnn_DNI(myRNN, pred_par, beh_par, X, Ytrue)

   % il va falloir relire tout ce code pour m'assurer que je n'ai pas fait d'erreur bête
   % voir comment éliminer les boucles (après avoir vérifié que tout marche bien)
   
   % revoir l'écriture de SnAp-1, UORO, et RTRL également... 
   % notamment myRNN.Ypred(:,t) = myRNN.Wc*myRNN.x;  au lieu de myRNN.Ypred(:,t) = myRNN.Wc*new_x
   % il y avait un problème avec myRNN.dtheta(:,idx_min_Wc:nb_weights) qui vaut reshape(-e*(new_x.'), [1, p*q]) et pas reshape(-e*(myRNN.x.'), [1, p*q])

   
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
        myRNN.dtheta(:,idx_min_Wc:nb_weights) = reshape(-e*(new_x.'), [1, p*q]); 

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
    
    if beh_par.GPU_COMPUTING
        myRNN.Ypred = gather(myRNN.Ypred);
        myRNN.pred_time_array = gather(myRNN.pred_time_array);
        myRNN.pred_loss_function = gather(myRNN.pred_loss_function);
    end    

end