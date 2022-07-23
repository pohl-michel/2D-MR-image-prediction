function [myRNN] = rnn_UORO( myRNN, pred_par, beh_par, Xdata, Ydata)

   % est-ce que j'ai bien g�r� les biais +1 (le vecteur b en gros)
   % il va �galement falloir relire tout ce code pour m'assurer que je n'ai pas fait d'erreur b�te
   % voir comment �liminer les boucles (apr�s avoir v�rifi� que tout marche bien)

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
            % we need z to compute myRNN.dtheta_g
        myRNN.Ypred(:,t) = myRNN.Wc*myRNN.x; 
            % here we do not use new_x. myRNN.x is updated at the end of the loop
        e = Ydata(:,t) - myRNN.Ypred(:,t);

        dx = -transpose(e)*myRNN.Wc; 
            % peut-�tre que je n'ai pas besoin de cette ligne (code optimization)...
            % je renomme ds (Tallec) en dx (Haykin)
              
        myRNN.dtheta(:,idx_min_Wc:nb_weights) = reshape(-e*(myRNN.x.'), [1, p*q]); 
            % also when optimizing the code, I can inject directly into the gtilde formula
        
        % gradient estimate computation
        gtilde = (dx*myRNN.xtilde)*myRNN.theta_tilde + myRNN.dtheta;
            % gtilde has size 1*k (line vector)
        
        % vector column of random signs
        if beh_par.GPU_COMPUTING
            nu = 2*(rand(q,1, 'gpuArray')>.5) - 1;
        else
            nu = 2*(rand(q,1)>.5) - 1;
        end
        
        % tangent forward propagation
        [~, Fx_plus_delta]= RNN_state_fwd_prop(myRNN, u, myRNN.x + pred_par.eps_tgt_fwd_prp*myRNN.xtilde);
        myRNN.xtilde = (1/pred_par.eps_tgt_fwd_prp)*(Fx_plus_delta - new_x);
        
        dtheta_g_aux = nu.*myRNN.phi_prime(z); % column vector of size q used for calculating myRNN.dtheta_g
        myRNN.dtheta_g(:,1:size_Wa) = reshape(dtheta_g_aux*transpose(myRNN.x), [1, size_Wa]);
        myRNN.dtheta_g(:,(1+size_Wa):(size_Wa+size_Wb)) = reshape(dtheta_g_aux*transpose(u), [1, size_Wb]);
        
        %normalizers
        rho0 = sqrt(norm(myRNN.theta_tilde)/(norm(myRNN.xtilde)+pred_par.eps_normalizers)) + pred_par.eps_normalizers;
        rho1 = sqrt(norm(myRNN.dtheta_g)/(norm(nu)+pred_par.eps_normalizers)) + pred_par.eps_normalizers;
        
        % update of x_tilde and theta_tilde
        myRNN.xtilde = rho0*myRNN.xtilde + rho1*nu;
        myRNN.theta_tilde = (myRNN.theta_tilde/rho0)+(myRNN.dtheta_g/rho1);
        
        % Weight updates
        theta_vec =[reshape(myRNN.Wa, [1, q*q]), reshape(myRNN.Wb, [1, q*(m+1)]), reshape(myRNN.Wc, [1, p*q])];
            % line vector containing the concatenation of Wa, Wb and Wc
        new_theta = update_param_optim(theta_vec, gtilde, pred_par, myRNN.grad_moments, t);
        myRNN.Wa = reshape(new_theta(:,1:size_Wa), [q, q]);
        myRNN.Wb = reshape(new_theta(:,(1 + size_Wa):(size_Wa + size_Wb)), [q, m+1]);
        myRNN.Wc = reshape(new_theta(:,idx_min_Wc:(size_Wa+size_Wb+size_Wc)), [p, q]);

        % States update
        myRNN.x = new_x;
        
        myRNN.pred_time_array(t) = toc;
        
        myRNN.pred_loss_function(t) = 0.5*(e.')*e;
            % error evaluation so it is performed after time performance evaluation
        
    end   
    
    if beh_par.GPU_COMPUTING
        myRNN.Ypred = gather(myRNN.Ypred);
        myRNN.pred_time_array = gather(myRNN.pred_time_array);
        myRNN.pred_loss_function = gather(myRNN.pred_loss_function);
    end    

end
