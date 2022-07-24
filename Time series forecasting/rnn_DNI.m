function [myRNN] = rnn_DNI( myRNN, pred_par, beh_par, Xdata, Ydata)

   % est-ce que j'ai bien géré les biais +1 (le vecteur b en gros)
   % il va également falloir relire tout ce code pour m'assurer que je n'ai pas fait d'erreur bête
   % voir comment éliminer les boucles (après avoir vérifié que tout marche bien)


   %A compléter notamment:
   %    - la fonction initialize_rnn
   %    - la fonction reset_rnn
   %    - le milieu de cette fonction.
   

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
              
        myRNN.dtheta(:,idx_min_Wc:nb_weights) = reshape(-e*(myRNN.x.'), [1, p*q]); 
            % also when optimizing the code, I can inject directly into the gtilde formula
        
        
        % To complete here  



        
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
        
        myRNN.pred_loss_function(t) = 0.5*(e.')*e;
            % error evaluation so it is performed after time performance evaluation
        
    end   
    
    if beh_par.GPU_COMPUTING
        myRNN.Ypred = gather(myRNN.Ypred);
        myRNN.pred_time_array = gather(myRNN.pred_time_array);
        myRNN.pred_loss_function = gather(myRNN.pred_loss_function);
    end    

end