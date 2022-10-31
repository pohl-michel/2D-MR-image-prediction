function [pred_par, myRNN] = initialize_rnn(pred_par, beh_par, p, M)
% Initialization of the variables controlling the internal dynamics of the RNN :
%   - the synaptic weights  
%   - the system states x
%   - selection of the activation function
%   - ...
% The variable p represents the dimension of the RNN output space.
% The symnaptic weights are  randomly distributed according to a normal distribution of standard deviation sg = pred_par.Winit_std_dev. 
%
% Rk: GPU computing not implemented (yet) with SnAp-1 and DNI
%
% Author : Pohl Michel
% Date : September 27th, 2021
% Version : v1.0
% License : 3-clause BSD License


    m = pred_par.SHL*p;
    q = pred_par.rnn_state_space_dim;
    sg = pred_par.Winit_std_dev;

    % We make some variables easier to access in other functions:
    myRNN.input_space_dim = m;
    myRNN.state_space_dim = q;
    myRNN.output_space_dim = p;
    myRNN.nb_weights = q*(p+q+m+1);
    
    % evaluation variables initialization
    myRNN.Ypred = zeros(p, M);
    myRNN.pred_loss_function = zeros(M, 1, 'single');
    myRNN.pred_time_array = zeros(M, 1, 'single');  
    
    % weights initialization
    myRNN.Wa = normrnd(0, sg, [q, q]);
    myRNN.Wb = normrnd(0, sg, [q, m+1]); % "+1" because of the bias unit
    myRNN.Wc = normrnd(0, sg, [p, q]);

    % states initialization
    myRNN.x = zeros(q, 1);

    % activation function and its derivative
    myRNN.phi = @(v) tanh(v);
    myRNN.phi_prime = @(v) 1./((cosh(v)).^2);
    
    %% Initialization of variables specific to the training method chosen
    
    switch(pred_par.pred_meth)
        case 'RTRL'
            
            % state space dynamics 3D tensor  
            myRNN.LBDA = zeros(q, q + m + 1, q);
                % myRNN.LBDA(:,:,j) is the matrix LBDA_{j,n} , ie the Jacobian matrix of x_n as a function of w_j (j in 1,...q)
                % cf the Haykin's book

            % w(:,j) corresponds to the w_j matrix at time n in Haykin's book
            % w_j = [wa_j , wb_j]
            % with Wa^T = [wa_1, ..., wa_q] and Wb^T = [wb_1, ..., wb_q]
            myRNN.w = zeros(m+q+1, q);

            % gradient of the instantaneous energy loss En with respect to each entry in w (corresponding to the gradients with respect to Wa and Wb) - cf Haykin's book
            myRNN.w_gradient = zeros(m+q+1, q);

            % matrix U{:,:,j) ( "U_{j,n}" in Haykin's book)
            myRNN.U = zeros(q, m+q+1, q);    
            
        case 'UORO'

            % Variables xtilde and theta_tilde such that 
            % dx/dtheta is approximated by xtilde*theta_tilde

            myRNN.xtilde = zeros(myRNN.state_space_dim, 1);
                % named stilde in the paper from Tallec et al.
            myRNN.theta_tilde = zeros(1, myRNN.nb_weights);
            
            myRNN.dtheta = zeros(1, myRNN.nb_weights);
            myRNN.dtheta_g = zeros(1, myRNN.nb_weights);
            
        case 'SnAp-1'
            
            myRNN.dtheta = zeros(1, myRNN.nb_weights);

            myRNN.It = zeros(q, q+m+1);
            myRNN.Jt = zeros(q, q+m+1);

        case 'DNI'

            myRNN.dtheta = zeros(1, myRNN.nb_weights);

            % Ydata(:,t-1).' does not exist when t=0 so we have to initialize x_tilde
            myRNN.x_tilde = [zeros(1, p+q), 1]; % because [1, p] = size(Ydata(:, t-1).') and [1, q] = size(myRNN.x.')
            
            % matrix A such that c = x_tilde*A where c is the credit assignment and x_tilde = [x, Ytrue(:, t).', 1]
            myRNN.A = normrnd(0, 1/sqrt(q), [p+q+1, q]); % same as in Marschall's paper

            % I use the following trick to be able to do grid search over pred_par.learn_rate_A instead of directly defining pred_par.optim_par_A in load_pred_par.m
            pred_par.optim_par_A.learn_rate = pred_par.learn_rate_A;
            pred_par.optim_par_A.GRAD_CLIPPING = pred_par.GRAD_CLIPPING_A;
            pred_par.optim_par_A.update_meth = pred_par.update_meth_A;
            
    end

    if strcmp(pred_par.update_meth, 'ADAM')
         switch(pred_par.pred_meth)
            case 'RTRL'
                myRNN.grad_moments.m_t = zeros(m+p+q+1, q);
                myRNN.grad_moments.v_t = zeros(m+p+q+1, q);
            case {'UORO', 'SnAp-1', 'DNI'} % RNN UORO, SnAp1, or DNI
                myRNN.grad_moments.m_t = zeros(1, myRNN.nb_weights);
                myRNN.grad_moments.v_t = zeros(1, myRNN.nb_weights);           
         end  
    else
        myRNN.grad_moments = struct();
    end    
    
    if beh_par.GPU_COMPUTING
        
        myRNN.Ypred = gpuArray(myRNN.Ypred);
        myRNN.pred_loss_function = gpuArray(myRNN.pred_loss_function);
        myRNN.pred_time_array = gpuArray(myRNN.pred_time_array);

        myRNN.Wa = gpuArray(myRNN.Wa);
        myRNN.Wb = gpuArray(myRNN.Wb);
        myRNN.Wc = gpuArray(myRNN.Wc);
        myRNN.x = gpuArray(myRNN.x);
        
        switch(pred_par.pred_meth)
            case 'RTRL'
                myRNN.LBDA = gpuArray(myRNN.LBDA);
                myRNN.w = gpuArray(myRNN.w);
                myRNN.w_gradient = gpuArray(myRNN.w_gradient);
                myRNN.U = gpuArray(myRNN.U);
            case 'UORO'
                myRNN.xtilde = gpuArray(myRNN.xtilde);
                myRNN.theta_tilde = gpuArray(myRNN.theta_tilde);
                myRNN.dtheta = gpuArray(myRNN.dtheta);
                myRNN.dtheta_g = gpuArray(myRNN.dtheta_g);
        end
        
        if strcmp(pred_par.update_meth, 'ADAM')
            myRNN.grad_moments.m_t = gpuArray(myRNN.grad_moments.m_t);
            myRNN.grad_moments.v_t = gpuArray(myRNN.grad_moments.v_t);                 
        end        
        
    end

        
end