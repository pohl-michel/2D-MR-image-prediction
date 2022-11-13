function [Ypred, avg_pred_time, pred_loss_function] = train_and_predict(path_par, pred_par, beh_par )
% This function trains different prediction methods (including RNNs trained with RTRL, UORO, DNI, and SnAp-1)
% for the prediction of time series data. The information concerning the RNN and the predicted data are stored in the file 'myRNN.mat'.
% Training and prediction are performed multiple times to account for the random initialization of the initial synaptic weights.
%
% Author : Pohl Michel
% Date : September 27, 2021
% Version : v1.0
% License : 3-clause BSD License


    % loading the "past" data matrix X and the "future" data matrix Y.
    [ X, Y, Mu, Sg] = load_pred_data_XY( path_par, pred_par, beh_par);

    switch(pred_par.pred_meth)
        
        case 'multivariate linear regression' %linear regression
            
            fprintf('Performing prediction with multivariate linear regression \n');  
            [Ypred, avg_pred_time, pred_loss_function] = multivar_lin_pred(pred_par, X, Y);
            
        case {'RTRL', 'UORO', 'SnAp-1', 'DNI', 'RTRL v2', 'fixed W'} % prediction with an RNN        

            [p, M] = size(Y); %p is the RNN output dimension
            Ypred = zeros([size(Y), pred_par.nb_runs]);
            avg_pred_time = zeros(pred_par.nb_runs, 1);
            pred_loss_function = zeros(M, pred_par.nb_runs);
            
            [pred_par, myRNN] = initialize_rnn(pred_par, beh_par, p, M);
            
            for run_idx=1:pred_par.nb_runs
                % we run the prediction algorithm with different initial weigths
                
                switch(pred_par.pred_meth)
                    case 'RTRL'
                        myRNN = rnn_RTRL(myRNN, pred_par, beh_par, X, Y); 
                    case 'UORO'
                        myRNN = rnn_UORO(myRNN, pred_par, beh_par, X, Y); 
                    case 'SnAp-1'
                        myRNN = rnn_SnAp1(myRNN, pred_par, beh_par, X, Y);
                    case 'DNI'
                        myRNN = rnn_DNI(myRNN, pred_par, beh_par, X, Y);
                    case 'RTRL v2'
                        myRNN = rnn_RTRLv2(myRNN, pred_par, beh_par, X, Y); 
                    case 'fixed W'
                        myRNN = rnn_fixed_W(myRNN, pred_par, beh_par, X, Y);
                end
                
                Ypred(:,:,run_idx) =  myRNN.Ypred;
                avg_pred_time(run_idx) = mean(myRNN.pred_time_array);
                pred_loss_function(:,run_idx) = myRNN.pred_loss_function;
            
                myRNN = reset_rnn(myRNN, pred_par, beh_par);
                
            end
            
        case 'no prediction'
             
            [Ypred, avg_pred_time, pred_loss_function] = no_prediction(pred_par, X, Y);

        case 'LMS' % least mean squares (LMS)
            
            fprintf('Performing prediction with clipped multivariate linear mean squares (LMS) \n');
            [Ypred, avg_pred_time, pred_loss_function] = LMS_predict(pred_par, X, Y);
            
        case 'univariate linear regression' % univariate linear regression

            fprintf('Performing prediction with univariate linear regression \n');
            [Ypred, avg_pred_time, pred_loss_function] = univar_lin_predict(pred_par, X, Y);            
            
    end
    
    if pred_par.NORMALIZE_DATA
        for run_idx=1:pred_par.nb_runs
            Ypred(:,:,run_idx) = uncenterZ( Ypred(:,:,run_idx), Mu, Sg );
        end
    end
    
    if beh_par.SAVE_PRED_RESULTS
        pred_results_filename = write_pred_result_variables_filename(path_par, pred_par);
        save(pred_results_filename, 'Ypred', 'avg_pred_time', 'pred_loss_function');
    end

end

