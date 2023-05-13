function fprintfpred_par( fid, pred_par, beh_par )
% Prints the prediction parameters relative to each prediction method.
%
% Author : Pohl Michel
% Date : September 27th, 2021
% Version : v1.0
% License : 3-clause BSD License
    
        fprintf(fid, 'Prediction method : %s \n', pred_par.pred_meth);
        fprintf(fid, 'Training between t = 1 and t = %d \n', pred_par.tmax_training);
        fprintf(fid, 'Evaluation between t = %d and t = %d \n', pred_par.t_eval_start, pred_par.tmax_pred);
        fprintf(fid, 'Horizon of the prediction h = %d \n', pred_par.horizon);
        if pred_par.NORMALIZE_DATA
            fprintf(fid, 'data normalized before prediction\n');
        else
            fprintf(fid, 'data not normalized before prediction \n');
        end
        
        switch(pred_par.pred_meth)
            case 'multivariate linear regression'
                fprintf(fid, 'Signal history length k = %d \n', pred_par.SHL);                
            case {'RTRL', 'SnAp-1', 'RTRL v2', 'fixed W'}
                fprintfRNN_common(fid, pred_par)
            case 'LMS'
                fprintf(fid, 'Signal history length / filter order k = %d \n', pred_par.SHL);  
                fprintf(fid, 'Learning rate / step size eta = %g \n', pred_par.learn_rate);
                fprintfoptim_par(fid, pred_par)              
            case 'UORO'
                fprintfRNN_common(fid, pred_par)             
                fprintf(fid, 'Step epsilon used for tangent forward propagation eps_tgt_fwd = %g \n', pred_par.eps_tgt_fwd_prp);
                fprintf(fid, 'Parameter epsilon used when computing the normalizers rho1 and rho2 : eps_nlzer = %d \n', pred_par.eps_normalizers);  
            case 'univariate linear regression'
                fprintf(fid, 'Signal history length k = %d \n', pred_par.SHL);   
            case 'DNI'
                fprintfRNN_common(fid, pred_par)
                fprintf(fid, 'Optimization method to find A such that c = x_tilde*A where c is the credit assignment \n');
                pred_par.optim_par_A.learn_rate = pred_par.learn_rate_A;
                pred_par.optim_par_A.GRAD_CLIPPING = pred_par.GRAD_CLIPPING_A;
                pred_par.optim_par_A.update_meth = pred_par.update_meth_A;
                fprintfoptim_par(fid, pred_par.optim_par_A)
        end

end


function fprintfRNN_common(fid, pred_par)

    fprintf(fid, 'Signal history length k = %d \n', pred_par.SHL);
    fprintf(fid, 'Nb of neurons in the hidden layer q = %d \n', pred_par.rnn_state_space_dim);
    fprintf(fid, 'Synaptic weights standard deviation (initialization) sg = %g \n', pred_par.Winit_std_dev);
    fprintf(fid, 'Number of runs due to random weights initialization (for computing RMSE of time signals) nb_runs = %d \n', pred_par.nb_runs);
    fprintfoptim_par( fid, pred_par )
    if pred_par.GPU_COMPUTING
        fprintf(fid, 'Computation with the GPU \n');
    else
        fprintf(fid, 'Computation with the CPU \n');
    end

end