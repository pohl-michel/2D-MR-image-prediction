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
            case 'transformer'
                fprintf(fid, 'Signal history length k = %d \n', pred_par.SHL); % common to all algorithms really - to move before switch
                fprintf(fid, 'Batch size: %d \n', pred_par.batch_size);
                fprintf(fid, 'Number of epochs: %d \n', pred_par.num_epochs);
                fprintf(fid, 'Embedding dimension: %d \n', pred_par.d_model);
                fprintf(fid, 'Number of heads: %d \n', pred_par.nhead);
                fprintf(fid, 'Number of encoder layers: %d \n', pred_par.num_layers);
                fprintf(fid, 'Hidden layer dimension of the feedforward network inside the encoder layers: %d \n', pred_par.dim_feedforward);
                fprintf(fid, 'Hidden layer dimension of the output feedforward network: %d \n', pred_par.final_layer_dim);
                fprintf(fid, 'Dropout rate: %g \n', pred_par.dropout);
                fprintf(fid, 'Learning rate: %g \n', pred_par.learn_rate);
                fprintf(fid, 'Number of runs due to random weights initialization (for computing RMSE, etc.) nb_runs = %d \n', pred_par.nb_runs);
                print_device_used(fid, pred_par.GPU_COMPUTING);
            case 'multivariate linear regression'
                fprintf(fid, 'Signal history length k = %d \n', pred_par.SHL);  
            case 'SVR'
                fprintf(fid, 'Signal history length k = %d \n', pred_par.SHL);    
                fprintf(fid, 'Kernel scale (~sigma, controls the spread of the RBF kernel) = %d \n', pred_par.svr_kernel_scale); 
                fprintf(fid, 'Box constraint C (regularization parameter) = %d \n', pred_par.svr_box_constraint); 
                fprintf(fid, 'Epsilon (regression margin of tolerance) = %d \n', pred_par.svr_epsilon); 
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
    print_device_used(fid, pred_par.GPU_COMPUTING);

end

function print_device_used(fid, gpu_computing)
    if gpu_computing
        fprintf(fid, 'Computation with the GPU \n');
    else
        fprintf(fid, 'Computation with the CPU \n');
    end
end