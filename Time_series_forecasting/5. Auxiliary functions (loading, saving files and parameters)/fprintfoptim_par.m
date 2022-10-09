function fprintfoptim_par(fid, pred_par)

    if pred_par.GRAD_CLIPPING % gradient clipping
        fprintf(fid, 'Gradient clipping with threshold grd_tshld = %f \n', pred_par.grad_threshold);
    else
        fprintf(fid, 'No gradient clipping \n');
    end

    fprintf(fid, 'Optimization method: %s \n', pred_par.update_meth);
    fprintf(fid, 'Learning rate eta = %g \n', pred_par.learn_rate);    
    switch pred_par.update_meth 
        case 'stochastic gradient descent'
            % do nothing
        case 'ADAM'
            fprintf(fid, 'beta1 = %f  \n', pred_par.ADAM_beta1);
            fprintf(fid, 'beta2 = %f  \n', pred_par.ADAM_beta2);
            fprintf(fid, 'epsilon = %f  \n', pred_par.ADAM_eps);
    end    

end