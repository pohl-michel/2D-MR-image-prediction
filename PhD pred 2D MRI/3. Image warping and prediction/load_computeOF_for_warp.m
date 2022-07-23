function [ u_t, OFcalc_time_t ] = load_computeOF_for_warp( of_type_idx, t, OF_par, path_par, im_par, br_model_par, pred_par, beh_par, warp_par)
%UNTITLED21 Summary of this function goes here
%   Detailed explanation goes here

    switch of_type_idx
        case 1 % initial optical flow
            OF_t_filename = write_2DOF_t_mat_filename(OF_par, path_par, t );
            load(OF_t_filename, 'u_t');
            OFcalc_time_t = 0.0; % non relevant here
   
        case 2 % DVF from PCA
            
            PCA_results_filename = write_PCAresults_mat_filename( beh_par, OF_par, path_par );
            load(PCA_results_filename, 'Wtrain', 'F', 'Xtrain_mean');   
            u_t = zeros(im_par.W, im_par.L, 2);
            % Programmation pas efficace du tout à cause des boucles mais pour le moment on peut faire comme cela
            tic
            U = Xtrain_mean + F(t,1:br_model_par.nb_pca_cp)*transpose(Wtrain(:,1:br_model_par.nb_pca_cp));
            OFcalc_time_t = toc;      
            for x=1:im_par.L
                for y = 1:im_par.W
                    px_lin_idx = x + (y-1)*im_par.L; 
                    u_t(y,x,1) = U(2*px_lin_idx-1);
                    u_t(y,x,2) = U(2*px_lin_idx);
                end
            end  
        
        case 3 % DVF prediction
            
            pred_results_filename = write_pred_result_variables_filename(path_par, pred_par);
            load(pred_results_filename, 'Ypred');
            
            [~, M, nb_runs] = size(Ypred);
            nb_runs = min(warp_par.nb_runs_for_cc_eval, nb_runs);
            tmax = pred_par.tmax_pred;            
            t_Yidx = t - tmax + M; % time index corresponding to t in the array Ypred

            PCA_results_filename = write_PCAresults_mat_filename( beh_par, OF_par, path_par );
            load(PCA_results_filename, 'Wtrain', 'Xtrain_mean');               

            u_t = zeros(im_par.W, im_par.L, 2, nb_runs);
            OFcalc_time_tab = zeros(nb_runs, 1);
            
            for run_idx = 1:nb_runs
            
                Fpred_t = transpose(Ypred(:, t_Yidx, run_idx));

                tic
                U = Xtrain_mean + Fpred_t*transpose(Wtrain(:,1:br_model_par.nb_pca_cp));
                OFcalc_time_tab(run_idx) = toc;   
                
                for x=1:im_par.L
                    for y = 1:im_par.W
                        px_lin_idx = x + (y-1)*im_par.L; 
                        u_t(y,x,1, run_idx) = U(2*px_lin_idx-1);
                        u_t(y,x,2, run_idx) = U(2*px_lin_idx);
                    end
                end  
            
            end

            OFcalc_time_t = mean(OFcalc_time_tab);
            
    end

end