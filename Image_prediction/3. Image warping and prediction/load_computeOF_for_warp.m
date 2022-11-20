function [ u_t, OFcalc_time_t ] = load_computeOF_for_warp(dvf_type, t, OF_par, path_par, im_par, br_model_par, pred_par, beh_par, warp_par)
% Returns u_t, the two-dimensional optical flow at time t.
% This function is called in "eval_of_warp_corr" to load u_t and warp the initial image with u_t.
% u_t can be either:
%   1) the initial optical flow
%   2) the optical flow reconstructed from PCA
%   3) the optical flow reconstructed from PCA with the predicted weights
% In case 2 and 3, the PCA results (and the predicted weights in case 3) are loaded and reconstruction is performed in the current function.
% The reconstruction time is stored in OFcalc_time_t.
% In case 3, the u_t array has a fourth dimension corresponding to the run index, as prediction can be random and require multiple initializations,
% and OFcalc_time_t is averaged over the number of runs.
%
% Rk 1: removing the loops in case 2 and 3 (using reshape) could improve time performance
% Rk 2: The PCA notations in the code, in the paper/thesis, and in the book (Adachi, Matrix-based introduction to Mult. data analysis) differ:
%  - U in the code <-> X in the paper/thesis/book (data matrix)
%  - F in the code/book <-> W in the paper/thesis (weight matrix)
%  - W in the code/book <-> U in the paper/thesis (principal component matrix)
% Rk 3 (23/09/2022): I refactored the code using an auxiliary function "reconstruct_u_from"; 
% if the additional statement "u_t = zeros(im_par.W, im_par.L, 2)" takes too much time, I can revert the function back to the previous state.
%
% Author : Pohl Michel
% Date : Sept 23rd, 2022
% Version : v1.1
% License : 3-clause BSD License


    switch dvf_type

        case 'initial DVF'

            OF_t_filename = write_2DOF_t_mat_filename(OF_par, path_par, t );
            load(OF_t_filename, 'u_t');
            OFcalc_time_t = 0.0; % non relevant here
   
        case 'DVF from PCA'
            
            % loading the PCA results
            PCA_results_filename = write_PCAresults_mat_filename( beh_par, OF_par, path_par );
            load(PCA_results_filename, 'Wtrain', 'F', 'Xtrain_mean');   
            
            % reconstruction of the DVF using only the first nb_pca_cp-th principal components
            pca_obj = myPCA(br_model_par.nb_pca_cp, Wtrain(:,1:br_model_par.nb_pca_cp));
            [OFcalc_time_t, u_t] = reconstruct_u_from(Xtrain_mean, F(t,1:br_model_par.nb_pca_cp), pca_obj, im_par);
        
        case 'predicted DVF'
            
            % loading the predicted weights of the principal components
            pred_results_filename = write_pred_result_variables_filename(path_par, pred_par);
            load(pred_results_filename, 'Ypred');
            
            % loading the other PCA results
            PCA_results_filename = write_PCAresults_mat_filename( beh_par, OF_par, path_par );
            load(PCA_results_filename, 'Wtrain', 'Xtrain_mean');   

            % some useful parameters
            [~, M, nb_runs] = size(Ypred);
            nb_runs = min(warp_par.nb_runs_for_cc_eval, nb_runs);
            tmax = pred_par.tmax_pred;            
            t_Yidx = t - tmax + M; % t_Yidx: time index corresponding to t in the array Ypred - reminder: M is the number of predictions           

            % preparing arrays
            u_t = zeros(im_par.W, im_par.L, 2, nb_runs);
            OFcalc_time_tab = zeros(nb_runs, 1);
            pca_obj = myPCA(br_model_par.nb_pca_cp, Wtrain(:,1:br_model_par.nb_pca_cp)); % we keep only the first nb_pca_cp-th principal components
            Ypred_t = Ypred(1:br_model_par.nb_pca_cp, t_Yidx, :); % idem

            for run_idx = 1:nb_runs
 
                Fpred_t = transpose(Ypred_t(:, :, run_idx));
                [OFcalc_time_t_run_idx, u_t_run_idx] = reconstruct_u_from(Xtrain_mean, Fpred_t, pca_obj, im_par); % DVF reconstruction
                
                OFcalc_time_tab(run_idx) = OFcalc_time_t_run_idx;  
                u_t(:,:,:,run_idx) = u_t_run_idx;
           
            end

            OFcalc_time_t = mean(OFcalc_time_tab);
            
    end

end


function [OFcalc_time_t, u_t] = reconstruct_u_from(X, F, pca_obj, im_par)

    % reconstruction of the DVF using only the first nb_pca_cp-th principal components
    [U, OFcalc_time_t] = pca_obj.reconstruct_data(X, F);
    
    % reshaping the matrix U containing the optical flow information
    u_t = zeros(im_par.W, im_par.L, 2);
    for x=1:im_par.L
        for y = 1:im_par.W
            px_lin_idx = x + (y-1)*im_par.L; 
            u_t(y,x,1) = U(2*px_lin_idx-1);
            u_t(y,x,2) = U(2*px_lin_idx);
        end
    end

end