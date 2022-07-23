function [Wtrain, F, Xtrain_mean, eval_results] = PCA_of_DVF(beh_par, disp_par, OF_par, im_par, path_par, pred_par, br_model_par, eval_results)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    fprintf('Computing the PCA of the DVF... \n');
    
    % Calcul de la matrice X (faire une fonction qui dépend de t pour cela)

    X = zeros(im_par.nb_im, 2*(im_par.L)*(im_par.W));
    
    for t=2:im_par.nb_im
            OF_t_filename = write_2DOF_t_mat_filename(OF_par, path_par, t );
            load(OF_t_filename, 'u_t');
            
            for x=1:im_par.L
                for y = 1:im_par.W
                    px_lin_idx = x + (y-1)*im_par.L; 
                    X(t, 2*px_lin_idx-1) = u_t(y,x,1); 
                    X(t, 2*px_lin_idx)   = u_t(y,x,2); 
                end
            end
                
    end
    
    Xtrain = X(1:pred_par.tmax_training,:);
    
	% Centering the data matrix X 
    n = pred_par.tmax_training;
	J = eye(n) - (1/n)*ones(n,1)*ones(1,n);
	Xtrain_mean = mean(Xtrain); % mean of each column of the data matrix X
	Xtrain_centered = J*Xtrain;
    Xcentered = X - ones(im_par.nb_im, 1)*Xtrain_mean;
    
    % Faire la PCA
    [ Wtrain, ~, ~ ] = myPCA( Xtrain_centered, br_model_par.nb_pca_cp);
    tic
    F = (1/n)*Xcentered*Wtrain;
    %F = Xcentered*Wtrain; %normally (but this is not the version that I used in Chapter 4 of my thesis)
    eval_results.PCA_time_weights_calc_time = (1/im_par.nb_im)*toc;
    
    if beh_par.SAVE_PCA_CP_WEIGHTS_JPG    
    
        % Afficher les champs de deplacement principaux (composantes principales W)
            % normalement les composantes principales sont F mais cela fait plus de sens pour ce probleme la de parler de W - je garde les notations du
            % livre cependant)

        save_main_DVF_jpg(beh_par, path_par, disp_par, OF_par, im_par, br_model_par, Wtrain, Xtrain_mean);
        plot_weights( F, OF_par, path_par, disp_par);
    
    end   

    % save variables
    PCA_results_filename = write_PCAresults_mat_filename( beh_par, OF_par, path_par );
    save(PCA_results_filename, 'Wtrain', 'F', 'Xtrain_mean');
    
    org_data = F';
    save(path_par.time_series_data_filename , 'org_data');
    
end

