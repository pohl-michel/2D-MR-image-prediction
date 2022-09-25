function [Wtrain, F, Xtrain_mean, eval_results] = PCA_of_DVF(beh_par, disp_par, OF_par, im_par, path_par, pred_par, br_model_par, eval_results)
% Computes PCA from the deformation vector field (DVF) data using the function "myPCA".
% Returns and saves (in path_par.temp_var_dir):
% - Xtrain_mean: a line vector containing the mean of the DVF data matrix of the training set
% - F: the weight matrix using the whole data (it contains the weights to be forecast)
% - Wtrain: the principal component matrix computed using the training set data matrix
% If beh_par.SAVE_PCA_CP_WEIGHTS_JPG is set to true, the principal components and associated weights are plotted and saved.
%
% The data matrix containing the DVF information is the following: 
%     [ u_x(x1, t1), u_y(x1, t1), ... , u_x(xm, t1), u_y(xm, t1)]
% X = [                           ...                           ]
%     [ u_x(x1, tM), u_y(x1, tM), ... , u_x(xm, tM), u_y(xm, tM)]
% where u_x(xi, t) (resp. u_j(xi, t)) is the x (resp. y) component of the DVF between t=1 and t at pixel xi.
% With the notations above, m denotes the number of pixels in the image and M the number of images in the training set.  
%
% Using PCA, we find two matrices F and W such that the X = F*(W') [cf details in the documentation of myPCA.m]
% Because (W')*W = eye(np_cpts), we also have F = X*W, property that we use in this function.
% 
% Rk 1: In my thesis I used Y = (1/n)*double(X*(X')) in myPCA.m and F = (1/n)*Xcentered*Wtrain in PCA_of_DVF.m, which is correct,
% but in v1.1 of this function, I use Y = double(X*(X')) in myPCA.m and F = Xcentered*Wtrain, which is standard and leads to W'*W = eye(nb_pca_cp),
%
% Rk 2: when performing prediction in load_pred_data_XY, the data matrix was standardized, not just centered,
% and the mean considered was the mean along the lines (here it is the mean along the columns), so I cannot refactor the code to include the same centering
% function easily.
%
% Rk 3: The PCA notations in the code, in the paper/thesis, and in the book (Adachi, Matrix-based introduction to Mult. data analysis) differ:
%  - F in the code/book <-> W in the paper/thesis (weight matrix)
%  - W in the code/book <-> U in the paper/thesis (principal component matrix)
%
% Author : Pohl Michel
% Date : September 24, 2022
% Version : v1.1
% License : 3-clause BSD License


    fprintf('Computing the PCA of the DVF... \n');
    
    % Calculation of the data matrix X containing the DVF information
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
        
	% Centering the data matrix of the training set Xtrain
    n = pred_par.tmax_training;
	J = eye(n) - (1/n)*ones(n,1)*ones(1,n);
	Xtrain = X(1:pred_par.tmax_training,:);
    Xtrain_mean = mean(Xtrain); % line vector containing the mean of each column of Xtrain
	Xtrain_centered = J*Xtrain;

    % Centering the data matrix X using the mean of the training data
    Xcentered = X - ones(im_par.nb_im, 1)*Xtrain_mean;
    
    % Computation of the principal components using the centered training data
    [ Wtrain, ~, ~ ] = myPCA(Xtrain_centered, br_model_par.nb_pca_cp);

    % Computation of the weights for all time t using the former principal components
    tic
    % F = (1/n)*Xcentered*Wtrain; version that I used in Chapter 4 of my thesis, along with Y = (1/n)*double(X*(X')) in myPCA.m in the preivous code version.
    F = Xcentered*Wtrain;
    eval_results.PCA_time_weights_calc_time = (1/im_par.nb_im)*toc;
   
    % Plotting the principal components (the 2D principal deformation vectors)
    if beh_par.SAVE_PCA_CP_WEIGHTS_JPG    
        save_main_DVF_jpg(beh_par, path_par, disp_par, OF_par, im_par, br_model_par, Wtrain, Xtrain_mean);
        plot_weights( F, OF_par, path_par, disp_par);
    end   

    % Saving variables
    PCA_results_filename = write_PCAresults_mat_filename( beh_par, OF_par, path_par );
    save(PCA_results_filename, 'Wtrain', 'F', 'Xtrain_mean');
    
    org_data = F';
    save(path_par.time_series_data_filename , 'org_data');
    
end