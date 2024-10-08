classdef myPCA
    % class to compute PCA
    %
    % Author : Pohl Michel
    % Date : Novemeber 20, 2022
    % Version : v2.0
    % License : 3-clause BSD License

    properties
        nb_cpt
        W
    end

    methods

        function pca_obj = myPCA(nb_pca_cpt, Wtrain)
            % PCA class constructor

            pca_obj.nb_cpt = nb_pca_cpt;
            if nargin == 2
                pca_obj.W = Wtrain;
            end

        end

        function [ W, F, LAMBDA_squared_elts, pca_obj ] = fit(pca_obj, X, normalize_eigenvectors)
            % Computes PCA given the data matrix X and returns the m principal components' matrix F, the associated weight matrix W, and the eigenvalues LAMBDA_squared_elts.
            % Usually, the input matrix X needs to be centered, i.e., each column of X should have a mean equal to 0. 
            %
            % Given a matrix X of size (n, m), PCA enables finding two matrices F and W which minimize the quantity || X - F*(W') ||_2 subject to the following conditions:
            % - size(F) = [n, nb_cpts] and size(W) = [m, nb_cpts]
            % - (F')*F is a diagonal matrix
            % - (W')*W = eye(np_cpts)
            %
            % Rk 1: In the classic formulation, Y = double(X*(X')) instead of Y = (1/n)*double(X*(X')),
            % our formulation implies that F = (1/n)*X*W instead of F = X*W (cf the function "PCA_of_DVF")
            %
            % Rk 2: The PCA notations in the code, in the paper/thesis, and in the book (Adachi, Matrix-based introduction to Mult. data analysis) differ:
            %  - F in the code/book <-> W in the paper/thesis (weight matrix)
            %  - W in the code/book <-> U in the paper/thesis (principal component matrix)
            %
            % Rk 3: % the "eigs" method return eigenvectors with signs varying depending on the machine or the Matlab version
            % https://www.mathworks.com/help/matlab/ref/eigs.html 
            % When normalize_eigenvectors is set to true, we multiply each eigenvector by the sign of its first nonzero element for repeatability.

            % spectral decomposition of the matrix X*X^t
            % [n,~] = size(X);
            % Y = (1/n)*double(X*(X')); % (version that I used in Chapter 4 of my thesis along with F = (1/n)*Xcentered*Wtrain in PCA_of_DVF.m)
            Y = double(X*(X')); % normally
            
            [V,D] = eigs((Y+Y')./2, pca_obj.nb_cpt); % enforce that the matrix is numerically symmetric
            
            % sort the elements of D and return them as a vector 
            [LAMBDA_squared_elts, ind] = sort(diag(D), 'descend');
            
            % sorted eigenvector matrix
            K_m = V(:,ind);

            if normalize_eigenvectors
                K_m = normalize(K_m);
            end

            % vector containing the square root of the eigenvectors 
            lambda_m_vec = sqrt(LAMBDA_squared_elts);
            
            % principal components scores matrix
            F = K_m*diag(lambda_m_vec);
            
            % weight matrix
            W = (X')*K_m*diag((lambda_m_vec.^-1));
            pca_obj.W = W;

        end

        function [F, weights_calc_time] = compute_weights(pca_obj, Xcentered)
            % used to compute the matrix F from the components W in the training data

            [nb_ex, ~] = size(Xcentered);
            tic
            % F = (1/n)*Xcentered*pca_obj.W; % version that I used in Chapter 4 of my thesis, along with Y = (1/n)*double(X*(X')) in myPCA.m in the previous code version.
            F = Xcentered*pca_obj.W;
            weights_calc_time = (1/nb_ex)*toc;

        end

        function [U, reconstruction_time] = reconstruct_data(pca_obj, Xtrain_mean, F)
            % reconstruction/estimation of the data

            tic
            U = Xtrain_mean + F*transpose(pca_obj.W);
            reconstruction_time = toc;

        end

    end
end

function normalized_K = normalize(K)
    % multiplies each column of the matrix K by the sign of its first non-zero element

    % Find indices of first non-zero elements in each column
    [~, first_nonzero_idx] = max(K~=0, [], 1);

    % column indices
    col_idces = (0:size(K, 2)-1);
    
    % number of rows
    nb_rows = size(K, 1);

    % Extract signs of first non-zero elements
    signs = sign(K(first_nonzero_idx + col_idces*nb_rows));

    normalized_K = K .* signs;

end