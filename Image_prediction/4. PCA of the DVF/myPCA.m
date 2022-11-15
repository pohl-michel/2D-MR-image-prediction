function [ W, F, LAMBDA_squared_elts ] = myPCA( X, m )
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
% Author : Pohl Michel
% Date : September 24, 2022
% Version : v1.2
% License : 3-clause BSD License


    [n,~] = size(X);
    
    % spectral decomposition of the matrix X*X^t
    %Y = (1/n)*double(X*(X')); (version that I used in Chapter 4 of my thesis along with F = (1/n)*Xcentered*Wtrain in PCA_of_DVF.m)
    Y = double(X*(X')); % normally
    
    [V,D] = eigs((Y+Y')./2, m); % enforce that the matrix is numerically symmetric
    
    % sort the elements of D and return them as a vector 
    [LAMBDA_squared_elts, ind] = sort(diag(D), 'descend');
    
    % matrices from the singular decomposition of X
    K_m = V(:,ind);
    lambda_m_vec = sqrt(LAMBDA_squared_elts);
    lambda_m = diag(lambda_m_vec);
    
    % principal components scores matrix
    F = K_m*lambda_m;
    
    % weight matrix
    W = (X')*K_m*diag((lambda_m_vec.^-1));

end