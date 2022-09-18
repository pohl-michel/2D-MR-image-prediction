function [ W, F, LAMBDA_squared_elts ] = myPCA( X, m )
% Computes PCA given the data matrix X and returns the m principal components' matrix F and associated weight matrix W

[n,~] = size(X);

% spectral decomposition of the matrix X*X^t
Y = (1/n)*double(X*(X'));

%Y = double(X*(X')); normally (but this is not the version that I used in Chapter 4 of my thesis)
[V,D] = eigs((Y+Y')./2, m); % enforce that the matrix is numerically symmetric

% sort the elements of D and return them as a vector 
[LAMBDA_squared_elts, ind] = sort(diag(D),'descend');

% matrices from the singular decomposition of X
K_m = V(:,ind);
lambda_m_vec = sqrt(LAMBDA_squared_elts);
lambda_m = diag(lambda_m_vec);

% principal components scores matrix
F = K_m*lambda_m;

% weight matrix
W = (X')*K_m*diag((lambda_m_vec.^-1));

end