function u = reshape_DVF(im_par, U)
% Reshapes the DVF u (as an array) from the vector U.
%
% Author : Pohl Michel
% Date : September 25th, 2022
% Version : v2.0
% License : 3-clause BSD License

    u = zeros(im_par.W, im_par.L, 2);
    for x=1:im_par.L
        for y=1:im_par.W
            px_lin_idx = x + (y-1)*im_par.L; %linear indexing here because Xtrain_mean is a line vector
            u(y,x,1) = U(2*px_lin_idx-1); 
            u(y,x,2) = U(2*px_lin_idx);                
        end
    end 

end