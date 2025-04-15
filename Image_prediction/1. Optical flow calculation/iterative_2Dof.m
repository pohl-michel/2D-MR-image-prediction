function [ u_temp ] = iterative_2Dof( I, J, u_temp, OF_par )
% Given two images I and J and an initial guess of the deformation vector field (DVF) u_temp between I and J,
% this function refines u_temp iteratively using the Lucas Kanade method.
%
% Author : Pohl Michel
% Date : July 16th, 2020
% Version : v1.0
% License : 3-clause BSD License

    grad_I = spatial_grad_2D(I, OF_par);
    for k = 1:OF_par.nb_iter
        fprintf('\t\t refinement iteration n°%d \n', k); 
        u_temp = u_temp + OF_LK_2D( grad_I, translate2DIm(J, u_temp) - double(I), OF_par);
    end

end