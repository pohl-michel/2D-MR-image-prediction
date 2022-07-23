function [ u_temp ] = iterative_2Dof( I, J, u_temp, OF_par )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    grad_I = spatial_grad_2D(I, OF_par);
    for k = 1:OF_par.nb_iter
        fprintf('\t\t refinement iteration n°%d \n', k); 
        u_temp = u_temp + OF_LK_2D( grad_I, translate2DIm(J, u_temp) - I, OF_par);
    end

end

