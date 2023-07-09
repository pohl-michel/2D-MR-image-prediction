function [ J ] = translate2DIm( I, u )
% This function translates the image I by the vector field u
% informally J(vec_x) = I(vec_x + u(vec_x)) 
%
% Author : Pohl Michel
% Date : Sept 18th, 2022
% Version : v1.0
% License : 3-clause BSD License


    [H, L] = size(I);
    J = zeros(H, L);
    for x=1:L
        for y =1:H
            J(y, x) = my_bil_interp(I, H, L, y+u(y,x,2),x+u(y,x,1));
        end
    end

end

