  function [ spatial_grad_I ] = spatial_grad_2D(I, OF_par)
% Computation of the spatial gradient of a 2D image I
% If the dimension of the tensor I is H*L, then the dimension of delta_I is (H,L,2)
% if x = 1 or x = L or y = 1 or y = H or z = 1 or z = Zmax or t = 1 or t = T, then
% delta_I(y,x,t,k) = 0.
%
% Author : Pohl Michel
% Date : Sept 18th, 2022
% Version : v1.0
% License : 3-clause BSD License


    [W, L] = size(I);
    spatial_grad_I = zeros(W, L, 2, 'single');
    
	switch OF_par.grad_meth
    
        case 1 % Central difference method
    
            for x = 2:L-1
               spatial_grad_I(:,x,1) = I(:,x+1)-I(:,x-1);
            end

            for y = 2:W-1
                spatial_grad_I(y,:,2) = I(y+1,:)-I(y-1,:);
            end

            spatial_grad_I = 0.5*spatial_grad_I;
        
        case 2 % Schaar gradient
           % https://en.wikipedia.org/wiki/Image_gradient
           % https://en.wikipedia.org/wiki/Sobel_operator
 
           K_2DSchaar_x = 0.1*[1, 0, -1; 3, 0, -3; 1, 0, -1];
           spatial_grad_I(:,:,1) = conv2(I, K_2DSchaar_x, 'same');
           
           K_2DSchaar_y = 0.1*[1, 3, 1; 0, 0, 0; -1, -3, -1];
           spatial_grad_I(:,:,2) = conv2(I, K_2DSchaar_y, 'same');
        
	end
    
end