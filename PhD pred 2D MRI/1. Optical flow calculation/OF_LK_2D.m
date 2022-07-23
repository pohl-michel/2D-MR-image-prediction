function [ u ] = OF_LK_2D( spatial_grad_I, deltaIJ, OF_par )
% This function computes the optical flow between two images I and J using the Lucas Kanade method.
% The spatial gradient of I and the difference J - I are used as inputs, not the original images I and J.
% The output is a two dimensional vector u.
%
% u(y,x,1) is the optical flow vector x-value at the position (x,y), and 
% u(y,x,2) -------------------------- y-value ---------------------.
%
% Velocity vectors are not calculated at the borders of the image (no
% padding) so if x = 1 or x = L or y = 1 or y = H then u(y,x,k) = 0. (k= 1 or 2)

    [im_2nd_dim, im_1st_dim, ~] = size(spatial_grad_I);
    u = zeros(im_2nd_dim, im_1st_dim, 2, 'single');

    % use of the imgaussfit function in matlab for convolving with a gaussian
    % function / standard deviation : std_dev_LK / square filter of size
    % 2*ceil(2*std_dev_LK)+1 / padding replicating the borders
    
    % in place calculations while computing M and b in order to save memory

    % Calculus of the matrices M and b;
    M = zeros(im_2nd_dim,im_1st_dim,2,2, 'single');
    b = zeros(im_2nd_dim,im_1st_dim,2, 'single');

    % The function imgaussfilt was introduced in R2015a.
    M(:,:,1,1) = imgaussfilt(spatial_grad_I(:,:,1).^2, OF_par.sigma_LK);                           % Ix_squared - then filtering
    M(:,:,1,2) = imgaussfilt(spatial_grad_I(:,:,1).*spatial_grad_I(:,:,2), OF_par.sigma_LK);       % Ix_Iy      - then filtering
    M(:,:,2,1) = M(:,:,1,2);
    M(:,:,2,2) = imgaussfilt(spatial_grad_I(:,:,2).^2, OF_par.sigma_LK);                           % Iy_squared - then filtering
    b(:,:,1) = - imgaussfilt(spatial_grad_I(:,:,1).*deltaIJ, OF_par.sigma_LK);                     % Ix_It      - then filtering
    b(:,:,2) = - imgaussfilt(spatial_grad_I(:,:,2).*deltaIJ, OF_par.sigma_LK);                     % Iy_It      - then filtering

    % loop over position to solve the linear system
    % memory allocation before loop in order to avoid using the "squeeze" function within the loop, which seems to allocate memory
    M_temp = zeros(2,2);
    b_temp = zeros(2,1);

    for x = 2:im_1st_dim-1
        for y = 2:im_2nd_dim-1
            
            M_temp(1,1) = M(y,x,1,1);
            M_temp(1,2) = M(y,x,1,2);
            M_temp(2,1) = M(y,x,2,1);
            M_temp(2,2) = M(y,x,2,2);
            b_temp(1,1) = b(y,x,1);
            b_temp(2,1) = b(y,x,2);
            
            det_M = det(M_temp);
            if (abs(det_M) > OF_par.epsilon_detG)
                u_temp = M_temp\b_temp;
            else
                pinv_M = pinv(M_temp);
                u_temp = pinv_M*b_temp;
            end
            
            u(y,x,1) = u_temp(1);
            u(y,x,2) = u_temp(2);

        end
    end

end