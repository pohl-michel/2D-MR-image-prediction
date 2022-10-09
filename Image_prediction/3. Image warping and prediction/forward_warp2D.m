function [Iwarped, im_warp_calc_time] = forward_warp2D(I, u, warp_par)
% forward_warp warps the 2D image I forward according to the displacement field u
% using a kernel described in warp_par
%
% Reminder :
% u(y,x,1) is the x-value of the optical flow and coordinates (x,y)
% u(y,x,2) is the y-value of the optical flow and coordinates (x,y)
% 
% Possible improvement: write that in C and/or use GPU acceleration.
%
% Author : Pohl Michel
% Date : Sept 19th, 2022
% Version : v1.0
% License : 3-clause BSD License


    [W, L] = size(I);
    [~,~,~,nb_runs] = size(u);
    nb_runs = min(warp_par.nb_runs_for_cc_eval, nb_runs);
        % because sometimes we perform RMS eval on time signal and c.c. eval on image signal and c.c. eval takes much much time 
    Iwarped = zeros(W, L, nb_runs, 'single');
    ksum = zeros(W, L, 'single'); % auxiliary coefficient 
    K = kernel_function2D(warp_par); 
    d = warp_par.filter_dim; % d : half size of the square filter
    [X,Y] = meshgrid(1:(2*d)); % preparing a small coordinate grid fot computing the kernel array used in the gaussian smoothering    
    im_warp_calc_time_tab = zeros(nb_runs, 1, 'single');
    
    for run_idx = 1:nb_runs
    % Selection of a pixel in the image I to be warped ("org" means "arrow origin")
        tic
        for x_org = 1:L
            for y_org = 1:W   

                x_tip = x_org + u(y_org,x_org,1, run_idx);
                y_tip = y_org + u(y_org,x_org,2, run_idx);

                Ex_tip = floor(x_tip);
                Ey_tip = floor(y_tip);

                x_m = max([1, Ex_tip - d + 1]);
                x_M = min([L, Ex_tip + d]);

                y_m = max([1, Ey_tip - d + 1]);
                y_M = min([W, Ey_tip + d]);  

                if (x_m <= x_M)&&(y_m <= y_M)

                    switch warp_par.kernel_appl_meth % kernel application method
                        case 'matrix computation'

                            mu_x = x_tip - Ex_tip + d;
                            mu_y = y_tip - Ey_tip + d;
                            Kloc = K(mu_y, mu_x); % loc means "local"
                            Kloc_mat = Kloc(Y,X);

                            if (x_M - x_m + 1 ~= 2*d)||(y_M - y_m + 1 ~= 2*d)
                                % cropping the kernel matrix is necessary due to the image borders
                                x_m_kern = x_m - Ex_tip + d;
                                x_M_kern = x_M - Ex_tip + d;
                                y_m_kern = y_m - Ey_tip + d;
                                y_M_kern = y_M - Ey_tip + d;
                                Kloc_mat = Kloc_mat(y_m_kern:y_M_kern, x_m_kern:x_M_kern);
                            end

                            ksum(y_m:y_M, x_m:x_M) = ksum(y_m:y_M, x_m:x_M) + Kloc_mat;
                            Iwarped(y_m:y_M, x_m:x_M, run_idx) = Iwarped(y_m:y_M, x_m:x_M, run_idx) + I(y_org, x_org)*Kloc_mat;

                        case 'pointwise computation'

                            Kloc = K(y_tip, x_tip); % loc means "local"
                            for x = x_m:x_M
                                for y = y_m:y_M
                                        ksum(y,x) = ksum(y,x) + Kloc(y,x);
                                        Iwarped(y,x, run_idx) = Iwarped(y,x, run_idx) + I(y_org, x_org)*Kloc(y,x);
                                end
                            end

                    end

                end
                    
            end
        end

        ksum(ksum == 0) = 1; % avoiding division by zero
        Iwarped(:, :, run_idx) = floor(Iwarped(:, :, run_idx)./ksum);
        im_warp_calc_time_tab(run_idx) = toc;
        ksum(:)=0; % this command resets the array to zero without allocating memory.

    end
    
    im_warp_calc_time = mean(im_warp_calc_time_tab);
    
end


function K = kernel_function2D(warp_par)
% Returns a function handle corresponding to a kernel function, used in forward warping
% filter_dim is the half size of the square filter
% 
% Author : Pohl Michel
% Date : Sept 19th, 2022
% Version : v1.0
% License : 3-clause BSD License


    switch warp_par.kernel
        
        case 'gaussian kernel'
            C = 1/(2*(warp_par.sg_fw_wrp^2)); % constant computed in advance to make the calculations faster
            K = @(mu_y, mu_x) @(y,x) exp(-C*((x-mu_x).^2 + (y-mu_y).^2));  
            
        case 'averaging kernel'
            K = @(mu_y, mu_x) @(y,x) ones(size(y,1), size(x,1), 'single'); 
            % K = @(mu_y, mu_x) @(y,x) 1 does not work properly
    end

end
