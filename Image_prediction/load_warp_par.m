function [ warp_par ] = load_warp_par()
% Returns the structure warp_par, which contains information about the parameters used for warping the image.
%
% Author : Pohl Michel
% Date : August 31st, 2020
% Version : v1.0
% License : 3-clause BSD License


    warp_par.kernel_idx = 1;
        % 1 = gaussian kernel
        % 2 = averaging kernel
    
        switch(warp_par.kernel_idx)
            case 1
               warp_par.kernel_str = 'gaussian kernel';
            case 2
               warp_par.kernel_str = 'averaging kernel';
        end
        
    warp_par.kernel_appl_meth = 1;
        % 1 = matrix computation
        % 2 = pointwise computation
        
    % standard deviation in the case of a gaussian kernel
    warp_par.sg_fw_wrp = 0.5; 
        
    warp_par.min_filter_dim = 3;
        % integer value
        % if this value is too low, black pixels may occer in the warped image
        
    % filter_dim : half size of the square filter
    switch(warp_par.kernel_idx)
        case 1 % gaussian kernel
           warp_par.filter_dim = max(warp_par.min_filter_dim, ceil(2*warp_par.sg_fw_wrp));
        case 2 % averaging kernel
           warp_par.filter_dim = warp_par.min_filter_dim;
    end
         
    warp_par.nb_runs_for_cc_eval = 5;    
    
end