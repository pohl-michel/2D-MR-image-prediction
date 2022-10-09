function [ warp_par ] = load_warp_par()
% Returns the structure warp_par, which contains information about the parameters used for warping the image.
%
% Author : Pohl Michel
% Date : August 31st, 2020
% Version : v1.0
% License : 3-clause BSD License


    warp_par.kernel = 'gaussian kernel';                 % two choices currently: 'gaussian kernel' or 'averaging kernel'        
    warp_par.kernel_appl_meth = 'matrix computation';    % two choices currently: 'matrix computation' or 'pointwise computation'        
        
    % standard deviation in the case of a gaussian kernel
    warp_par.sg_fw_wrp = 0.5; 
        
    warp_par.min_filter_dim = 3;
        % integer value; if this value is too low, black pixels may occer in the warped image
        
    % filter_dim : half size of the square filter
    switch warp_par.kernel
        case 'gaussian kernel'
           warp_par.filter_dim = max(warp_par.min_filter_dim, ceil(2*warp_par.sg_fw_wrp));
        case 'averaging kernel'
           warp_par.filter_dim = warp_par.min_filter_dim;
    end
         
    warp_par.nb_runs_for_cc_eval = 5;    
    
end