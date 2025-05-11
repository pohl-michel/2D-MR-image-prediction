function [ warp_par ] = load_warp_par(pred_meth)
% LOAD_WARP_PAR Returns a structure `warp_par` containing parameters for image warping.
%
% INPUTS:
%   (None)
%
% OUTPUTS:
% - warp_par (struct): Structure containing the parameters used for image warping.
%
% Author : Pohl Michel
% License : 3-clause BSD License

    % The type of kernel used for warping (options: 'gaussian kernel' or 'averaging kernel')
    warp_par.kernel = 'gaussian kernel';                 
    
    % The method for applying the kernel (options: 'matrix computation' or 'pointwise computation')
    warp_par.kernel_appl_meth = 'matrix computation'; 
        
    % Standard deviation of the Gaussian kernel (relevant only if using 'gaussian kernel')
    warp_par.sg_fw_wrp = 0.5; 
        
    % Minimum filter dimension (square half size). Black pixels can appear in the warped image if this value is too low and motion amplitude is high.
    warp_par.min_filter_dim = 3;

    % Determining the filter dimension (square half size) based on the kernel type
    switch warp_par.kernel
        case 'gaussian kernel'
           warp_par.filter_dim = max(warp_par.min_filter_dim, ceil(2*warp_par.sg_fw_wrp)); % the filter size depends on the standard deviation
        case 'averaging kernel'
           warp_par.filter_dim = warp_par.min_filter_dim; % filter size set to the minimum filter dimension
    end
         
    % Number of runs for evaluating accuracy metrics (e.g., correlation coefficient) when warping
    switch(pred_meth)
        case {'RTRL', 'RTRL v2'} 
            warp_par.nb_runs_for_cc_eval = 5;
        case {'multivariate linear regression', 'LMS', 'univariate linear regression', 'no prediction', 'SVR'}  % deterministic
            warp_par.nb_runs_for_cc_eval = 1;
        otherwise % other (online) RNN methods and transformers
            warp_par.nb_runs_for_cc_eval = 25;
    end
end