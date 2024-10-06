function [my_nrmse_val] = my_nrmse(Ipred, Iorg, EVALUATE_IN_ROI, im_par)
% MY_NRMSE Computes the Normalized Root Mean Square Error (nRMSE) between the predicted and original images.
% The evaluation can be performed over the entire image or restricted to a specific region of interest (ROI).
%
% INPUTS:
%  - Ipred (matrix): Predicted image . It should have the same dimensions as Iorg.
%  - Iorg (matrix): The reference or ground-truth image. It should have the same dimensions as Ipred.
%  - EVALUATE_IN_ROI (boolean): If true, nRMSE is computed only within a specific ROI defined by 'im_par'. Otherwise, the entire image is used.
%  - im_par (struct): Contains image parameters, including the ROI boundaries if needed (x_m, x_M, y_m, y_M)
%
% OUTPUTS:
% - my_nrmse_val (scalar): The computed nRMSE value, which indicates the similarity between `Ipred` and `Iorg`. A lower value indicates higher similarity.
%
% Author : Pohl Michel
% License : 3-clause BSD License

    % Converting to double precision
    Ipred = double(Ipred);
    Iorg = double(Iorg);

    % Auxiliary function computing the root mean square difference between two vectors
    my_aux_func = @(x,y) sqrt(sum((x-y).^2));

    % Extracting the ROI if requested
    if EVALUATE_IN_ROI
        Iorg_temp = Iorg(im_par.y_m:im_par.y_M, im_par.x_m:im_par.x_M);
        Ipred_temp = Ipred(im_par.y_m:im_par.y_M, im_par.x_m:im_par.x_M);   
    else
        Iorg_temp = Iorg;
        Ipred_temp = Ipred;           
    end

    % Flattening the ROI or entire image
    Iorg_temp = Iorg_temp(:);
    Ipred_temp = Ipred_temp(:);
    
    % nRMSE computation using the flattened vectors
    num = my_aux_func(Iorg_temp, Ipred_temp);
    den = my_aux_func(Iorg_temp, mean(Iorg_temp)); % broadcasting inside my_aux_func
    my_nrmse_val = num/den;

end
