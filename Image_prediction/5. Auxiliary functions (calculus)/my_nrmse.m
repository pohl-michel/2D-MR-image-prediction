function [my_nrmse_val] = my_nrmse(Ipred, Iorg, EVALUATE_IN_ROI, im_par)
% Computes the nRMSE between the original image Iorg and the predicted image Ipred 
%
% Author : Pohl Michel
% Date : Sept 18th, 2022
% Version : v1.0
% License : 3-clause BSD License


    my_aux_func = @(x,y) sqrt(sum((x-y).^2));

    if EVALUATE_IN_ROI
        Iorg_temp = Iorg(im_par.y_m:im_par.y_M, im_par.x_m:im_par.x_M);
        Ipred_temp = Ipred(im_par.y_m:im_par.y_M, im_par.x_m:im_par.x_M);   
    else
        Iorg_temp = Iorg;
        Ipred_temp = Ipred;           
    end

    Iorg_temp = Iorg_temp(:);
    Ipred_temp = Ipred_temp(:);
    
    num = my_aux_func(Iorg_temp, Ipred_temp);
    den = my_aux_func(Iorg_temp, mean(Iorg_temp)); % broadcasting inside my_aux_func
    
    my_nrmse_val = num/den;

end
