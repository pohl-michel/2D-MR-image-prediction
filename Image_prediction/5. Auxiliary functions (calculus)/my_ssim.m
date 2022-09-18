function [my_ssim_val] = my_ssim(I, J, EVALUATE_IN_ROI, im_par)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    if EVALUATE_IN_ROI
        Itemp = I(im_par.y_m:im_par.y_M, im_par.x_m:im_par.x_M);
        Jtemp = J(im_par.y_m:im_par.y_M, im_par.x_m:im_par.x_M);   
    else
        Itemp = I;
        Jtemp = J;           
    end

    [my_ssim_val, ~] = ssim_index(Itemp, Jtemp);


end

