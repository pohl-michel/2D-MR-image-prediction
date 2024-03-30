function [ corr ] = corr_two_im2d( I, J, EVALUATE_IN_ROI, im_par)
% Calculation of the correlation coefficient of 2 images of the same size I & J (viewed as vectors).
% If EVALUATE_IN_ROI is true, then the RMS is calculated only in the region of interest.
%
% Author : Pohl Michel
% Date : Sept 18th, 2022
% Version : v1.0
% License : 3-clause BSD License

    I = double(I);
    J = double(J);

    if EVALUATE_IN_ROI
        Itemp = I(im_par.y_m:im_par.y_M, im_par.x_m:im_par.x_M);
        Jtemp = J(im_par.y_m:im_par.y_M, im_par.x_m:im_par.x_M);   
    else
        Itemp = I;
        Jtemp = J;           
    end

    temp_corr_mat = corrcoef(Itemp(:), Jtemp(:));
    corr = temp_corr_mat(1,2);

end