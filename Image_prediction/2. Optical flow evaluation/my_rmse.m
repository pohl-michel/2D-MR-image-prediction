function [ rms ] = my_rmse( I, J, EVALUATE_IN_ROI, im_par)
% Computes the RMSE between two images I and J
% if EVALUATE_IN_ROI is set to true, the RMS error is calculated using only the region of interest.
% 
% 
% Author : Pohl Michel
% Date : Nov. 15th, 2022
% Version : v1.0
% License : 3-clause BSD License

    if EVALUATE_IN_ROI
        Wevl = im_par.y_M - im_par.y_m + 1;
        Levl = im_par.x_M - im_par.x_m + 1;
        x_eval_m = im_par.x_m;
        x_eval_M = im_par.x_M;
        y_eval_m = im_par.y_m;
        y_eval_M = im_par.y_M;
    else
        Wevl = im_par.W;
        Levl = im_par.L;
        x_eval_m = 1;
        x_eval_M = im_par.L;
        y_eval_m = 1;
        y_eval_M = im_par.W; 
    end

    pix_errors = zeros(Wevl, Levl, 'single');

    for x=x_eval_m:x_eval_M
        for y=y_eval_m:y_eval_M
                x_tab = x - x_eval_m + 1;
                y_tab = y - y_eval_m + 1;              
                pix_errors(y_tab, x_tab) = I(y,x) - J(y,x);
        end
    end

    rms = sqrt(sum(sum(((1/sqrt(Wevl*Levl))*pix_errors).^2)));

end
