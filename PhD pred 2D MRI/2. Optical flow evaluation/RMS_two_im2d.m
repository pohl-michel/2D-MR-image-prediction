function [ rms_two_im3d ] = RMS_two_im2d( I, J, EVALUATE_IN_ROI, im_par)
% Calculation of the RMS between the intensity levels of 2 images of the same size I & J
% if EVALUATE_IN_ROI is true, then the RMS is calculated only in the region of interest.

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

    rms_two_im3d = sqrt(sum(sum(((1/sqrt(Wevl*Levl))*pix_errors).^2)));

end
