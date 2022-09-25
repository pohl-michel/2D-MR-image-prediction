function G = return_mask_for_DVF_display(im, disp_par)
% creation of a mask for displaying arrows spaced by dist_vec only.
%
% Author : Pohl Michel
% Date : September 25th, 2022
% Version : v2.0
% License : 3-clause BSD License


    [H, L] = size(im);
    G = zeros(H,L);
    i_max = floor((L-1)/disp_par.dist_vec);
    j_max = floor((H-1)/disp_par.dist_vec);

    for i = 0:i_max % x-coordinate / dist_vec
        for j = 0:j_max % y-coordinate / dist_vec
            G(1+j*disp_par.dist_vec,1+i*disp_par.dist_vec) = 1;
        end
    end

end