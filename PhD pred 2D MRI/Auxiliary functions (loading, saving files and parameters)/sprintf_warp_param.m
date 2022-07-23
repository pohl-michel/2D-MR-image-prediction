function [ warp_par_str ] = sprintf_warp_param( warp_par )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here

    switch(warp_par.kernel_idx)
        case 1 % gaussian kernel
            warp_par_str = sprintf('gaussian kernel filter_size %d sg_warp %f', warp_par.filter_dim, warp_par.sg_fw_wrp);
        case 2 % averaging kernel
            warp_par_str = sprintf('averaging kernel filter size %d', warp_par.filter_dim);
    end

end

