function [ im ] = load_crop_filter2D(t, CROP, FILTER, sigma_init, im_par, input_im_dir)
% Return the image at time t.
% The image is cropped (resp. filtered) if CROP (resp. FILTER) is set to "true".
% The returned image is of type 'single' (that helps minimize the memory used).
%
% Remark (2021/01/27): works only with 2D dicom images at the moment
% Remark (2022/09/18): check again if I need to use the "squeeze" function
%
% Author : Pohl Michel
% Date : Sept 18th, 2022
% Version : v1.0
% License : 3-clause BSD License
       
 
    %fprintf('loading 2D image at time t = %d ...\n', t);
    
    switch im_par.imtype
        case {"dcm", "IMA"}
            im_filename = sprintf('%s\\image%d.%s',input_im_dir, t, im_par.imtype);
            im = single(squeeze(dicomread(im_filename)));
                % squeeze is necessary because when Matlab opens an image with dicomread, the 3rd dimension is a singleton - Note (2022/09/18): to check again
        case "mat"
            im_filename = sprintf('%s\\image%d.mat',input_im_dir, t);
            load(im_filename, 'im')
    end
        
    if CROP
        fprintf('cropping image at time t = %d \n', t);
        im = im(im_par.y_m:im_par.y_M, im_par.x_m:im_par.x_M);
    end

    if FILTER
        fprintf('low pass gaussian filtering of the image at time t = %d \n', t);
            im = floor(imgaussfilt(im, sigma_init));
                % 1) floor is necessary because otherwise, the filtered image has real values and in turn, enhance_brightness_contrast does not work well.  
                % 2) the variable type of "im" is still 'single' after this operation
    end

end