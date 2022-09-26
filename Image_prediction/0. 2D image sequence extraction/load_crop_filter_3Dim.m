function [ im ] = load_crop_filter_3Dim(t, CROP, FILTER, sigma_init, im_par, input_im_dir)
% Returns the image at time t of the image sequence.
% The image is cropped and/or filtered if specified in the behavior parameters.
% The returned image is of type 'single' (in order to minimize the memory used).
%
% Author : Pohl Michel
% Date : September 26th, 2022
% Version : v1.0
% License : 3-clause BSD License


    fprintf('loading 3D image at time t = %d ...\n', t);
    
    switch im_par.imtype
        case 'dcm' %dicom
            im_filename = sprintf('%s\\image%d.dcm',input_im_dir, t);
            im = squeeze(dicomread(im_filename));
                % squeeze is necessary because when Matlab opens a 3D image with dicomread the 3rd dimension is a singleton
        case 'mha' %metaimage
            im_filename = sprintf('%s\\image%d.mha',input_im_dir, t);
            [im,~] = ReadData3D(string(im_filename));
    end
                
    if CROP
        fprintf('cropping image at time t = %d \n', t);
        im = im(im_par.y_m:im_par.y_M, im_par.x_m:im_par.x_M, im_par.z_m:im_par.z_M);
    end

    if FILTER
        fprintf('low pass gaussian filtering of the image at time t = %d \n', t);
            im = floor(imgaussfilt3(im, sigma_init));
                % 1) floor is necessary because otherwise filtered_image has real
                % values and then enhance_brightness_contrast do not work well.  
                % 2) the matrix type is still 'single after this operation'
    end

end