function [ im ] = load_crop_filter2D(t, CROP, FILTER, sigma_init, im_par, input_im_dir)
% Return the image at time t of the image sequence.
% The image is cropped and/or filtered if specified in the behavior parameters.
% The returned image is of type 'single' (in order to minimize the memory used).
% 2021 - 01 - 27 : works only with 2D dicom images at the moment
       
%     fprintf('loading 2D image at time t = %d ...\n', t);
    
    switch im_par.imtype
        case "dcm"
            im_filename = sprintf('%s\\image%d.dcm',input_im_dir, t);
            im = single(squeeze(dicomread(im_filename)));
                % squeeze is necessary because when Matlab opens a 3D image with dicomread the 3rd dimension is a singleton
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
                % 1) floor is necessary because otherwise filtered_image has real
                % values and then enhance_brightness_contrast do not work well.  
                % 2) the matrix type is still 'single after this operation'
    end

end