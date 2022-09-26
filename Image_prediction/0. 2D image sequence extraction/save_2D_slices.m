function save_2D_slices(disp_par, transformation_par, org_im_par, path_par, im_seq_idx)
% Extracts saggital slices of the original 3D image sequence, shifts that image sequence so that the new sequences starts in the middle of the breathing cycle,
% and saves that 2D image sequence in the corresponding folder in "2D images".
%
% Author : Pohl Michel
% Date : September 26th, 2022
% Version : v1.0
% License : 3-clause BSD License

    
    t_org_sq = transformation_par.t_init(im_seq_idx); % time in the original image sequence 
        
    for t=1:org_im_par.nb_im

        fprintf('\n')
        fprintf('Extracting 2D slice Xcs=%d from image %d/%d in sequence %s \n', transformation_par.Xcs(im_seq_idx), t, org_im_par.nb_im, path_par.input_im_seq);
        
        % Loading the image at time t
        crop_flag = false; filter_flag = false; sigma_init = 'whatever';
        I_3Dt = load_crop_filter_3Dim(t_org_sq, crop_flag, filter_flag, sigma_init, org_im_par, path_par.input_im_dir);
        
        % Slicing the 3D image
        sag_slice_t = transpose(squeeze(I_3Dt(transformation_par.Xcs(im_seq_idx),:,:)));
        %enhanced_xslice = enhance_2Dim( xslice, enhance_flag); % 16 bit image in the original enhance function (May 2021 version)
        
        % Invert the images along the z axis (for better display - whether this is needed or not depends on the input sequence though)
        sag_slice_aux = sag_slice_t;
        for z=1:org_im_par.H
            sag_slice_t(z,:) = sag_slice_aux(org_im_par.H - z +1, :);
        end

        % saving dcm images of the cross sections
        dcm_filename = sprintf('%s//image%d.dcm', path_par.output_im_dir, t);
        dicomwrite(uint8(sag_slice_t), dcm_filename)
        
        % saving jpg images of the cross sections
        crop_flag = false; enhance_flag = false;  
        im_filename_suffix = sprintf('image%d', t);
        save_crop_enhance_2Dim_jpg(sag_slice_t, im_filename_suffix, crop_flag, enhance_flag, disp_par, path_par, 0, 0, 0, 0, t);
        
        t_org_sq = t_org_sq+1;
        if (t_org_sq == (1+org_im_par.nb_im))
            t_org_sq = 1;
        end    

    end

end

