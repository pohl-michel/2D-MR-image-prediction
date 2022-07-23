function eval_OF_results = compute_2Dof( OF_par, im_par, path_par)
% Compute optical flow between t = 1 et t_current
% and saves the result in a mat file
% The optical flow is saved rather than returned as an output because it takes up much memory.

    fprintf('OPTICAL FLOW CALCULATION \n');
    fid = 1; % screen display
    fprintfOFpar( fid, OF_par );
    
    % Calculation of the pyramidal representation of the images at t=1 and t=2    
    initial_filtering_flag = true;
    crop_flag = false;
    
    t_init = 1;
    I = load_crop_filter2D(t_init, crop_flag, initial_filtering_flag, OF_par.sigma_init, im_par, path_par.input_im_dir);
    pyr_I = im_to_pyr2D( I, OF_par );
    
    time_calc_tab = zeros(im_par.nb_im - t_init, 1);
    
    for t = (t_init + 1):im_par.nb_im 
 
        J = load_crop_filter2D(t, crop_flag, initial_filtering_flag, OF_par.sigma_init, im_par, path_par.input_im_dir); 
        pyr_J = im_to_pyr2D( J, OF_par );
        
        % Calculation of optical flow between images at 1 and t
        tic
        u_t = pyr_LK_2D( pyr_I, pyr_J, OF_par); % la valeur de u_t n'est pas utilisée mais u_t est enregistré dans un fichier .mat - ignorer le warning
        time_calc_tab(t-t_init) = toc;
        
        fprintf('saving optical flow between t=1 and t = %d \n', t);
        OF_t_filename = write_2DOF_t_mat_filename( OF_par, path_par, t );
        save(OF_t_filename, 'u_t');   
        clear u_t ;

    end

    eval_OF_results.OF_calc_time = mean(time_calc_tab);
    
end

