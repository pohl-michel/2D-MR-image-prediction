function save_2DOF(beh_par, path_par, disp_par, OF_par, im_par)

    % Important : en gros la stratégie est :
    %    - ouvrir une figure
    %    - y afficher l'image à t avec imshow en améliorant le contraste
    %    - y afficher mon flot optique avec quiver
    %    - sauvegarder (par exemple avec 'print' si png) 
    %           Rq : je ne peux pas sauvegarder directement avec imshow car la superposition de l'image et du flot optique n'est pas une image au sens d'un
    %           tableau 2D
    %    - puis si png avec RemoveWhiteSpace supprimer les blancs autour du fichier png qui est alors une image
    
    % loading image at time t=1
    im_t1 = enhance_2Dim(load_crop_filter2D(1, beh_par.CROP_FOR_DISP_SAVE, false, 0, im_par, path_par.input_im_dir), true);    
        % the image is cropped if we select a specific region where the optical flow should be displayed (beh_par.CROP_FOR_DISPLAY = 1)
        % The image is not filtered because it is used to display the optical flow here (FILTER = false)
    [H, L] = size(im_t1);
 
    x = 1:L; % coordinates in the image
    y = 1:H;
    [X,Y] = meshgrid(x,y);
    % the matrices X and Y have the same dimensions : H*L 
    
    % creation of a mask for displaying arrows spaced by dist_vec only.
    G = zeros(H,L);
    i_max = floor((L-1)/disp_par.dist_vec);
    j_max = floor((H-1)/disp_par.dist_vec);
    for i = 0:i_max % x-coordinate / dist_vec
        for j = 0:j_max % y-coordinate / dist_vec
            G(1+j*disp_par.dist_vec,1+i*disp_par.dist_vec) = 1;
        end
    end
    
    for t=2:im_par.nb_im
        % Rq : la variable im_par est stockée dans workspace_var_filename donc elle n'est pas en argument de display_save_OF    
        
        % loading optical flow at time t
        OF_t_filename = write_2DOF_t_mat_filename( OF_par, path_par, t );
        load(OF_t_filename, 'u_t');
        if beh_par.CROP_FOR_DISP_SAVE
           v_temp = u_t(im_par.y_m:im_par.y_M, im_par.x_m:im_par.x_M, :);
           u_t = v_temp;
        end
        
        f = figure;        
        imshow(im_t1, []);      
        set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
        
        hold on
        quiver(X,Y,disp_par.arrow_scale_factor*(u_t(:,:,1).*G), disp_par.arrow_scale_factor*(u_t(:,:,2).*G), 'Autoscale', 'off', 'linewidth', disp_par.OF_arw_width, ...
            'color', [1 1 1]);
        hold off
        
        if beh_par.SAVE_OF_JPG
            
            filename = write_2DOF_t_png_filename( beh_par, OF_par, path_par, t );
            set(gca,'position',[0 0 1 1],'units','normalized');           
            set(gcf, 'InvertHardCopy', 'off');
            print(filename, '-djpeg', disp_par.OF_res);
                % print(filename, '-dpng', '-r1200'); % premiere version
                % je pourrais utiliser la fonction saveas mais elle ne permet pas
                % de controler la resolution de l'image enregistree, qui est trop
                % basse pour analyser les resultats
        end
        
        close(f);
        
    end
    
end