function save_main_DVF_jpg(beh_par, path_par, disp_par, OF_par, im_par, br_model_par, W, Xtrain_mean)

    % Important : en gros la stratégie est :
    %    - ouvrir une figure
    %    - y afficher l'image à t avec imshow en améliorant le contraste
    %    - y afficher mon flot optique avec quiver
    %    - sauvegarder (par exemple avec 'print' si png) 
    %           Rq : je ne peux pas sauvegarder directement avec imshow car la superposition de l'image et du flot optique n'est pas une image au sens d'un
    %           tableau 2D
    %    - puis si png avec RemoveWhiteSpace supprimer les blancs autour du fichier png qui est alors une image
    
    %% Initialization
    
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
    
    %% First we plot the temporal mean of the DVF
 
    DVF_mean = zeros(im_par.W, im_par.L, 2);  
    
    % We "reconstruct" the main cpt_idx-th component from the matrix W 
    for x=1:im_par.L
        for y=1:im_par.W
            px_lin_idx = x + (y-1)*im_par.L; 
            DVF_mean(y,x,1) = Xtrain_mean( 2*px_lin_idx-1); 
                %linear indexing here because Xtrain_mean is a line vector
            DVF_mean(y,x,2) = Xtrain_mean( 2*px_lin_idx);                
        end
    end 

    if beh_par.CROP_FOR_DISP_SAVE
       DVF_mean_temp = DVF_mean(im_par.y_m:im_par.y_M, im_par.x_m:im_par.x_M, :);
       DVF_mean = DVF_mean_temp;
    end        

    f = figure;        
    imshow(im_t1, []);      
    set(gcf, 'units','normalized','outerposition',[0 0 1 1]);

    hold on
    quiver(X,Y,disp_par.arrow_scale_factor*(DVF_mean(:,:,1).*G), disp_par.arrow_scale_factor*(DVF_mean(:,:,2).*G), 'Autoscale', 'off', 'linewidth', disp_par.OF_arw_width, ...
        'color', [1 1 1]);
    hold off

    filename = write_DVFmean_jpg_filename( beh_par, OF_par, path_par );
    set(gca,'position',[0 0 1 1],'units','normalized');           
    set(gcf, 'InvertHardCopy', 'off');
    print(filename, '-djpeg', disp_par.OF_res);
        % print(filename, '-dpng', '-r1200'); % premiere version
        % je pourrais utiliser la fonction saveas mais elle ne permet pas
        % de controler la resolution de l'image enregistree, qui est trop
        % basse pour analyser les resultats

    close(f);
    
    %% Then we plot each component

    DVF_cpt = zeros(im_par.W, im_par.L, 2);    
    G = disp_par.PCA_cp_scale_factor*G;
    
    for cpt_idx=1:br_model_par.nb_pca_cp
        
        % We "reconstruct" the main cpt_idx-th component from the matrix W 
        for x=1:im_par.L
            for y=1:im_par.W
                px_lin_idx = x + (y-1)*im_par.L; 
                DVF_cpt(y,x,1) = W( 2*px_lin_idx-1, cpt_idx);
                DVF_cpt(y,x,2) = W( 2*px_lin_idx, cpt_idx);                
            end
        end
        
        if beh_par.CROP_FOR_DISP_SAVE
           DVF_cpt_temp = DVF_cpt(im_par.y_m:im_par.y_M, im_par.x_m:im_par.x_M, :);
           DVF_cpt = DVF_cpt_temp;
        end        
        
        f = figure;        
        imshow(im_t1, []);      
        set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
        
        hold on
        quiver(X,Y,disp_par.arrow_scale_factor*(DVF_cpt(:,:,1).*G), disp_par.arrow_scale_factor*(DVF_cpt(:,:,2).*G), 'Autoscale', 'off', 'linewidth', disp_par.OF_arw_width, ...
            'color', [1 1 1]);
        hold off
            
        filename = write_DVF_component_jpg_filename( beh_par, OF_par, path_par, cpt_idx, disp_par );
        set(gca,'position',[0 0 1 1],'units','normalized');           
        set(gcf, 'InvertHardCopy', 'off');
        print(filename, '-djpeg', disp_par.OF_res);
            % print(filename, '-dpng', '-r1200'); % premiere version
            % je pourrais utiliser la fonction saveas mais elle ne permet pas
            % de controler la resolution de l'image enregistree, qui est trop
            % basse pour analyser les resultats
        
        close(f);
        
        DVF_cpt(:) = 0;
        
    end
    
end