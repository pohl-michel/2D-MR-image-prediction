function display_save_DVF(u, im, G, filename, im_par, beh_par, disp_par, SAVE_OF_JPG)
% Saves the deformation vector field u on top of the image im (using a mask G so that it does not appear to dense on the display).
% The saved filename is "filename".
%
% Author : Pohl Michel
% Date : September 25th, 2022
% Version : v2.0
% License : 3-clause BSD License


    [H, L] = size(im);
    x = 1:L; % coordinates in the image
    y = 1:H;
    [X,Y] = meshgrid(x,y); % the matrices X and Y have the same dimensions : H*L 
    
    if beh_par.CROP_FOR_DISP_SAVE
        DVF_temp = u(im_par.y_m:im_par.y_M, im_par.x_m:im_par.x_M, :);
        u = DVF_temp;
    end        
    
    f = figure;        
    imshow(im, []);      
    set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
    
    hold on
    quiver(X,Y,disp_par.arrow_scale_factor*(u(:,:,1).*G), disp_par.arrow_scale_factor*(u(:,:,2).*G), 'Autoscale', 'off', 'linewidth', disp_par.OF_arw_width, ...
        'color', [1 1 1]);
    hold off
    
    if SAVE_OF_JPG
        set(gca,'position',[0 0 1 1],'units','normalized');           
        set(gcf, 'InvertHardCopy', 'off');
        % print(filename, '-dpng', '-r1200'); % if saving as png file
            % Rk: saveas does not enable controlling the resolution of the saved image
        print(filename, '-djpeg', disp_par.OF_res);
    end

    close(f);

end