function [ u_bottom ] = pyr_LK_2D( pyr_I, pyr_J, OF_par )
% Given the pyramidal representation pyr_I and pyr_J of two images I and J,
% this function computes the optical flow between I and J using a pyramidal and iterative method.
%
% Author : Pohl Michel
% Date : Sept 18th, 2022
% Version : v1.0
% License : 3-clause BSD License

    u_temp = cell(OF_par.nb_layers,1);
    [top_w, top_l] = size(pyr_I{OF_par.nb_layers});
    u_temp{OF_par.nb_layers} = zeros(top_w, top_l, 2);
        % u_temp{lyr_idx}(y,x,1) contains the x coordinate of the original guess at layer lyr_idx and pixel (x,y)
        % u_temp{lyr_idx}(y,x,2) ------------ y -----------------------------------------------------------------
    aux = @(x) floor((x+1)/2); % auxiliary function for indexing
             
    for lyr_idx = OF_par.nb_layers:(-1):1 % "-1" for decrementation

        % Optical flow refinement 
        fprintf('\t Refining the optical flow at layer %d \n', lyr_idx);  
        u_temp{lyr_idx} = iterative_2Dof( pyr_I{lyr_idx}, pyr_J{lyr_idx}, u_temp{lyr_idx}, OF_par );
            % I = pyr{lyr_idx}; % representation of the 1st image of the sequence at layer lyr_idx
            % J = pyr{lyr_idx}; % --------------------- 2nd --------------------------------------
            % u_temp{lyr_idx} in the arguments is the guess at layer lyr_idx ("g" in the paper from Intel, 2010)
            % u_temp{lyr_idx} in the output is the refined optical flow at layer lyr_idx ("g+d" in the paper from Intel, 2010)
        
        % guess g at the layer below
        if (lyr_idx>1)
            fprintf('\t Guessing the optical flow at the layer %d \n', lyr_idx-1);           
            [W_below, L_below] = size(pyr_I{lyr_idx-1});            
            Y = aux(1:W_below);
            X = aux(1:L_below);
            u_temp{lyr_idx-1} = 2*u_temp{lyr_idx}(Y,X,:);
                % equivalent to a loop over (x,y): u_temp{lyr_idx-1}(y, x, :) = 2*u_temp{lyr_idx}(aux(y), aux(x), :)
        end

    end

    u_bottom = u_temp{1};

end