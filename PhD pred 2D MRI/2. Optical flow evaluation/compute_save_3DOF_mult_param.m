function compute_save_3DOF_mult_param( OFeval_par, path_par, im_par)
% Compute and save optical flow for several parameters as specified in the structure OFeval_par
% This function is used for evaluating the influence of the parameters on the accuracy of the computed optical flow.

    length_sigma_LK_tab = length(OFeval_par.sigma_LK_tab);
    OF_par.sigma_init = OFeval_par.sigma_init;
    OF_par.sigma_subspl = OFeval_par.sigma_subspl;
    OF_par.epsilon_detG = OFeval_par.epsilon_detG;
    OF_par.grad_meth = OFeval_par.grad_meth;
    OF_par.grad_meth_str = OFeval_par.grad_meth_str;

    for sigma_LK_tab_idx = 1:length_sigma_LK_tab    
        OF_par.sigma_LK = OFeval_par.sigma_LK_tab(sigma_LK_tab_idx);
        for nb_layers = OFeval_par.nb_layers_min:OFeval_par.nb_layers_max
            OF_par.nb_layers = nb_layers;

            compute_3Dof(OF_par, im_par, path_par);
            % display_save_OF(beh_par, path_par, disp_par, OF_par, im_par);

        end
    end

end

