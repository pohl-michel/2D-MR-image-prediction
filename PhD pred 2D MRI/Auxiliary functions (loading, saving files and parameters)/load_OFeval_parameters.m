function [ OFeval_par ] = load_OFeval_parameters()
% parameters used when evaluating the optical flow

% parametres à changer manuellement en fonction de la sequence d'image utilisée
OFeval_par.epsilon_detG = 0.001;
OFeval_par.sigma_init_tab = [0.1, 0.5, 1.0];
OFeval_par.sigma_subspl_tab = [0.1, 0.5, 1.0];

% choix des paramètres de calcul du flot optique
OFeval_par.grad_meth = 2;
    % 2 : gradient de Shaar utilisant un 3-voisinage
    % 1 : méthode normale

switch(OFeval_par.grad_meth)
    case 1
        OFeval_par.grad_meth_str = 'ctrl diff grdt';
    case 2
        OFeval_par.grad_meth_str = 'Schaar grdt';
end    
    
OFeval_par.sigma_LK_tab = [1.0, 2.0, 3.0, 4.0];
OFeval_par.nb_layers_min = 1;
OFeval_par.nb_layers_max = 3;
OFeval_par.nb_min_iter = 1;
OFeval_par.nb_max_iter = 3;

end