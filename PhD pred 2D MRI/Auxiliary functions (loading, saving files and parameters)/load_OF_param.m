function [ OF_par ] = load_OF_param( path_par )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

OF_calc_param_file = sprintf('%s\\%s', path_par.input_im_dir, path_par.OFpar_filename);
opts = detectImportOptions(OF_calc_param_file);
opts = setvartype(opts,'double');
opts.DataRange = '2:2'; % pour pouvoir écrire commentaires sur les variables en dessous ds fichier excel
OF_par = table2struct(readtable(OF_calc_param_file,opts)); 

switch(OF_par.grad_meth)
    case 1
        OF_par.grad_meth_str = 'ctrl diff grdt';
    case 2
        OF_par.grad_meth_str = 'Schaar grdt';
end


end

