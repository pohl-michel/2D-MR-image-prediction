function [ im_par ] = load_im_param( path_par )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

im_par_file = sprintf('%s\\%s', path_par.input_im_dir, path_par.im_par_filename);
opts = detectImportOptions(im_par_file);
opts.DataRange = '2:2'; % pour pouvoir écrire commentaires sur les variables en dessous ds fichier excel
im_par = table2struct(readtable(im_par_file, opts));

end

