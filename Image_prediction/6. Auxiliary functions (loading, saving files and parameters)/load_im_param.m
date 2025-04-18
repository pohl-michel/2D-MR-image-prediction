function [ im_par ] = load_im_param(path_par)
% Load the parameters concerning the loaded image sequence,
% which are initially stored in the file named path_par.im_par_filename.
%
% Author : Pohl Michel
% Date : July 16th, 2020
% Version : v1.0
% License : 3-clause BSD License


    im_par_file = sprintf('%s\\%s', path_par.input_im_dir, path_par.im_par_filename);
    opts = detectImportOptions(im_par_file);
    opts.DataRange = '2:2'; % pour pouvoir �crire commentaires sur les variables en dessous ds fichier excel
    im_par = table2struct(readtable(im_par_file, opts));

    if ~isfield(im_par, "x_m")
        im_par.x_m = 1;
    end
    if ~isfield(im_par, "y_m")
        im_par.y_m = 1;
    end
    if ~isfield(im_par, "x_M")
        im_par.x_M = im_par.W;
    end
    if ~isfield(im_par, "y_M")
        im_par.y_M = im_par.L;
    end    

end

