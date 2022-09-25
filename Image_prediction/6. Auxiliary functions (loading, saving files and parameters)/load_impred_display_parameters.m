function [ disp_par ] = load_impred_display_parameters(path_par)
% Load the parameters concerning image display,
% which are initially stored in the file named path_par.disp_par_filename.
%
% Author : Pohl Michel
% Date : Sept 18th, 2022
% Version : v1.1
% License : 3-clause BSD License


    disp_par_file = sprintf('%s\\%s', path_par.input_im_dir, path_par.disp_par_filename);
    opts = detectImportOptions(disp_par_file);
    opts = setvartype(opts,'double');
    opts.DataRange = '2:2'; % pour pouvoir écrire commentaires sur les variables en dessous ds fichier excel
    disp_par = table2struct(readtable(disp_par_file, opts));

    disp_par.OF_res = sprintf('-r%d', int16(disp_par.OF_res));
    disp_par.wrp_im_res = sprintf('-r%d', int16(disp_par.wrp_im_res));
    disp_par.pred_plot_res = sprintf('-r%d', int16(disp_par.pred_plot_res));


end