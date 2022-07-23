clear all
close all
clc


%% PARAMETERS

% Behavior - parametres que je peux changer (1) :
beh_par.EVALUATE_IN_ROI = false;
%     % when this flag is true, the numerical evaluation is carried only using the ROI
%     %   -> elimination of the problem of padding with zeros
%     %   -> evaluation in the region of interest (resulting DVF more accurate in the ROI)

% Directories 
path_par = load_path_parameters();

% Input image sequences
path_par.input_im_dir_suffix_tab = [
    %string('2. sq sl010 sag Xcs=125');    
    %string('3. sq sl010 sag Xcs=80');   
    %string('4. sq sl014 sag Xcs=165');  
    %string('5. sq sl014 sag Xcs=95')
    %string('6. sq sl010 sag Xcs=110')
    %string('7. sq sl010 sag Xcs=100')
    %string('8. sq sl010 sag Xcs=140')
    %string('9. sq sl014 sag Xcs=110')
    %string('10. sq sl014 sag Xcs=125')
    string('11. sq sl014 sag Xcs=150')
    ];

% OF evaluation - je peux �galement changer ces param�tres :
OFeval_par = load_OFeval_parameters();

% Parameters for image warping
%warp_par = load_3Dwarp_par();    

% Behavior - � ne pas changer
beh_par.DISPLAY_OF = false;

% variables r�sultat finales
nb_seq = length(path_par.input_im_dir_suffix_tab);
length_sigma_LK_tab = length(OFeval_par.sigma_LK_tab);
length_sigma_init_tab = length(OFeval_par.sigma_init_tab);
length_sigma_subspl_tab = length(OFeval_par.sigma_subspl_tab);
nb_layers_test = OFeval_par.nb_layers_max - OFeval_par.nb_layers_min +1;
nb_iter_test = OFeval_par.nb_max_iter - OFeval_par.nb_min_iter +1;
% final array containing the desired rms scores :
rms_error_all_seq = zeros(nb_layers_test, length_sigma_LK_tab, nb_iter_test, length_sigma_init_tab, length_sigma_subspl_tab, nb_seq, 'single');
best_par_all_seq = cell(nb_seq, 1);

for im_seq_idx = 1:nb_seq
        
    % directory of the input images (text string inside the function)
    path_par.input_im_dir_suffix = path_par.input_im_dir_suffix_tab(im_seq_idx);
    path_par.input_im_dir = sprintf('%s\\%s', path_par.input_im_dir_pref, path_par.input_im_dir_suffix);

    % Image parameters
    im_par = load_im_param(path_par);
    
    im_par.nb_im = 90; % temporaire, juste pour tester

    %% PROCEDURE DE TEST (PROGRAMME PRINCIPAL EN GROS)
    
    compute_save_OF_mult_param( OFeval_par, path_par, im_par);

    [rms_error, best_par] = rms_of2D( beh_par, OFeval_par, path_par, im_par);
    path_par.OFeval_log_file_entire_fname = sprintf('%s\\%s %s', path_par.txt_file_dir, char(path_par.input_im_dir_suffix), path_par.OFoptim_log_filename);
    write_rms_eval_log_file( beh_par, OFeval_par, path_par, im_par, rms_error, best_par );

    rms_error_all_seq(:,:,:,:,:,im_seq_idx) = rms_error;
    best_par_all_seq{im_seq_idx} = best_par;

end

analyze_OF_param_influence( rms_error_all_seq, OFeval_par, beh_par, path_par);

% saving the workspace
save(sprintf('%s\\%s', path_par.temp_var_dir, path_par.OFoptim_workspace_filename));
