% This script optimize the parameters involved in the calculation of the deformation vector field (DVF) of 3D dicom image sequences 
% using the pyramidal Lucas Kanade optical flow algorithm.
% 
% Author : Pohl Michel
% Date : Sept. 19th, 2022
% Version : v1.1
% License : 3-clause BSD License

clear all
close all
clc

% current directory
pwd_split_cell = strsplit(pwd, '\');
pwdir = string(pwd_split_cell(end));

if pwdir == "Future_frame_prediction"
    cd Image_prediction
end

%% PARAMETERS

% Program behavior (can be changed)
beh_par.EVALUATE_IN_ROI = false;
    % when this flag is true, the numerical evaluation is carried only using the ROI
    %   -> elimination of the problem of padding with zeros
    %   -> evaluation in the region of interest (resulting DVF more accurate in the ROI)

% Directories 
path_par = load_impred_path_parameters();

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
    %string('11. sq sl014 sag Xcs=150')
    string('2020-11-10_KS81_Nav_Pur_1')
    %string('2020-11-12_QN76_Nav_Pur_1')
    %string('2020-11-17_CS31_Nav_Pur_2')
    %string('2020-11-17_JY02_Nav_Pur_2')
    %string('2020-11-23_ON65_Nav_Pur_2')
    %string('2020-11-23_PS11_Nav_Pur_1')
    %string('2020-11-25_II29_Nav_Pur_1')
    %string('2020-11-26_NE38_Nav_Pur_1')
    ];

% Hyper-parameter grid search :
OFeval_par = load_OFeval_parameters();    

% Behavior - do not change
beh_par.DISPLAY_OF = false;

% final results variables
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
    im_par.nb_im = 170; 
        % ETH Zürich: optimization using the first 90 images of the sequence (arbitrary)
        % Magdeburg University: 170 images

    %% EVALUATION FOR EACH HYPER-PARAMETER SET IN THE GRID
    
    compute_save_OF_mult_param( OFeval_par, path_par, im_par);

    [rms_error, best_par] = rms_of2D( beh_par, OFeval_par, path_par, im_par);
    path_par.OFeval_log_file_entire_fname = sprintf('%s\\%s %s', path_par.txt_file_dir, char(path_par.input_im_dir_suffix), path_par.OFoptim_log_filename);
    write_OFhyperpar_optim_log_file( beh_par, OFeval_par, path_par, im_par, rms_error, best_par );

    rms_error_all_seq(:,:,:,:,:,im_seq_idx) = rms_error;
    best_par_all_seq{im_seq_idx} = best_par;

    % We re-compute the optical flow with the best parameters on the whole sequence:
    im_par = load_im_param(path_par); % We reload the image parameters to recover the "original" number of images 
    fn = fieldnames(best_par);
    for k=1:numel(fn)
        OF_par.(fn{k}) = best_par.(fn{k});
    end
    OF_par.epsilon_detG = OFeval_par.epsilon_detG;
    OF_par.grad_meth_str = OFeval_par.grad_meth_str;
    compute_2Dof(OF_par, im_par, path_par)

end

analyze_OF_param_influence( rms_error_all_seq, OFeval_par, beh_par, path_par);

% saving the workspace
save(sprintf('%s\\%s', path_par.temp_var_dir, path_par.OFoptim_workspace_filename));
