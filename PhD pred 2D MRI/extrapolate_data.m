clear all
close all
clc

addpath(genpath('ReadData3D_version1k'))
addpath(genpath('rician'))
addpath(genpath('Auxiliary functions (loading, saving files and parameters)'))
addpath(genpath('0. 2D im seq extraction and extrapol'))

%% PARAMETERS 

    % Rq : il va peut-être falloir écrire une fonction d'interpolation au
    % début, afin que je n'aie pas à interpoler moi-même

% Program behavior
beh_par.RESAMPLE_3DIM = true; % so that each voxel corresponds to 1mm^3
beh_par.CROP_3DIM = true;

% Directories 
%path_par = load_path_parameters3D();
path_par.input_im_dir_pref = 'Input images\\Original 3D images';
path_par.output_im_dir_pref = 'Input images\\2D images';
path_par.im_par_filename = 'im_seq_par.xlsx';
path_par.im_seq_par_txt_filename = 'Sequence parameters.txt';
path_par.disp_par_filename = 'disp_par.xlsx';

% Input image sequences
input_im_seq_tab = [
    %string('1. sq sl010 resampled');    
    string('2. sq sl014 resampled');
    ];

% Corresponding output image sequences 
output_im_seq_tab = [
    string('11. sq sl014 sag Xcs=150')   %"sag" like "saggital slices" 
    ];  

extrl_par.nb_im = [200]; % number of images in the extrapolated sequence
%extrl_par.rician_s = [15]; % noise level (NB actual Rician stdev depends on signal, see ricestat)

extrl_par.t_init = [2];
    % Initial time when we start creating the sequence
    % image au milieu par laquelle on va commencer à cropper (pour avoir un flot optique correct)
    
    % t_init = 4 pour 1.sq sl010
    % t_init = 2 pour 1.sq sl014
    
    
nb_seq = length(input_im_seq_tab);
for im_seq_idx = 1:nb_seq
    
    % directory of the input images (text string inside the function)
    path_par.input_im_seq = input_im_seq_tab(im_seq_idx);
    path_par.input_im_dir = sprintf('%s\\%s', path_par.input_im_dir_pref, path_par.input_im_seq);

    % directory of the ROI images (text string inside the function)
    path_par.output_im_seq = output_im_seq_tab(im_seq_idx);
    path_par.output_im_dir = sprintf('%s\\%s', path_par.output_im_dir_pref, path_par.output_im_seq);
    
    path_par.temp_im_dir = sprintf('%s\\jpg images', path_par.output_im_dir);
        % for saving jpg images with save_crop_enhance_2Dim_jpg
    
    % Image parameters
    org_im_par = load_im_param(path_par);
    
    % Display parameters
    disp_par = load_display_parameters(path_par);

    % Display parameters
    beh_par.CROP_FOR_DISP_SAVE = false; % ne pas changer
    

%% ---------------------------------------------------------------------------------------------------------------------------------------------------
%  PROGRAM  - EXTRAPOLATION OF 4D-CBCT DATA   --------------------------------------------------------------------------------------------------------
%  --------------------------------------------------------------------------------------------------------------------------------------------------- 

    % Because we are using saggital slices 
    new_im_par.x_m = org_im_par.y_m;
    new_im_par.x_M = org_im_par.y_M;

    new_im_par.y_m = org_im_par.z_m;
    new_im_par.y_M = org_im_par.z_M;    
    
    new_im_par.L = org_im_par.W;
    new_im_par.W = org_im_par.H;

    new_im_par.x_m = 1;
    new_im_par.x_M = org_im_par.y_M - org_im_par.y_m + 1;
    new_im_par.y_m = 1;
    new_im_par.y_M = org_im_par.z_M - org_im_par.z_m + 1;
    
    new_im_par.nb_im = extrl_par.nb_im(im_seq_idx);
    new_im_par.imtype = 'dcm';
    
%     if beh_par.RESAMPLE_3DIM
%         
%         path_par.resampled_input_im_seq = sprintf('%s resampled', path_par.input_im_seq);
%         path_par.resampled_input_im_dir = sprintf('%s\\%s', path_par.input_im_dir_pref, path_par.resampled_input_im_seq);
%         
%         % à compléter (changer un peu la fonction)
%         create_im_seq_directory(path_par, path_par.resampled_input_im_dir, org_im_par, resampled_im_par, extrl_par, im_seq_idx);
%         resample3D_im(); % idem à écrire
%         
%     end
    
    % Creating the folder and the file parameters related to the extended cross-sections sequence
    create_im_seq_directory(path_par, path_par.output_im_dir, org_im_par, new_im_par, extrl_par, im_seq_idx);
    extrapolate_and_save2Dimseq(disp_par, extrl_par, org_im_par, path_par, im_seq_idx);
    
    
end
    
addpath(genpath('1.1. Functions master project'))    
addpath(genpath('1.2. Functions PhD project'))    