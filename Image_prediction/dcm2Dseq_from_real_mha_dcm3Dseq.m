% Loads 3D image sequences in the folder "Original 3D images" and return a 2D image sequence corresponding to a particular saggital cross-section Xcs.
% The image sequence is shifted along the time axis so that subsequent deformable image registration starts from an image 
% where the organs are in the middle of the breathing cycle (see below transformation_par.t_init).
% These 2D image sequences are then used subsequently for video prediction. 
% 
% Author : Pohl Michel
% Date : September 26th, 2022
% Version : v1.0
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

addpath(genpath('0. 2D image sequence extraction'))
addpath(genpath('6. Auxiliary functions (loading, saving files and parameters)'))


%% PARAMETERS (CAN BE CHANGED MANUALLY)

% Input image sequences
input_im_seq_tab = [
    string('1. sq sl010 resampled'); % image resampled using Slicer so that one voxel corresponds to 1mm^3
    string('1. sq sl010 resampled'); 
    string('2. sq sl014 resampled'); % image resampled using Slicer so that one voxel corresponds to 1mm^3
    string('2. sq sl014 resampled');
    ];

% Corresponding output image sequences 
output_im_seq_tab = [
    string('6. sq sl010 sag Xcs=125'); % Rk: "sag" here means "saggital slices"
    string('7. sq sl010 sag Xcs=80');
    string('8. sq sl014 sag Xcs=165');
    string('9. sq sl014 sag Xcs=95');
    ];  

transformation_par.Xcs = [125, 80, 165, 95]; 
    % Coordinate of the 2D slice in the original 3D images 

transformation_par.t_init = [4, 4, 2, 2];
    % Initial time when we start creating the sequence
    % image au milieu par laquelle on va commencer à cropper (pour avoir un flot optique correct)
    
    % t_init = 4 pour 1.sq sl010
    % t_init = 2 pour 1.sq sl014


%% OTHER PARAMETERS

% Directories 
path_par.input_im_dir_pref = 'a. Input images\\Original 3D images';
path_par.output_im_dir_pref = 'a. Input images\\2D images';
path_par.im_par_filename = 'im_seq_par.xlsx';
path_par.im_seq_par_txt_filename = 'Sequence parameters.txt';
path_par.disp_par_filename = 'disp_par.xlsx';    
    
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
    disp_par = load_impred_display_parameters(path_par);
    

%%  MAIN PROGRAM - PROCESSING 4D DATA 

    % Because we are using saggital slices 
    new_im_par.x_m = org_im_par.y_m;
    new_im_par.x_M = org_im_par.y_M;

    new_im_par.y_m = org_im_par.z_m;
    new_im_par.y_M = org_im_par.z_M;    
    
    new_im_par.L = org_im_par.W;
    new_im_par.W = org_im_par.H;
    
    new_im_par.nb_im = org_im_par.nb_im;
    new_im_par.imtype = 'dcm';
    
    % Creating the folder and the file parameters related to the extended cross-sections sequence
    create_2Dim_directory(path_par, path_par.output_im_dir, org_im_par, new_im_par, transformation_par, im_seq_idx);
    save_2D_slices(disp_par, transformation_par, org_im_par, path_par, im_seq_idx);
    
end

