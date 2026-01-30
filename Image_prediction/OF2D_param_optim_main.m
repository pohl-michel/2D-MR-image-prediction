% This script uses grid search to identify the optimal parameters for calculating deformation vector fields (DVF) based on the Lucas-Kanade algorithm.
% It loads image sequences, performs evaluations over parameter sets, and saves results for further analysis.
% The best parameters for each sequence are recorded in Excel files and the workspace is saved at the end.
% 
% Author : Pohl Michel
% License : 3-clause BSD License

clear all
close all
clc

% Move to the 'Image_prediction' folder if it exists in the current directory
if any(strcmp({dir(pwd).name},'Image_prediction'))
    cd Image_prediction
end

%% PARAMETERS

% Set random number generator seed for reproducibility (Mersenne Twister algorithm)
rng(0, "twister");

% Program behavior parameters
beh_par.EVALUATE_IN_ROI = false;
% When set to true, evaluation is restricted to the region of interest (ROI), improving the DVF accuracy within the latter
% Restricting the evaluation to the DVF also eliminates zero-padding effects

% manually define nb of images for optimization
nb_optim_imgs = 2;
    % ETH Zürich: optimization using the first 90 images of the sequence (training set)
    % Magdeburg University: 170 images (training set)
    % software testing: 2 images

% Directories and paths
path_par = load_impred_path_parameters();

% Input image sequences (modify or add sequences as needed)
path_par.input_im_dir_suffix_tab = [
    %string('2. sq sl010 sag Xcs=125');    
    %string('3. sq sl010 sag Xcs=80');   
    %string('4. sq sl014 sag Xcs=165');  
    %string('5. sq sl014 sag Xcs=95')
    string('2020-11-10_KS81_Nav_Pur_1');
    % string('2020-11-12_QN76_Nav_Pur_1');   
    % string('2020-11-17_CS31_Nav_Pur_2');  
    % string('2020-11-17_JY02_Nav_Pur_2');
    % string('2020-11-23_ON65_Nav_Pur_2');
    % string('2020-11-23_PS11_Nav_Pur_1');   
    % string('2020-11-25_II29_Nav_Pur_1');  
    % string('2020-11-26_NE38_Nav_Pur_1');      
    %string('Add the name of your sequence here')
    ];

% Hyper-parameter grid search settings:
OFeval_par = load_OFeval_parameters();    

% Behavior (display flag) - do not change
beh_par.DISPLAY_OF = false;

% Initialization of output variables
nb_seq = length(path_par.input_im_dir_suffix_tab);
length_sigma_LK_tab = length(OFeval_par.sigma_LK_tab);
length_sigma_init_tab = length(OFeval_par.sigma_init_tab);
length_sigma_subspl_tab = length(OFeval_par.sigma_subspl_tab);
nb_layers_test = OFeval_par.nb_layers_max - OFeval_par.nb_layers_min +1;
nb_iter_test = OFeval_par.nb_max_iter - OFeval_par.nb_min_iter +1;

% final arrays containing the RMS errors for each parameter combination and best parameters for each sequence :
rms_error_all_seq = zeros(nb_layers_test, length_sigma_LK_tab, nb_iter_test, length_sigma_init_tab, length_sigma_subspl_tab, nb_seq);
best_par_all_seq = cell(nb_seq, 1);

% MAIN LOOP: Process Each Image Sequence
for im_seq_idx = 1:nb_seq
        
    % Set up input image directory for current sequence
    path_par.input_im_dir_suffix = path_par.input_im_dir_suffix_tab(im_seq_idx);
    path_par.input_im_dir = sprintf('%s\\%s', path_par.input_im_dir_pref, path_par.input_im_dir_suffix);

    % Load image parameters and define number of images for optimization
    im_par = load_im_param(path_par);
    im_par.nb_im = nb_optim_imgs; 

    %% EVALUATION FOR EACH HYPER-PARAMETER SET IN THE GRID

    % Compute the optical flow for each parameter in the grid
    compute_save_OF_mult_param(OFeval_par, path_par, im_par);

    % Compute RMS error and find best parameters for this sequence
    [rms_error, best_par] = rms_of2D( beh_par, OFeval_par, path_par, im_par);
    path_par.OFeval_log_file_entire_fname = sprintf('%s\\%s %s', path_par.txt_file_dir, char(path_par.input_im_dir_suffix), path_par.OFoptim_log_filename);
    write_OFhyperpar_optim_log_file( beh_par, OFeval_par, path_par, im_par, rms_error, best_par );

    % Store results for current sequence
    rms_error_all_seq(:,:,:,:,:,im_seq_idx) = rms_error;
    best_par_all_seq{im_seq_idx} = best_par;

    %% Re-compute optical flow with optimal parameters on entire sequence
    im_par = load_im_param(path_par); % Reload original image parameters (with correct number of images) 
    fn = fieldnames(best_par);
    for k=1:numel(fn)
        OF_par.(fn{k}) = best_par.(fn{k});
    end
    OF_par.epsilon_detG = OFeval_par.epsilon_detG;
    OF_par.grad_meth = OFeval_par.grad_meth;
    OF_par.grad_meth_str = OFeval_par.grad_meth_str;
    compute_2Dof(OF_par, im_par, path_par);

    % Save best parameters in an Excel file if not already present
    OF_bestpar_xlsx = rmfield(best_par , 'rms_error');
    OF_bestpar_xlsx.grad_meth = OFeval_par.grad_meth;
    OF_bestpar_xlsx.epsilon_detG = OFeval_par.epsilon_detG;
    date = sprintf('%s_%s', datestr(datetime, 'yyyy_mm_dd_HH_AM_MM'), 'min');
    OF_calc_param_file = sprintf('%s\\%s_optimized_%s', path_par.input_im_dir, date, path_par.OFpar_filename);
    if not(isfile(OF_calc_param_file))
        writetable(struct2table(OF_bestpar_xlsx), OF_calc_param_file);
    end

end

% Analyze the influence of the parameters on RMS Error
analyze_OF_param_influence(rms_error_all_seq, OFeval_par, beh_par, path_par, nb_optim_imgs);

% Save the workspace data for future use
save(sprintf('%s\\%s', path_par.temp_var_dir, path_par.OFoptim_workspace_filename));
