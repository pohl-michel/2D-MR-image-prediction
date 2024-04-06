function [ path_par ] = load_impred_path_parameters()
% This function returns path_par, which contains path and filename information for loading and saving files.
% 
% Author : Pohl Michel
% Date : Sept 18th, 2022
% Version : v1.1
% License : 3-clause BSD License

    % current time
    path_par.date_and_time = sprintf('%s %s', datestr(datetime, 'yyyy - mm - dd HH AM MM'), 'min');

    % directory of the input images ("pref" means "prefix")
    path_par.input_im_dir_pref = 'input_imgs\2D images';
    % directory for saving temporary fig files
    path_par.temp_fig_dir = 'tmp_figs';
    % directory for saving temporary images
    path_par.temp_im_dir = 'tmp_imgs';
    % directory for saving temporary variables
    path_par.temp_var_dir = 'tmp_vars';
    % directory for saving log files
    path_par.txt_file_dir = 'tmp_txt_files';

    % check if the directories listed above exist and create them if they do not
    if ~exist(path_par.temp_fig_dir, 'dir')
       mkdir(path_par.temp_fig_dir)
    end
    if ~exist(path_par.temp_im_dir, 'dir')
       mkdir(path_par.temp_im_dir)
    end    
    if ~exist(path_par.temp_var_dir, 'dir')
       mkdir(path_par.temp_var_dir)
    end  
    if ~exist(path_par.txt_file_dir, 'dir')
       mkdir(path_par.txt_file_dir)
    end        

    % image prediction folder ("Image_prediction" - I do not specify the name to make the code more robust to changes)
    pwd_split_cell = strsplit(pwd, '\');
    path_par.im_pred_dir = string(pwd_split_cell(end));
    
    % File containing the optical flow parameters
    path_par.OFpar_filename = 'OF_calc_par.xlsx';
    % File containing the image parameters
    path_par.im_par_filename = 'im_seq_par.xlsx';
    % File containing the display parameters
    path_par.disp_par_filename = 'disp_par.xlsx';
    % File containing the prediction parameters
    path_par.pred_par_filename_suffix = 'pred_par.xlsx';
    % txt file name suffix containing the image prediction results
    path_par.log_txt_filename = sprintf('log file %s.txt', path_par.date_and_time);
      
    % txt file containing the best hyperparameters that optimize the DVF accuracy in the grid search
    path_par.OFoptim_log_filename = sprintf('LK_OFoptim log file %s.txt', path_par.date_and_time);
    % txt file containing info about the influence of each hyper-parameter on the DVF accuracy
    path_par.OFparam_influence_log_filename = sprintf('LK_OFparam_influence log file %s.txt', path_par.date_and_time);    
    % Optical flow parameter optimization workspace filename
    path_par.OFoptim_workspace_filename = sprintf('LK_OFoptim workspace %s.mat', path_par.date_and_time);

    
end
