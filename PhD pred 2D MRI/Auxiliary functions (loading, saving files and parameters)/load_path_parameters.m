function [ path_par ] = load_path_parameters()
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

    % time
    path_par.date_and_time = sprintf('%s %s', datestr(datetime, 'yyyy - mm - dd HH AM MM'), 'min');

    % directory of the imput images (argument of the function "find_2D_im_dir" - "pref" means "prefix")
    path_par.input_im_dir_pref = 'Input images\2D images';
    % directory for saving temporary fig files (located in the Matlab workspace)
    path_par.temp_fig_dir = 'Temporarily saved figures';
    % directory for saving temporary images (located in the Matlab workspace)
    path_par.temp_im_dir = 'Temporarily saved images';
    % directory for saving temporary variables
    path_par.temp_var_dir = 'Temporarily saved variables';
    %path_par.temp_var_dir = 'Temporarily saved variables (laptop)';
    % directory for saving log files
    path_par.txt_file_dir = 'Temporarily saved txt files';

    % check if the directories above exist and create them if they do not
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
    
    % optical flow parameters filename
    path_par.OFpar_filename = 'OF_calc_par.xlsx';
    % image parameters filename
    path_par.im_par_filename = 'im_seq_par.xlsx';
    % display parameters filename
    path_par.disp_par_filename = 'disp_par.xlsx';
    % prediction parameters filename
    path_par.pred_par_filename_suffix = 'pred_par.xlsx';
    % txt file with parameters filename
    path_par.log_txt_filename = sprintf('log file %s.txt', path_par.date_and_time);
   
    % prediction parameter optimization log file with parameters filename
    path_par.pred_par_optim_log_filename = sprintf('pred par optim log file %s.txt', path_par.date_and_time);    
    % Optical flow parameter optimization log file with parameters filename
    path_par.OFoptim_log_filename = sprintf('LK_OFoptim log file %s.txt', path_par.date_and_time);
    % Optical flow parameter optimization (param influence) log file with parameters filename
    path_par.OFparam_influence_log_filename = sprintf('LK_OFparam_influence log file %s.txt', path_par.date_and_time);    
        % j'avais oublié que ce fichier existait ^^'
    % Optical flow parameter optimization workspace filename
    path_par.OFoptim_workspace_filename = sprintf('LK_OFoptim workspace %s.mat', path_par.date_and_time);

    
end
