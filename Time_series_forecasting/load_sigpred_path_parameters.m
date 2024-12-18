function [ path_par ] = load_sigpred_path_parameters()
% LOAD_SIGPRED_PATH_PARAMETERS Initializes and returns the structure `path_par`, which contains file paths and folder information
% used in the signal prediction algorithm pipeline. This function creates necessary directories if they do not already exist and 
% organizes paths for time series data, result figures, and other intermediate files.
%
% INPUTS:
%   (None)
%
% OUTPUTS:
% - path_par (struct): Structure containing paths and file/folder names used within the signal prediction pipeline. 
%
% REMARK:
% - to perform prediction with a new time series, just add the corresponding folder to "path_par.time_series_dir_tab"
%
% Author : Pohl Michel
% License : 3-clause BSD License

    % date and time
    path_par.date_and_time = sprintf('%s %s', datestr(datetime, 'yyyy - mm - dd HH AM MM'), 'min');

    % directory containing the input sequences
    path_par.parent_seq_dir = 'a. Input time series sequences';    
    % directory for saving fig files
    path_par.temp_fig_dir = 'b. Prediction results (figures)';
    % directory for saving images files
    path_par.temp_im_dir = 'c. Prediction results (images)';
    % directory for saving auxiliary RNN variables
    path_par.temp_var_dir = 'd. RNN variables (temp)';
    % directory for saving log files
    path_par.txt_file_dir = 'e. Log txt files';

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
    
    % time series data directories
    path_par.time_series_dir_tab = [ % You can add additional directories here if needed
        
    % Original external marker positions sampled at 10 Hz
        % string('Ext markers seq 1');
        % string('Ext markers seq 2');
        % string('Ext markers seq 3');
        % string('Ext markers seq 4');
        % string('Ext markers seq 5');
        % string('Ext markers seq 6');
        % string('Ext markers seq 7');
        % string('Ext markers seq 8');   
        % string('Ext markers seq 9');  
        
    % External marker positions resampled at 10 Hz        
        % string('Ext markers seq 1 30 Hz');
        % string('Ext markers seq 2 30 Hz');
        % string('Ext markers seq 3 30 Hz');
        % string('Ext markers seq 4 30 Hz');
        % string('Ext markers seq 5 30 Hz');
        % string('Ext markers seq 6 30 Hz');
        % string('Ext markers seq 7 30 Hz');
        % string('Ext markers seq 8 30 Hz');   
        % string('Ext markers seq 9 30 Hz'); 

    % External marker positions resampled at 3.33 Hz         
        string('Ext markers seq 1  3.33 Hz'); 
        string('Ext markers seq 2  3.33 Hz');
        string('Ext markers seq 3  3.33 Hz'); 
        string('Ext markers seq 4  3.33 Hz'); 
        string('Ext markers seq 5  3.33 Hz'); 
        string('Ext markers seq 6  3.33 Hz'); 
        string('Ext markers seq 7  3.33 Hz'); 
        string('Ext markers seq 8  3.33 Hz'); 
        string('Ext markers seq 9  3.33 Hz');      
        
    % Normalized PCA weights associated with the time-varying deformation fields in the cine-MR sequence
        % string('2. sq sl010 sag Xcs=125 3 cpts 2024 normalised');
        % string('3. sq sl010 sag Xcs=80 3 cpts 2024 normalised');
        % string('4. sq sl014 sag Xcs=165 3 cpts 2024 normalised');
        % string('5. sq sl014 sag Xcs=95 3 cpts 2024 normalised');        
        ];
    
    % time series data filenames
    path_par.time_series_data_filename_suffix = 'data.mat';
    % display parameters filename
    path_par.disp_par_filename = 'disp_par.xlsx';
    % prediction parameters filename
    path_par.pred_par_filename_suffix = 'pred_par.xlsx';
    % txt file with parameters and results
    path_par.log_txt_filename = sprintf('log file %s.txt', path_par.date_and_time);
    % directory containing the results of hyperparameter optimization
    path_par.hyperpar_optim_log_filename = sprintf('optim log file %s.txt', path_par.date_and_time);
    
end