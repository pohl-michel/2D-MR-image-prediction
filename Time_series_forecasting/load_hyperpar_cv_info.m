function [ hppars ] = load_hyperpar_cv_info( pred_par )
% load information about the hyperparameters for performing cross correlation
% hppars = hyper-parameters
% Important: the last hyper-parameter index must not correspond to a singleton (permutation is necessary in that case).
%
% Author : Pohl Michel
% Date : September 11th, 2021
% Version : v1.1
% License : 3-clause BSD License 							

    switch(pred_par.pred_meth)
        case 'multivariate linear regression'
            
            hppars.nb_runs_cv = 1;            
            hppars.nb_runs_eval_test = 1;
            
            hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7];  
            %hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];
            %hppars.horizon_tab = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63];
            
            hppars.other(1).name = 'SHL';
            hppars.other(1).val = [6, 12, 18, 24, 30];  
            %hppars.other(1).val = [4, 8, 12, 16, 20];  
            %hppars.other(1).val = [12, 24, 36, 48, 60];  
            %hppars.other(1).val = [36, 72, 108, 144, 180]; 
            %hppars.other(1).val = [10, 20, 30, 40, 50, 60, 70, 80, 90];             
            %hppars.other(1).val = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, ...
            %                           31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50];
            
        case 'RTRL'
          
            hppars.nb_runs_cv = 10;
            hppars.nb_runs_eval_test = 10;
            
            hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7];

            hppars.other(1).name = 'SHL';
            %hppars.other(1).val = [10, 25, 40, 55];     
            hppars.other(1).val = [6, 12, 18, 24, 30];             
            %hppars.other(1).val = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50];   
            
%             hppars.other(2).name = 'grad_threshold';
%             hppars.other(2).val = [2.0];
%             
%             hppars.other(3).name = 'Winit_std_dev';
%             hppars.other(3).val = [0.01, 0.02, 0.05];
%             
%             hppars.other(4).name = 'learn_rate';
%             hppars.other(4).val = [0.02, 0.05, 0.1, 0.2];

            hppars.other(2).name = 'learn_rate';
            %hppars.other(2).val = [0.02, 0.05, 0.1, 0.2];
            hppars.other(2).val = [0.005, 0.01, 0.02];            

            hppars.other(3).name = 'rnn_state_space_dim';
            %hppars.other(3).val = [10, 20, 30, 40, 50, 60, 70, 80, 90];
            hppars.other(3).val = [10, 30, 50, 70, 90, 110];            


        case 'RTRL v2'

            hppars.nb_runs_cv = 50;
            hppars.nb_runs_eval_test = 300;    

            hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];      
            %hppars.horizon_tab = [1, 2];      

            hppars.other(1).name = 'SHL';
            hppars.other(1).val = [12, 24, 36, 48, 60];
            %hppars.other(1).val = [12, 24];

            hppars.other(2).name = 'learn_rate';
            hppars.other(2).val = [0.005, 0.01, 0.02];            
            %hppars.other(2).val = [0.005, 0.01]; 

            hppars.other(3).name = 'rnn_state_space_dim'; 
            hppars.other(3).val = [10, 25, 40];   
            %hppars.other(4).val = [30, 60]; 

        case 'no prediction'

            hppars.nb_runs_cv = 1;
            hppars.nb_runs_eval_test = 1;            
            
            %hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];      
            %hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7];  
            hppars.horizon_tab = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63];
            %hppars.horizon_tab = 21;
            
            hppars.other(1).name = 'SHL';
            hppars.other(1).val = [1]; % The lastest acquired value is used as the predicted value        
            
        case 'LMS'
            
            hppars.nb_runs_cv = 1;
            hppars.nb_runs_eval_test = 1;            
               
            %hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];
            %hppars.horizon_tab = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63];
            hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7];
            
            hppars.other(1).name = 'SHL';           
            hppars.other(1).val = [6, 12, 18, 24, 30]; 
            %hppars.other(1).val = [4, 8, 12, 16, 20];  
            %hppars.other(1).val = [12, 24, 36, 48, 60];      
            %hppars.other(1).val = [36, 72, 108, 144, 180];  
            
            hppars.other(2).name = 'learn_rate';
            %hppars.other(2).val = [0.00005, 0.0001, 0.0002];
            %hppars.other(2).val = [0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01];
            hppars.other(2).val = [0.02, 0.05, 0.1, 0.2];
            
        case 'UORO'

%             %hppars.nb_runs_cv = 10;
%             %hppars.nb_runs_eval_test = 50;
%             hppars.nb_runs_cv = 250;
%             hppars.nb_runs_eval_test = 250;
%             
%             hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7];
% 
%             hppars.other(1).name = 'SHL';
%             %hppars.other(1).val = [4, 8, 12, 16, 20];   
%             hppars.other(1).val = [6, 12, 18, 24, 30];  
%             
%             hppars.other(2).name = 'learn_rate';
%             hppars.other(2).val = [0.005, 0.01, 0.02];
%             
%             hppars.other(3).name = 'rnn_state_space_dim'; 
%             %hppars.other(3).val = [30, 60, 90, 120, 150, 180];  
%             hppars.other(3).val = [10, 30, 50, 70, 90, 110]; 
            

            % test
            hppars.nb_runs_cv = 50;
            hppars.nb_runs_eval_test = 300;
            
            hppars.horizon_tab = [1, 5];

            hppars.other(1).name = 'SHL';
            hppars.other(1).val = [12, 24]; 
            
            hppars.other(2).name = 'learn_rate';
            hppars.other(2).val = [0.005, 0.01];
            
            hppars.other(3).name = 'rnn_state_space_dim'; 
            hppars.other(3).val = [30, 60];    
            
            
            
        case 'SnAp-1'

            %hppars.nb_runs_cv = 10;
            %hppars.nb_runs_eval_test = 50;
            hppars.nb_runs_cv = 250;
            hppars.nb_runs_eval_test = 250;            
            
            hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7];

            hppars.other(1).name = 'SHL';
            hppars.other(1).val = [6, 12, 18, 24, 30];  
            %hppars.other(1).val = [4, 8, 12, 16, 20];   
            
            hppars.other(2).name = 'learn_rate';
            hppars.other(2).val = [0.005, 0.01, 0.02];
            
            hppars.other(3).name = 'rnn_state_space_dim'; 
            %hppars.other(3).val = [30, 60, 90, 120, 150, 180];    
            hppars.other(3).val = [10, 30, 50, 70, 90, 110];             


        case 'DNI'

            hppars.nb_runs_cv = 50;
            hppars.nb_runs_eval_test = 300;    

            hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];      
            %hppars.horizon_tab = [1, 2];      

            hppars.other(1).name = 'SHL';
            hppars.other(1).val = [12, 24, 36, 48, 60];
            %hppars.other(1).val = [12, 24];

            hppars.other(2).name = 'learn_rate';
            hppars.other(2).val = [0.005, 0.01, 0.02];            
            %hppars.other(2).val = [0.005, 0.01]; 

            hppars.other(3).name = 'learn_rate_A';
            %hppars.other(3).val = [0.002, 0.005, 0.01];            
            hppars.other(3).val = [0.002];  

            hppars.other(4).name = 'rnn_state_space_dim'; 
            hppars.other(4).val = [30, 60, 90, 120, 150, 180];   
            %hppars.other(4).val = [30, 60]; 

        case 'fixed W'

            hppars.nb_runs_cv = 50;
            hppars.nb_runs_eval_test = 300;    

            hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];         

            hppars.other(1).name = 'SHL';
            hppars.other(1).val = [12, 24, 36, 48, 60];

            hppars.other(2).name = 'learn_rate';
            hppars.other(2).val = [0.005, 0.01, 0.02];             

            hppars.other(3).name = 'rnn_state_space_dim'; 
            hppars.other(3).val = [30, 60, 90, 120, 150, 180];
            

    end

    hppars.nb_additional_params = numel(hppars.other);
    hppars.nb_hrz_val = length(hppars.horizon_tab); % number of horizon values tested
    for hppar_idx = 1:hppars.nb_additional_params
        hppars.other(hppar_idx).nb_val = length(hppars.other(hppar_idx).val);
    end
    
    % variable setup for time performance analysis  
    for hppar_idx = 1:hppars.nb_additional_params
        if strcmp(hppars.other(hppar_idx).name, 'SHL')
            hppars.SHL_hyppar_idx = hppar_idx;
        end
        if strcmp(hppars.other(hppar_idx).name, 'rnn_state_space_dim')
            hppars.state_space_hyppar_idx = hppar_idx;
        end        
    end

    %nb_calc_temp = hppars.nb_hrz_val;
    nb_calc_temp = 1;
    for hppar_idx = 1:hppars.nb_additional_params
        nb_calc_temp = nb_calc_temp*hppars.other(hppar_idx).nb_val;
    end
    hppars.nb_calc = nb_calc_temp;
    
end

