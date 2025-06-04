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
            
            % Common parameters (always true)
            hppars.nb_runs_cv = 1;            
            hppars.nb_runs_eval_test = 1;
            
            % % Prediction of the position of external markers (CPMB paper)
            %
            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7];                                                               % 3.33 Hz sampling
            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];         % 10 Hz sampling
            % hppars.horizon_tab = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63];   % 30 Hz sampling
            %
            % hppars.other(1).name = 'SHL';
            % hppars.other(1).val = [4, 8, 12, 16, 20];       % 3.33 Hz sampling
            % hppars.other(1).val = [12, 24, 36, 48, 60];     % 10 Hz sampling    
            % hppars.other(1).val = [36, 72, 108, 144, 180];  % 30 Hz sampling

            % Next-frame MR image prediction (CMIG paper)
            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7];  % ETH Zurich
            hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];  % Magdeburg dataset
            
            hppars.other(1).name = 'SHL';
            % hppars.other(1).val = [6, 12, 18, 24, 30]; % ETH Zurich 
            hppars.other(1).val = [11, 23, 34, 45, 57]; % Magdeburg
            
        case 'RTRL' % Not used anymore - that was the method in my first CPMB and CMIG papers published in 2021 and 2022
          
            % The parameters below can correspond to the prediction of the position of markers at 10 Hz (to provide an example).
            hppars.nb_runs_cv = 10;
            hppars.nb_runs_eval_test = 10;
            
            hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7];

            hppars.other(1).name = 'SHL';  
            hppars.other(1).val = [12, 24, 36, 48, 60];         

            hppars.other(2).name = 'learn_rate';
            hppars.other(2).val = [0.005, 0.01, 0.02];            

            hppars.other(3).name = 'rnn_state_space_dim';
            hppars.other(3).val = [10, 25, 40];           


        case 'RTRL v2'

            % Prediction of the position of external markers (CPMB paper)
            % 
            % hppars.nb_runs_cv = 50;
            % hppars.nb_runs_eval_test = 300;
            % 
            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7];                                                               % 3.33 Hz sampling
            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];         % 10 Hz sampling
            % hppars.horizon_tab = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63];   % 30 Hz sampling
            % 
            % hppars.other(1).name = 'SHL';
            % hppars.other(1).val = [4, 8, 12, 16, 20];       % 3.33 Hz sampling
            % hppars.other(1).val = [12, 24, 36, 48, 60];     % 10 Hz sampling    
            % hppars.other(1).val = [36, 72, 108, 144, 180];  % 30 Hz sampling
            % 
            % hppars.other(2).name = 'learn_rate';
            % hppars.other(2).val = [0.005, 0.01, 0.02];            
            % 
            % hppars.other(3).name = 'rnn_state_space_dim'; 
            % hppars.other(3).val = [10, 25, 40];    


            % Next-frame MR image prediction (CMIG paper)   
            hppars.nb_runs_cv = 10;
            hppars.nb_runs_eval_test = 10;

            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7];  % ETH Zurich
            hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];  % Magdeburg dataset      

            hppars.other(1).name = 'SHL';
            % hppars.other(1).val = [6, 12, 18, 24, 30]; % ETH Zurich 
            hppars.other(1).val = [11, 23, 34, 45, 57]; % Magdeburg

            hppars.other(2).name = 'learn_rate';
            hppars.other(2).val = [0.005, 0.01, 0.015, 0.02];

            hppars.other(3).name = 'rnn_state_space_dim'; 
            hppars.other(3).val = [10, 30, 50, 70, 90, 110];            
            

        case 'no prediction'

            % Common parameters (always true)
            hppars.nb_runs_cv = 1;
            hppars.nb_runs_eval_test = 1;            

            hppars.other(1).name = 'SHL';
            hppars.other(1).val = [1]; % The lastest acquired value is used as the predicted value      

            % Prediction of the position of external markers (CPMB paper)
            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7];                                                               % 3.33 Hz sampling
            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];         % 10 Hz sampling
            % hppars.horizon_tab = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63];   % 30 Hz sampling
   
            % Next-frame MR image prediction (CMIG paper)
            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7];  % ETH Zurich
            hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];  % Magdeburg dataset
  
            
        case 'LMS'
            
            % Common parameters (always true)
            hppars.nb_runs_cv = 1;
            hppars.nb_runs_eval_test = 1;            

            % Prediction of the position of external markers (CPMB paper)
            %
            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7];                                                               % 3.33 Hz sampling
            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];         % 10 Hz sampling
            % hppars.horizon_tab = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63];   % 30 Hz sampling
            % 
            % hppars.other(1).name = 'SHL';           
            % hppars.other(1).val = [4, 8, 12, 16, 20];       % 3.33 Hz sampling
            % hppars.other(1).val = [12, 24, 36, 48, 60];     % 10 Hz sampling    
            % hppars.other(1).val = [36, 72, 108, 144, 180];  % 30 Hz sampling
            % 
            % hppars.other(2).name = 'learn_rate';
            % hppars.other(2).val = [0.0002, 0.0005, 0.001];             % 3.33 Hz sampling
            % hppars.other(2).val = [0.0001, 0.0002, 0.0005];            % 10 Hz sampling
            % hppars.other(2).val = [0.00005, 0.0001, 0.0002];           % 30 Hz sampling


            % Next-frame MR image prediction (CMIG paper)
            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7];  % ETH Zurich
            hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];  % Magdeburg dataset
            
            hppars.other(1).name = 'SHL';           
            % hppars.other(1).val = [6, 12, 18, 24, 30]; % ETH Zurich 
            hppars.other(1).val = [11, 23, 34, 45, 57]; % Magdeburg    
            
            hppars.other(2).name = 'learn_rate';
            hppars.other(2).val = [0.02, 0.05, 0.1, 0.2];
            
        case 'UORO'

            % Prediction of the position of external markers (CPMB paper)
            %
            % hppars.nb_runs_cv = 50;
            % hppars.nb_runs_eval_test = 300;
            % 
            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7];                                                               % 3.33 Hz sampling
            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];         % 10 Hz sampling
            % hppars.horizon_tab = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63];   % 30 Hz sampling
            % 
            % hppars.other(1).name = 'SHL'; 
            % hppars.other(1).val = [4, 8, 12, 16, 20];       % 3.33 Hz sampling
            % hppars.other(1).val = [12, 24, 36, 48, 60];     % 10 Hz sampling    
            % hppars.other(1).val = [36, 72, 108, 144, 180];  % 30 Hz sampling  
            % 
            % hppars.other(2).name = 'learn_rate';
            % hppars.other(2).val = [0.005, 0.01, 0.02];
            % 
            % hppars.other(3).name = 'rnn_state_space_dim'; 
            % hppars.other(3).val = [30, 60, 90, 120, 150, 180];


            % Next-frame MR image prediction (CMIG paper)            
            hppars.nb_runs_cv = 250;
            hppars.nb_runs_eval_test = 250;
            
            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7];  % ETH Zurich
            hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];  % Magdeburg dataset

            hppars.other(1).name = 'SHL'; 
            % hppars.other(1).val = [6, 12, 18, 24, 30]; % ETH Zurich 
            hppars.other(1).val = [11, 23, 34, 45, 57]; % Magdeburg
            
            hppars.other(2).name = 'learn_rate';
            hppars.other(2).val = [0.005, 0.01, 0.015, 0.02];
            
            hppars.other(3).name = 'rnn_state_space_dim'; 
            hppars.other(3).val = [10, 30, 50, 70, 90, 110];  
            
        case 'SnAp-1'

            % Prediction of the position of external markers (CPMB paper)
            %
            % hppars.nb_runs_cv = 50;
            % hppars.nb_runs_eval_test = 300;         
            % 
            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7];                                                               % 3.33 Hz sampling
            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];         % 10 Hz sampling
            % hppars.horizon_tab = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63];   % 30 Hz sampling
            % 
            % hppars.other(1).name = 'SHL';
            % hppars.other(1).val = [4, 8, 12, 16, 20];       % 3.33 Hz sampling
            % hppars.other(1).val = [12, 24, 36, 48, 60];     % 10 Hz sampling    
            % hppars.other(1).val = [36, 72, 108, 144, 180];  % 30 Hz sampling  
            % 
            % hppars.other(2).name = 'learn_rate';
            % hppars.other(2).val = [0.005, 0.01, 0.02];
            % 
            % hppars.other(3).name = 'rnn_state_space_dim'; 
            % hppars.other(3).val = [30, 60, 90, 120, 150, 180];

            % Next-frame MR image prediction (CMIG paper)            
            hppars.nb_runs_cv = 250;
            hppars.nb_runs_eval_test = 250;            
            
            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7];  % ETH Zurich
            hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];  % Magdeburg dataset

            hppars.other(1).name = 'SHL';
            % hppars.other(1).val = [6, 12, 18, 24, 30]; % ETH Zurich 
            hppars.other(1).val = [11, 23, 34, 45, 57]; % Magdeburg  
            
            hppars.other(2).name = 'learn_rate';
            hppars.other(2).val = [0.005, 0.01, 0.015, 0.02];
            
            hppars.other(3).name = 'rnn_state_space_dim';  
            hppars.other(3).val = [10, 30, 50, 70, 90, 110];             


        case 'DNI'

            % Prediction of the position of external markers (CPMB paper)
            %
            % hppars.nb_runs_cv = 50;
            % hppars.nb_runs_eval_test = 300;
            % 
            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7];                                                               % 3.33 Hz sampling
            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];         % 10 Hz sampling
            % hppars.horizon_tab = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63];   % 30 Hz sampling
            % 
            % hppars.other(1).name = 'SHL';
            % hppars.other(1).val = [4, 8, 12, 16, 20];       % 3.33 Hz sampling
            % hppars.other(1).val = [12, 24, 36, 48, 60];     % 10 Hz sampling    
            % hppars.other(1).val = [36, 72, 108, 144, 180];  % 30 Hz sampling  
            % 
            % hppars.other(2).name = 'learn_rate';
            % hppars.other(2).val = [0.005, 0.01, 0.02];            
            % 
            % hppars.other(3).name = 'learn_rate_A';           
            % hppars.other(3).val = [0.002];  
            % 
            % hppars.other(4).name = 'rnn_state_space_dim'; 
            % hppars.other(4).val = [30, 60, 90, 120, 150, 180];

            % Next-frame MR image prediction (CMIG paper)      
            hppars.nb_runs_cv = 250;
            hppars.nb_runs_eval_test = 250;  
   
            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7];  % ETH Zurich
            hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];  % Magdeburg dataset 

            hppars.other(1).name = 'SHL';
            % hppars.other(1).val = [6, 12, 18, 24, 30]; % ETH Zurich 
            hppars.other(1).val = [11, 23, 34, 45, 57]; % Magdeburg

            hppars.other(2).name = 'learn_rate';
            hppars.other(2).val = [0.005, 0.01, 0.015, 0.02];         

            hppars.other(3).name = 'learn_rate_A';           
            hppars.other(3).val = [0.002];  

            hppars.other(4).name = 'rnn_state_space_dim'; 
            hppars.other(4).val = [10, 30, 50, 70, 90, 110];            

        case 'fixed W'
            
            % Prediction of the position of external markers (CPMB paper)
            % 
            % hppars.nb_runs_cv = 50;
            % hppars.nb_runs_eval_test = 300;
            % 
            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7];                                                               % 3.33 Hz sampling
            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];         % 10 Hz sampling
            % hppars.horizon_tab = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63];   % 30 Hz sampling
            % 
            % hppars.other(1).name = 'SHL';
            % hppars.other(1).val = [4, 8, 12, 16, 20];       % 3.33 Hz sampling
            % hppars.other(1).val = [12, 24, 36, 48, 60];     % 10 Hz sampling    
            % hppars.other(1).val = [36, 72, 108, 144, 180];  % 30 Hz sampling  
            % 
            % hppars.other(2).name = 'learn_rate';
            % hppars.other(2).val = [0.005, 0.01, 0.02];             
            % 
            % hppars.other(3).name = 'rnn_state_space_dim'; 
            % hppars.other(3).val = [30, 60, 90, 120, 150, 180];

            % Next-frame MR image prediction (CMIG paper)      
            hppars.nb_runs_cv = 250;
            hppars.nb_runs_eval_test = 250;  
   
            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7];  % ETH Zurich
            hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];  % Magdeburg dataset

            hppars.other(1).name = 'SHL';
            % hppars.other(1).val = [6, 12, 18, 24, 30]; % ETH Zurich 
            hppars.other(1).val = [11, 23, 34, 45, 57]; % Magdeburg

            hppars.other(2).name = 'learn_rate';
            hppars.other(2).val = [0.005, 0.01, 0.015, 0.02];   

            hppars.other(3).name = 'rnn_state_space_dim'; 
            hppars.other(3).val = [10, 30, 50, 70, 90, 110];            

        case 'SVR' % Only in the CMPB paper - external marker position forecasting
            
            % Common parameters (always true)
            hppars.nb_runs_cv = 1;            
            hppars.nb_runs_eval_test = 1;
            
            % Prediction of the position of external markers (CPMB paper)
            hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7];                                                               % 3.33 Hz sampling
            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];         % 10 Hz sampling
            % hppars.horizon_tab = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63];   % 30 Hz sampling

            hppars.other(1).name = 'SHL';
            hppars.other(1).val = [4, 8, 12, 16, 20];       % 3.33 Hz sampling
            % hppars.other(1).val = [12, 24, 36, 48, 60];     % 10 Hz sampling    
            % hppars.other(1).val = [36, 72, 108, 144, 180];  % 30 Hz sampling            

            hppars.other(2).name = 'svr_kernel_scale';
            hppars.other(2).val = [100, 200, 500, 1000];            

            hppars.other(3).name = 'svr_epsilon'; 
            hppars.other(3).val = [0.005, 0.01, 0.02, 0.05];

            hppars.other(4).name = 'svr_box_constraint';
            hppars.other(4).val = [100, 200, 500, 1000];

        case 'transformer' % Only in the CMIG paper - future MR frame forecasting
            
            % Common parameters (always true)
            hppars.nb_runs_cv = 10;            
            hppars.nb_runs_eval_test = 10;

            % Next-frame MR image prediction (CMIG paper)
            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7];  % ETH Zurich
            hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];  % Magdeburg dataset

            hppars.other(1).name = 'SHL';
            % hppars.other(1).val = [6, 12, 18, 24, 30]; % ETH Zurich 
            hppars.other(1).val = [11, 23, 34, 45, 57]; % Magdeburg                         

            hppars.other(2).name = 'learn_rate';
            hppars.other(2).val = [0.0001, 0.0005];            

            hppars.other(3).name = 'num_layers'; 
            hppars.other(3).val = [1, 2];
            
            hppars.other(4).name = 'd_model';
            hppars.other(4).val = [8, 16];

        case 'population_transformer' % Only in the CMIG paper - future MR frame forecasting
            
            % Common parameters (always true)
            % hppars.nb_runs_cv = 10;               % try 10 in the paper      
            % hppars.nb_runs_eval_test = 10;        % try 10 in the paper  

            hppars.nb_runs_cv = 10;               % value for testing
            hppars.nb_runs_eval_test = 10;        % value for testing 

            % Next-frame MR image prediction (CMIG paper)
            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7];  % ETH Zurich
            % hppars.horizon_tab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];  % Magdeburg dataset

            hppars.horizon_tab = [1, 7];        % that's a test; I have 2 horizons at the moment

            % There is no hyperparameter to optimize on, rather we load trained models and evaluate them
            hppars.other(1).name = 'dummy';
            hppars.other(1).val = [1];

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

