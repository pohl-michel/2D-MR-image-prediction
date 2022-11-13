function par_influence = evaluate_par_influence_grid(hppars, pred_par, optim)
% The influence of each hyper-parameter on prediction performance is evaluated after performing grid search
%
% Author : Pohl Michel
% Date : October 16, 2022
% Version : v1.1
% License : 3-clause BSD License

    % Average prediction time to make one prediction (Columns : number of neurons in the hidden layer / lines : SHL)
    temp_avg_time = 0; % broadcasting as the size of temp_avg_time depends on pred_par.pred_meth
    for hrz_idx = 1:hppars.nb_hrz_val
        pred_time_tab = optim(hrz_idx).pred_time_tab;
        for hppar_idx = 1:hppars.nb_additional_params
            switch(pred_par.pred_meth)
                case {'RTRL', 'UORO', 'SnAp-1', 'DNI', 'RTRL v2', 'fixed W'} % prediction with an RNN
                    if (hppar_idx ~= hppars.state_space_hyppar_idx)&&(hppar_idx ~= hppars.SHL_hyppar_idx)
                        % we want to study the influence of the number of hidden neurons and SHL so we do not compute the mean over these variables 
                        pred_time_tab = mean(pred_time_tab, hppar_idx); 
                    end
                otherwise
                    if (hppar_idx ~= hppars.SHL_hyppar_idx)
                        % we want to study the influence of the SHL so we do not compute the mean over it
                        pred_time_tab = mean(pred_time_tab, hppar_idx);
                    end
            end 
        end
        temp_avg_time = temp_avg_time + pred_time_tab;
    end
    temp_avg_time = temp_avg_time/hppars.nb_hrz_val; % mean calculation
    par_influence.pred_time_avg = squeeze(temp_avg_time);
    
    if (hppars.nb_additional_params >=1) % we eliminate the case without prediction
        par_influence.min_nRMSE = cell(hppars.nb_additional_params, hppars.nb_hrz_val);
        if (hppars.nb_additional_params ==1) % typically linear regression
            for hrz_idx = 1:hppars.nb_hrz_val
                par_influence.min_nRMSE{1, hrz_idx} = optim(hrz_idx).nrmse_tab;
                % minimum of the nRMSE over the cross validation set
            end
        else
            for hppar_idx = 1:hppars.nb_additional_params
                for hrz_idx = 1:hppars.nb_hrz_val
                    vecdim = 1:hppars.nb_additional_params;
                    vecdim(hppar_idx) = [];
                    par_influence.min_nRMSE{hppar_idx, hrz_idx} = min(optim(hrz_idx).nrmse_tab, [], vecdim);
                    % minimum of the nRMSE over the cross validation set
                end
            end
        end
    end

end