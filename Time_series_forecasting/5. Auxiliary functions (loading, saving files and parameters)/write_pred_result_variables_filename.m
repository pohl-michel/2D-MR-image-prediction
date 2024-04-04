function [str] = write_pred_result_variables_filename(path_par, pred_par, br_model_par)
% returns the filename for the file containing the prediction results 
%
% Author : Pohl Michel
% Date : September 11th, 2021
% Version : v1.1
% License : 3-clause BSD License

    str = sprintf('%s\\pred_result_variables %s %s %s', path_par.temp_var_dir, path_par.time_series_dir, pred_par.pred_meth, sprintf_pred_param(pred_par));
    if nargin == 3 % br_model_par is passed as an argument
        str = sprintf('%s %d pca cpts', str, br_model_par.nb_pca_cp);
    end
    str = sprintf('%s.mat', str);

end