function create_2Dim_directory(path_par, output_im_dir, org_im_par, new_im_par, transformation_par, im_seq_idx)
% Creates a new directory in the folder with 2D images corresponding to the sequences to be processed.
% The display parameters, image parameters, and transformation parameters are saved within that folder.
%
% Author : Pohl Michel
% Date : September 26th, 2022
% Version : v1.0
% License : 3-clause BSD License


    % Creation of a new folder containing the new image sequence
    if exist(output_im_dir, 'dir')
        msgID = 'output_folder:already_existing';
        msg = 'The folder which you attempt to create is already existing';
        ME = MException(msgID, msg);
        throw(ME)
    end    
    mkdir(output_im_dir);
    mkdir(path_par.temp_im_dir);
    
    % Creation of excel files containing the display parameters associated with the new image sequence
    disp_par_org_flname = sprintf('%s\\disp_par.xlsx', path_par.input_im_dir);
    copyfile(disp_par_org_flname, output_im_dir);
    
    % Saving the excel file containing im_par 
    im_seq_par_dest_flname = sprintf('%s\\im_seq_par.xlsx', output_im_dir);
    writetable(struct2table(new_im_par), im_seq_par_dest_flname);
    
    % text file with the parameters for creating the sequence
    log_file_complete_filename = sprintf('%s\\%s', output_im_dir, path_par.im_seq_par_txt_filename);
    fid = fopen(log_file_complete_filename,'wt');
    
    fprintf(fid, 'original image sequence : %s \n', path_par.input_im_seq);
    fprintf(fid, 'number of images : %d \n', new_im_par.nb_im);
    fprintf(fid, 'The 1st image in the new sequence corresponds to the %d -th image in the original sequence \n\n', transformation_par.t_init(im_seq_idx));
    fprintf(fid, 'Saggital slice x = %d \n', transformation_par.Xcs(im_seq_idx));     
    fclose(fid);

end

