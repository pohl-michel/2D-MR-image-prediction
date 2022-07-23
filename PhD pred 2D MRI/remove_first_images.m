clear all 
close all
clc

input_im_dir = 'Input images\2D images\16. 2nd DCE sag Xcs = 195 from t=126';
t_start = 126;

for t=1:(t_start-1)
    im_filename = sprintf('%s\\image%d.mat',input_im_dir, t);
    delete(im_filename)
end

for t=t_start:500
   
    in_im_filename = sprintf('%s\\image%d.mat',input_im_dir, t);
    load(in_im_filename, 'im');
    
    out_im_filename = sprintf('%s\\image%d.mat',input_im_dir, t-t_start+1);
    save(out_im_filename, 'im');
    
    delete(in_im_filename)
    
end