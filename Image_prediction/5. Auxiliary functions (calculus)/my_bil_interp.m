function interVal = my_bil_interp(img, Ymax, Xmax, y, x)
% Bilinear interpolation using 4 pixels around the target location with ceil convention
% img is a 2D image
% y and x the coordinates of the point to be interpolated
% zeros are used for pixel values outside of the given image
%
% Example:
% [m,n]=meshgrid(1:3);img=[m+n]
% --> 2     3     4
%     3     4     5
%     4     5     6
% my_bil_interp(img,3, 3, 2.4, 2.2)
% --> 4.6
%
% Remark: this function takes Xmax and Ymax as an input because "my_bil_interp" is called many times
% (so there may be a calculation time gain by avoiding to retrieve the image size too many times)
%
% Author : Pohl Michel
% Date : Sept 18th, 2022
% Version : v1.0
% License : 3-clause BSD License


    x0 = floor(x);
    y0 = floor(y);
    
    eps_x = x - x0; eps_conj_x = 1 - eps_x;
    eps_y = y - y0; eps_conj_y = 1 - eps_y;
    
    interVal = ...
        eps_conj_x*(eps_conj_y*pixLookup(img, Ymax, Xmax, y0, x0  ) + eps_y*pixLookup(img, Ymax, Xmax, y0+1, x0  )) + ...
        eps_x*     (eps_conj_y*pixLookup(img, Ymax, Xmax, y0, x0+1) + eps_y*pixLookup(img, Ymax, Xmax, y0+1, x0+1));

end


function pixVal = pixLookup(img, Ymax, Xmax,y,x)
   % in this function x and y are integer coordinates

   if (x<=0)||(x>Xmax)||(y<=0)||(y>Ymax)
       pixVal = 0; % padding with zeros
   else
       pixVal = img(y,x);
   end
end
