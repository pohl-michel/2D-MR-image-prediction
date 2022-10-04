# Script loading reconstructed a 3D mri complex image sequence (npy format) 
# and returning corresponding matlab (real - magnitude) cross section sequences
#
# Author : Pohl Michel
# Date : October 3, 2022
# Version : v1.1
# License : 3-clause BSD License


import numpy as np
import scipy.io
from scipy import ndimage as nd
import itk
from PIL import Image
import os
import glob

# Program behavior
INVERT_Z_AXIS = True        # Inverting the original image so that the head appears on top (very specific to the current sequence)
REMOVE_FIRST_IMAGES = False  
    # If set to true, the original images up to t_start (non-inlcuded) are not loaded and saved in the output sequence
    # This can be useful if there are artifacts (e.g., varying contrast) in the initial images of the sequence 

# Parameters
zoomFactors = [1.8, 1, 1] # size of each voxel (in mm) in the original image (info from the original dicom file)
sag_Xcs_tab = [215, 195, 175] # coordinates of each sagittal cross section that we extract from the original image sequence 
t_start = 126

# Path variables
input_3Dim_path = "Image_prediction/a. Input images/Original 3D images"
imseq_name = "3. extreme_mri_DCE_dataset_copy" # info: resampled_vol_shape = [227, 163, 370]
imseq_folder = f"{input_3Dim_path}/{imseq_name}"
output_2Dim_path = "Image_prediction/a. Input images/2D images"
jpg_im_folder = "jpg images"

nb_im = len(glob.glob1(imseq_folder,"*.npy"))
t_start = t_start if REMOVE_FIRST_IMAGES else 0

for t in range(nb_im):

    print('Opening 3D image at t=%d' % (t+1))
    input_im_path = f"{imseq_folder}/Surya - dce_slice_{t}.npy"
    img_array = np.load(input_im_path, allow_pickle=True)
    magnitude_img_array = np.abs(img_array) # the input image is an array of complex numbers but we only care about their absolute values here.
    rspled_magnitude_img_array = nd.interpolation.zoom(magnitude_img_array, zoom=zoomFactors)
        # interpolation with cubic splines: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.interpolation.zoom.html

    if t == 0:
        H, L, Z = rspled_magnitude_img_array.shape
        cs_imgs = np.zeros((H, L, nb_im, len(sag_Xcs_tab)))

    for sag_Xcs_idx, sag_Xcs in enumerate(sag_Xcs_tab):
        cs_imgs[:, :, t, sag_Xcs_idx] = rspled_magnitude_img_array[:,:,sag_Xcs]

if INVERT_Z_AXIS:
    temp = cs_imgs.copy()
    for y in range(H):
        cs_imgs[y, :, : ,:] = temp[H-y-1, :, :, :]

for sag_Xcs_idx, sag_Xcs in enumerate(sag_Xcs_tab):

    # normalization between 0 and 255
    cs_max_intensity = cs_imgs[:, :, :, sag_Xcs_idx].max()
    cs_imgs[:, :, :, sag_Xcs_idx] = ((255/cs_max_intensity)*cs_imgs[:, :, :, sag_Xcs_idx]).astype('uint8')

    # output directory creation
    crt_2Dim_sq_path = f"{output_2Dim_path}/{imseq_name} Xcs = {sag_Xcs}"
    crt_jpg_im_sq_path = f"{crt_2Dim_sq_path}/{jpg_im_folder}"
    if REMOVE_FIRST_IMAGES:
        crt_2Dim_sq_path = crt_2Dim_sq_path + f" from t = {t_start}"
        crt_jpg_im_sq_path = crt_jpg_im_sq_path + f" from t = {t_start}"
    os.makedirs(crt_2Dim_sq_path)
    os.makedirs(crt_jpg_im_sq_path)

    print('saving cross-section images at sagittal coordinate X=%d' % sag_Xcs)

    for t in range(t_start, nb_im):

        sag_cs_array = cs_imgs[:, :, t, sag_Xcs_idx]

        # saving as mat file
        output_mat_name = '%s/image%d.mat' % (crt_2Dim_sq_path, t + 1 - t_start)
        scipy.io.savemat(output_mat_name, {'im': sag_cs_array})

        # saving as jpg file
        output_jpg_filename = '%s/image%d.jpeg' % (crt_jpg_im_sq_path, t + 1 - t_start)
        sag_cs_im_PIL = Image.fromarray(sag_cs_array)
        sag_cs_im_PIL.convert("L").save(output_jpg_filename)