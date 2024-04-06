import os
import shutil
import pydicom as dicom
import pandas as pd
import cv2 

# we skip some images due to varying contrast
NB_SKIPPED_IM = 15

# For saving a folder with jpg images 
SAVE_JPG = True

org_base_dir = "/mnt/d/OneDrive/Research in Uesaka Lab/1.5. Image databases/2D MRI liver slices with navigator frames pt 2"
org_data_rel_dirs = [
    "2020-11-10_KS81/Nav_Pur_1",
    "2020-11-12_QN76/Nav_Pur_1",
    "2020-11-17_CS31/Nav_Pur_2",
    "2020-11-17_JY02/Nav_Pur_2",
    "2020-11-23_ON65/Nav_Pur_2",
    "2020-11-23_PS11/Nav_Pur_1",
    "2020-11-25_II29/Nav_Pur_1",
    "2020-11-26_NE38/Nav_Pur_1"
]

out_base_dir = "/mnt/d/Programming2/Matlab_workspace/Future_frame_prediction/Image_prediction/input_imgs/2D images"

for rel_dir in org_data_rel_dirs:

    org_data_full_path = org_base_dir + "/" + rel_dir
    imgs = sorted(os.listdir(org_data_full_path)) # sorted to have the images in the correct order

    # Instead of using one directory inside another, each navigator sequence can have its own directory inside out_base_dir,
    # so "join" and "split" are used
    out_dir = out_base_dir + "/" + "_".join(rel_dir.split("/"))

    if os.path.exists(out_dir):
        print(f"The directory {out_dir} already exists! Skipping image sequence {rel_dir}...")
        continue

    print(f"Processing image sequence {rel_dir} ...")
    os.makedirs(out_dir)

    if SAVE_JPG:
        jpg_dir = out_dir + "/jpg_images"
        os.makedirs(jpg_dir)

    for org_im_idx in range(NB_SKIPPED_IM, len(imgs)):

        org_im_path = org_data_full_path + "/" + imgs[org_im_idx]

        out_im_idx = org_im_idx - NB_SKIPPED_IM + 1
        out_im_name = "image" + str(out_im_idx) + ".IMA"
        out_im_path = out_dir + "/" + out_im_name

        shutil.copyfile(org_im_path, out_im_path)

        if SAVE_JPG: # reference: https://stackoverflow.com/questions/48185544/read-and-open-dicom-images-using-python 
            ds = dicom.dcmread(org_im_path)
            pixel_array_numpy = ds.pixel_array

            out_jpg_im_name = "image" + str(out_im_idx) + ".jpg"
            jpg_im_path = jpg_dir + "/" + out_jpg_im_name
            cv2.imwrite(jpg_im_path, pixel_array_numpy)

    # Creating the im_par.xlsx file
    # Rq: org_im_path now corresponds to the last image in the original directory (the previous for loop ended)
    im_pars = {}
    im_pars["nb_im"] = len(imgs) - NB_SKIPPED_IM
    im_pars["imtype"] = str.split(org_im_path, ".")[-1]
    
    # Extracting height and length and putting that info into im_pars
    ds = dicom.dcmread(org_im_path)
    pixel_array_numpy = ds.pixel_array
    W, L = pixel_array_numpy.shape # W is the height of the 2D image
    im_pars["W"] = W
    im_pars["L"] = L

    # converting im_pars dict into pandas dataframe and then the latter into an excel file
    im_pars_df = pd.DataFrame([im_pars]) # reference: https://stackoverflow.com/questions/18837262/convert-python-dict-into-a-dataframe  
        # Rk: I use [im_pars] and not im_pars.items() because I want the keys to be the columns
    im_par_xlsx_path = out_dir + "/" + "im_seq_par.xlsx"
    with pd.ExcelWriter(im_par_xlsx_path) as writer:  
        im_pars_df.to_excel(writer, index=False) # index=False to not have a first column with zeros

    # To do also: create the directory with jpg images, it will be helpful when writing the paper
    # https://stackoverflow.com/questions/48185544/read-and-open-dicom-images-using-python