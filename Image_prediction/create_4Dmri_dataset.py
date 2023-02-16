import os
import shutil

# we skip some images due to varying contrast
NB_SKIPPED_IM = 13

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

out_base_dir = "/mnt/d/Programming2/Matlab_workspace/Future_frame_prediction/Image_prediction/a. Input images/2D images"

for rel_dir in org_data_rel_dirs:

    org_data_full_path = org_base_dir + "/" + rel_dir
    imgs = sorted(os.listdir(org_data_full_path)) # sorted to have the images in the correct order

    acquisition_identifier = rel_dir[:16]
    out_dir = out_base_dir + "/" + acquisition_identifier
    if os.path.exists(out_dir):
        print(f"The directory {out_dir} already exists! Skipping this image sequence...")
        continue

    os.makedirs(out_dir)

    im_par_file_exists = False

    for org_im_idx in range(NB_SKIPPED_IM, len(imgs)):

        org_im_path = org_data_full_path + "/" + imgs[org_im_idx]

        out_im_idx = org_im_idx - NB_SKIPPED_IM + 1
        out_im_name = "image" + str(out_im_idx) + ".IMA"
        out_im_path = out_dir + out_im_name

        shutil.copyfile(org_im_path, out_im_path)

        if not im_par_file_exists:

            # Creating the im_par.xlsx file
            im_pars = {}
            im_pars["nb_im"] = len(imgs) - NB_SKIPPED_IM
            im_pars["imtype"] = str.split(org_im_path, ".")[-1]
            
            # to do here:
            # - extract height and length and put that info into im_pars
            # - convert dict into pandas dataframe and then the latter into an excel file (it should work)

            im_par_file_exists = True

    # To do also: create the directory with jpg images, it will be helpful when writing the paper
    # https://stackoverflow.com/questions/48185544/read-and-open-dicom-images-using-python