# Script for creating gif file given ground-truth sequence and predicted sequence
# Author: Michel Pohl
# License: 3-clause BSD License

# Remark: possible improvement - the Matlab script outputs original images and predicted images with specific filenames
# That could be taken into account to get the output .gif file more easily after dragging and dropping the images to the gt and pred directories without renaming manually

import os 
from PIL import Image, ImageSequence

# sequence 1 in the paper
# gt_img_dir = "2._sq_sl010_sag_Xcs=125_gt_t=181_to_200"
# pred_img_dir = "prediction_2._sq_sl010_sag_Xcs=125_SnAp-1_k=12_q=110_eta=0.02_sg=0.02_h=6_3_cpts_t=181_to_200"

# sequence 1 in the paper - cropped
# gt_img_dir = "2._sq_sl010_sag_Xcs=125_gt_t=181_to_200_cropped"
# pred_img_dir = "prediction_2._sq_sl010_sag_Xcs=125_SnAp-1_k=12_q=110_eta=0.02_sg=0.02_h=6_3_cpts_t=181_to_200_cropped"

# sequence 2 in the paper 
# gt_img_dir = "3._sq_sl010_sag_Xcs=80_gt_t=181_to_200"
# pred_img_dir = "prediction_3._sq_sl010_sag_Xcs=80_SnAp-1_k=30_q=70_eta=0.02_sg=0.02_h=6_4_cpts_t=181_to_200"

# sequence 2 in the paper - cropped
# gt_img_dir = "3._sq_sl010_sag_Xcs=80_gt_t=181_to_200_cropped"
# pred_img_dir = "prediction_3._sq_sl010_sag_Xcs=80_SnAp-1_k=30_q=70_eta=0.02_sg=0.02_h=6_4_cpts_t=181_to_200_cropped"

# sequence 3 in the paper 
# gt_img_dir = "4. sq sl014 sag Xcs=165_gt_t=181_to_200"
# pred_img_dir = "prediction_4. sq sl014 sag Xcs=165 SnAp-1 k=6 q=110 eta=0.01 sg=0.02 h=6 3 cpts_t=181_to_200"

# sequence 3 in the paper - gt and predicted images cropped
# gt_img_dir = "4. sq sl014 sag Xcs=165_gt_t=181_to_200_cropped"
# pred_img_dir = "prediction_4. sq sl014 sag Xcs=165 SnAp-1 k=6 q=110 eta=0.01 sg=0.02 h=6 3 cpts_t=181_to_200_cropped"

# sequence 4 in the paper 
# gt_img_dir = "5. sq sl014 sag Xcs=95_gt_t=181_to_200"
# pred_img_dir = "prediction_5. sq sl014 sag Xcs=95 SnAp-1 k=6 q=110 eta=0.01 sg=0.02 h=6 2 cpts_t=181_to_200"

# sequence 4 in the paper - cropped
gt_img_dir = "5. sq sl014 sag Xcs=95_gt_t=181_to_200_cropped"
pred_img_dir = "prediction_5. sq sl014 sag Xcs=95 SnAp-1 k=6 q=110 eta=0.01 sg=0.02 h=6 2 cpts_t=181_to_200_cropped"

root_folder = os.path.dirname(__file__)
output_file = os.path.join(root_folder, "output.gif")

nb_imgs = 20
display_time_per_frame_ms = 500

# Paths to the folders containing your two image sequences
sequence1 = [os.path.join(root_folder, gt_img_dir, f"frame_{i}.jpg") for i in range(1, nb_imgs + 1)]
sequence2 = [os.path.join(root_folder, pred_img_dir, f"frame_{i}.jpg") for i in range(1, nb_imgs + 1)]

# Make sure both sequences have the same number of images and sizes
assert len(sequence1) == len(sequence2), "Both sequences must have the same number of frames."

combined_images = []

for img1_path, img2_path in zip(sequence1, sequence2):
    # Open both images
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # Make sure both images have the same height
    if img1.height != img2.height:
        img2 = img2.resize((img2.width, img1.height))

    # Combine them side by side
    combined_width = img1.width + img2.width
    combined_image = Image.new('RGB', (combined_width, img1.height))
    
    combined_image.paste(img1, (0, 0))
    combined_image.paste(img2, (img1.width, 0))

    # Add the combined image to the list
    combined_images.append(combined_image)

# Save the combined images as a GIF

combined_images[0].save(
   output_file, save_all=True, append_images=combined_images[1:], duration=display_time_per_frame_ms, loop=0
)