'''
Created on Apr 24, 2025

@author: sac
'''
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Directory containing .tif files
tif_dir = '/home/sac/saptadeep/Hyperspectral_Denoising/Dataset/more_hype_images/EO1H1990292017069110K3_1T/'

# Get all .tif files
tif_files = sorted([f for f in os.listdir(tif_dir) if f.endswith(('.TIF'))])

def extract_useful_bamds(tif_dir,n):
    
    # Get all .tif files
    tif_files = sorted([f for f in os.listdir(tif_dir) if f.endswith(('.TIF'))])
    
    # Load images into a list
    tif_images = []
    for file in tif_files:
        try:
            file_path = os.path.join(tif_dir, file)
            img = Image.open(file_path)
            img_array = np.array(img)
            tif_images.append(img_array)
        except (IOError, OSError) as e:
    #         print(f"Skipping corrupted image: {tif_files[file]} | Error: {e}")
            continue
      
    # Try stacking into a 3D or 4D numpy array
    try:
        tif_array = np.stack(tif_images)
        print(f"Stacked array shape: {tif_array.shape}")
    except ValueError:
        print("Images have different shapes. Keeping as list.")
        tif_array = tif_images
    
        
    def contrast_enhancement(local_im):
        non_zero_min = np.min(local_im[local_im > 0])
        im_max_ = np.quantile(local_im[local_im > 0],0.995)
        local_im = (local_im - non_zero_min) / (im_max_  - non_zero_min)
        local_im[local_im < 0] = 0
        local_im[local_im > 1] = 1
        return local_im
    
    tif_array = contrast_enhancement(tif_array)
    tif_array = np.transpose(tif_array,(1,2,0))
    print(f"Transposed array: {tif_array.shape}")
    
    #1st array
    real_ = []
    for i in range(8,58):
        tif_array_un_norm_new = tif_array[:,:,i]
        real_.append(tif_array_un_norm_new)
    real_arr = np.array(real_).transpose(1,2,0)
    print(f"1st array shape {real_arr.shape}")
    # exit(0)
    
    #2nd array
    real_2 = []
    for i in range(77,tif_array.shape[2]):
        tif_array_un_norm_new = tif_array[:,:,i]
        real_2.append(tif_array_un_norm_new)
    real_arr_2 = np.array(real_2).transpose(1,2,0)
    print(f"2nd array shape {real_arr_2.shape}")
    
    re = np.concatenate((real_arr,real_arr_2), axis = -1) 
    
    print(f"Shape of concatenated array: {re.shape}")
    # plt.figure
    # plt.imshow(tif_array[:,:,10], cmap = "gray")
    # plt.show()
    # exit(0)
    fp = "/home/sac/saptadeep/Hyperspectral_Denoising/Dataset/New_batch/"
    im_nm = f"image_{n}_norm"
    fp_ = fp+im_nm
    np.save(fp_, re)
    print(f"File {n} is saved at: {fp_}")

# save_npy = extract_useful_bamds(tif_dir, 10)
# print(save_npy)






