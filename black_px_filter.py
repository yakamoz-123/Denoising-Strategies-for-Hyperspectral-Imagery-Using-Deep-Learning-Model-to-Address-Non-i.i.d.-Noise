'''
Created on Apr 25, 2025

@author: sac
'''

import numpy as np
import matplotlib.pyplot as plt
import os
from osgeo import gdal


def extract_patches(img, patch_size):
#     img = np.load(npy_file)
    stride = patch_size
    h, w = img.shape
    print(f"initial img size: {img.shape}")
    patches = []
      
#     for b in range(img.shape[2]):
    for y in range(b, h - patch_size + 1, stride):
        for x in range(w - patch_size + 1, stride):
            patch = img[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
    patches_ar = np.array(patches).transpose(1,2,0)
    print(f"patches shape: {patches_ar.shape}")
     
    return patches_ar
# 
# 
def filter_black_pixel_patches(patches_ar):
     
    H, W, N = patches_ar.shape
    print(f"initial patches shape: {patches_ar.shape}")
    filtered = []
  
    for i in range(N):
        patch = patches_ar[:, :,i]
        if np.all(patch != 0):  # If no pixel is black
            filtered.append(patch)
       
    if len(filtered) > 0:
        filtered_ar = np.stack(filtered, axis=2)
        print(f"Only Filtered shape: {filtered_ar.shape}")
        return filtered_ar
    else:
        print("Array is empty")
#        
def extract_filter_save(img, patch_size):
#     for i in range(img.shape[2]):
    extract_128 = extract_patches(img, patch_size)
    filtered = filter_black_pixel_patches(extract_128)
    print(f"Shape of filtered array is: {filtered.shape}")
    plt.figure()
    plt.imshow(filtered[:,:,3000], cmap = "gray")
    plt.show()
#     np.save(f"/home/sac/saptadeep/Hyperspectral_Denoising/Dataset/New_hype_filtered/filtered_img_{im_no}", filtered)
    return filtered
#     
#          
# 
# fp = "/home/sac/saptadeep/Hyperspectral_Denoising/Dataset/New_hype_filtered/New_batch_unfiltered/"
# for i in range(9,10):
# #     im_no = 3
#     npy_nm = f"image_{i}_norm.npy"
#     file_names = fp+npy_nm
#     
#     # i = np.load(file_names)
#     # print(i.shape)
#     # exit(0)
#     patches_ar = extract_patches(file_names, 128)
#     filter_patches = filter_black_pixel_patches(patches_ar)
#     npy_save = extract_filter_save(file_names, 128, i)


# plt.figure()
# plt.imshow(save_npy[:,:,200], cmap ="gray")
# plt.show()

hyperion_imgs = [
"/home/sac/saptadeep/Hyperspectral_Denoising/Dataset/more_hype_images/EO1H0260472017068110KF_1T/EO1H0260472017068110KF_B160_L1T.TIF",
"/home/sac/saptadeep/Hyperspectral_Denoising/Dataset/more_hype_images/EO1H0430342017071110KF_1T/EO1H0430342017071110KF_B160_L1T.TIF",
# "/home/sac/saptadeep/Hyperspectral_Denoising/Dataset/more_hype_images/EO1H0470172017069110K4_1T/EO1H0470172017069110K4_B160_L1T.TIF",
# "/home/sac/saptadeep/Hyperspectral_Denoising/Dataset/more_hype_images/EO1H0690172017070110KF_1T/EO1H0690172017070110KF_B160_L1T.TIF",
"/home/sac/saptadeep/Hyperspectral_Denoising/Dataset/more_hype_images/EO1H1290502017068110KF_1T/EO1H1290502017068110KF_B160_L1T.TIF",
# "/home/sac/saptadeep/Hyperspectral_Denoising/Dataset/more_hype_images/EO1H1580372017070110K8_1T/EO1H1580372017070110K8_B160_L1T.TIF",
"/home/sac/saptadeep/Hyperspectral_Denoising/Dataset/more_hype_images/EO1H1710602017067110T1_1T/EO1H1710602017067110T1_B160_L1T.TIF",
"/home/sac/saptadeep/Hyperspectral_Denoising/Dataset/more_hype_images/EO1H1890172017065110KT_1T/EO1H1890172017065110KT_B160_L1T.TIF",
# "/home/sac/saptadeep/Hyperspectral_Denoising/Dataset/more_hype_images/EO1H1930492017068110T2_1T/EO1H1930492017068110T2_B160_L1T.TIF",
"/home/sac/saptadeep/Hyperspectral_Denoising/Dataset/more_hype_images/EO1H1990292017069110K3_1T/EO1H1990292017069110K3_B160_L1T.TIF"
]

im_ar = []
for im_nm in hyperion_imgs:
    img = gdal.Open(im_nm).ReadAsArray()
#     img_patches = extract_filter_save(img, 128)
    #Normalize
    img_min = np.min(img_patches[img_patches!=0]) #(all 1)
    img_max = np.quantile(img_patches,0.99) #455,380,334,233,198,240
    print(img_min, img_max)
    img = (img - img_min) / (img_max - img_min)
    img[img < 0] = 0
    img[img > 1] = 1 
    print(img.shape)
    
exit(0)
    



directory = "/home/sac/saptadeep/Hyperspectral_Denoising/Dataset/128_window_channels/Modified_Patches/separate_noise/"
filenames = [f for f in os.listdir(directory) if f.endswith(".npy")]
  
# print(filenames)
# exit(0)
clean =[]
for i in range(len(filenames)):
    clean_f = directory+filenames[i]
      
#     clean_load = np.load(clean_f)
#     print(f"shape of {filenames[i]}: {clean_load.shape}")
    clean.append(clean_f)
  
# print(clean)
# exit(0)
# c = np.load(directory+"high_noisy_img_10.npy")
# print(c.shape)
# exit(0)



patch_dir = "/home/sac/saptadeep/Hyperspectral_Denoising/Dataset/128_window_channels/Modified_Patches/Noisy_patches/"
def create_single_patches(img_list, patch_dir):
    global_counter=1
    for img_idx, img_path in enumerate(img_list, start = 1):
        print(f"Image Processing {img_idx}")
        
        load_npy = np.load(img_path)
        h,w,bands = load_npy.shape
        
        for b in range(bands):
            single_band = load_npy[:,:,b]
            
            fn = f"noisy_high_patch_{global_counter}_img_{img_idx}"
#             save_path = os.path.join(patch_dir,)
            np.save(patch_dir+fn, single_band)
            print(f" Saved band {b} as {fn}")
            global_counter+=1

    print(f"\n All bands ar saved at {patch_dir}")
    
save_single_patches = create_single_patches(clean, patch_dir)




















