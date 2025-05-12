'''
Created on Apr 21, 2025

@author: sac
'''
import numpy as np
import matplotlib.pyplot as plt

# 1. Load images as uint16
file = "/home/sac/saptadeep/Hyperspectral_Denoising/HYSIS_data/LTC_noise_compo/"
low_noise_dt = "low_intensity_noise_patched.npy"
mid_noise_dt = "mid_intensity_noise_patched.npy"
high_noise_dt = "high_intensity_noise_patched.npy"
img = np.load(file + low_noise_dt)
# # plt.show()
# print(img.shape)
# exit(0)

# clean_dt = "clean_img.npy"
# noise_img = np.load(high_noise_dt)
# noise_ = noise_img.astype(np.uint16)

# plt.figure()
# plt.imshow(noise_[:,:,3], cmap= "gray")

def extract_patches(img_path, patch_size, img_name):
    img = np.load(file + img_path).astype(np.uint16)
    
    stride = patch_size
    h, w, b = img.shape
    patches = []
    
    for b in range(img.shape[2]):
        for y in range(b, h - patch_size + 1, stride):
            for x in range(b, w - patch_size + 1, stride):
                patch = img[y:y+patch_size, x:x+patch_size, b]
                patches.append(patch)
    patches_arr = np.array(patches).transpose(1,2,0)
    print(f"Shape of patches: {patches_arr.shape}")
    print(patches_arr.dtype)
    
#     return patches_arr
    np.save(f"/home/sac/saptadeep/Hyperspectral_Denoising/Dataset/128_window_channels/noisy/128_{img_name}_noise",patches_arr)
    
  
# low_n_patch = extract_patches(low_noise_dt, 128, 'low')
# plt.figure()
# plt.imshow(img[:,:,3], cmap= "gray")
# plt.show()
# exit(0)
# f =extract_patches(high_noise_dt, 128, "high")
# plt.figure()
# plt.imshow(f[:,:,10], cmap = "gray")
# plt.show()

print(extract_patches(low_noise_dt, 128, "low")) #(128, 128, 100))


















