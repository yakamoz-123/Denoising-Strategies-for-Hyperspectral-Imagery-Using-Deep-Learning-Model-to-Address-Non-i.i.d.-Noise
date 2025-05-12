'''
Created on Apr 21, 2025

@author: sac
'''
import numpy as np
import matplotlib.pyplot as plt

# Dataset
dr = "/home/sac/saptadeep/Hyperspectral_Denoising/Dataset/Useful_bands/"
all_ = "useful_bands.npy"
blue_ = "useful_blue_ch.npy"
green_ = "useful_green_ch.npy"
red_ = "useful_red_ch.npy"
nir_ = "useful_nir.npy"
swir_ = "useful_swir.npy"
b_142_ = "useful_band_142.npy"

al = np.load(dr+all_)
blue = np.load(dr +blue_)
green = np.load(dr +green_)
red = np.load(dr +red_)
nir = np.load(dr +nir_)
swir = np.load(dr +swir_)
b_142 = np.load(dr+b_142_)
print(al.shape)
# exit(0)
# img_list = [al, blue, green, red, nir, swir]
# 
# 
# # print(blue.shape)
# # exit(0)
def extract_patches(img, patch_size):
    stride = patch_size
    h, w,b = img.shape
    patches = []
     
#     for b in range(img.shape[2]):
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
    patches_ar = np.array(patches)#.transpose(1,2,0)
    return patches_ar
def filter_black_pixel_patches(patches):
    """
    Filters out patches that contain even a single black (0) pixel.
    
    Args:
        patches (np.ndarray): Shape (H, W, N), where each patch is (H, W)
    
    Returns:
        np.ndarray: Filtered patches with shape (H, W, N_filtered)
    """
    H, W, N = patches.shape
    filtered = []
 
#     for i in range(N):
    patch = patches[:, :]
    if np.all(patch != 0):  # If no pixel is black
        filtered.append(patch)
 
    if len(filtered) > 0:
        return np.stack(filtered, axis=2)
    else:
        print("Array is empty")
         
# Extract 128x128 and 256x256 patches
# patches_128 = extract_patches(blue, 128)
# blue = filter_black_pixel_patches(patches_128)
def extract_filter_save(img, patch_size, dir):
#     for i in range(img.shape[2]):
    extract_128 = extract_patches(img, patch_size)
    filtered = filter_black_pixel_patches(extract_128)
    print(f"Shape of filtered array is: {filtered.shape}")
    return filtered
#     var_name = [name for name, val in globals().items() if val is img]
#     np.save(dir+f"128_{var_name[0] if var_name else 'unknown'}", filtered)
#     print(f"'{var_name[0] if var_name else 'unknown'}' array is saved at: {dir}")
         
dir = "/home/sac/saptadeep/Hyperspectral_Denoising/Dataset/128_window_channels/"
f = extract_filter_save(b_142 , 128, dir)
plt.figure()
plt.imshow(f, cmap="gray")
plt.show()

# f = np.load("/home/sac/saptadeep/Hyperspectral_Denoising/Dataset/128_window_channels/128_blue.npy")
# plt.figure()
# plt.imshow(f[:,:,0], cmap="gray")
# plt.show()


# print(f"Number of 256x256 patches: {patches_256.shape[2]}")