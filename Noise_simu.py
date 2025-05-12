'''
Created on Apr 17, 2025

@author: sac
'''
import numpy as np
import pywt
import matplotlib.pyplot as plt
from skimage.exposure import histogram_matching,match_histograms
import os

c = np.load("/home/sac/saptadeep/Hyperspectral_Denoising/New_clean/new_test_noisy.npy")
print(c.shape)
exit(0)

# 1. Load images as uint16
# ## Clean
# directory = "/home/sac/saptadeep/Hyperspectral_Denoising/New_Dataset/All_noisy_patches/"
# filenames = [f for f in os.listdir(directory) if f.endswith(".npy")]
# # 
# # print(filenames)
# # exit(0)
# clean =[]
# for i in range(len(filenames)):
#     clean_f = directory+filenames[i]
#        
#     clean_load = np.load(clean_f)
#      
# #     print(f"shape of {filenames[i]}: {clean_load.shape}")
#     clean.append(clean_load)
#  
# clean = np.array(clean)
#   
# clean=clean.transpose(1,2,0) #(128,128,22922)
# print(f"Original Clean shape is : {clean.shape}")
#  
# clean_save = np.save("/home/sac/saptadeep/Hyperspectral_Denoising/New_Dataset/Combined_patches/All_new_noisy", clean)
# print(f"All patches saved {clean_save}")
# exit(0)


# exit(0)

## Noisy
noise_fp = "/home/sac/saptadeep/Hyperspectral_Denoising/Dataset/128_window_channels/noisy/"
# fn_noise = [f for f in os.listdir(noise_fp) if f.endswith(".npy")] #(all_clean_patches.npy is saved with 3d)
# 
# noisy_list = [] # list is in low, mid, high
# for i in range(len(fn_noise)):
#     noisy_f = noise_fp+fn_noise[i]
# #     noisy_list.append(noisy_f)
#     noisy_load = np.load(noisy_f)
#     noisy_tile = np.tile(noisy_load, (1, 1, 2300))[:, :, :22922]
#     noisy_list.append(noisy_tile)
# 
# # print(noisy_list)
# noisy = np.array(noisy_list)
# noisy_save = np.save(noise_fp+"all_noisy_patches", noisy)
# print(noisy_save)
# print(f"Original High Noisy shape is : {noisy[2,:,:,:].shape}")
# # exit(0)


# print(noise[2,:,:,:].shape)

# exit(0)



def wavelet_fusion(clean, noise, img_nm, show_band):
    print(f"Noise shape : {noise.shape}")
    recovered_list = []
    for i in range(clean.shape[2]):
    #         level = 1
        
        clean_b = clean[:,:,i]
        noise_b = noise[:,:,i]
        wavelet_type = "haar"
        LL,(LH,HL,HH) = pywt.dwt2(clean_b, wavelet_type)
        LL_n,(LH_n,HL_n,HH_n) = pywt.dwt2(noise_b, wavelet_type)
        
        noise_add_coeffs_ = (LL+5*LL_n,(LH+6*LH_n,HL+7*HL_n,HH+6*HH_n)) 
        
        
#         noise_add_coeffs_ = (LL+50*LL_n,(LH+60*LH_n,HL+60*HL_n,HH+60*HH_n)) 
        
        recov_im = pywt.idwt2(noise_add_coeffs_,wavelet = wavelet_type)
        
        
        
        # plt.hist(clean.ravel(),bins=1024,label='clean')
        # plt.hist(recov_im.ravel(),bins=1024,label='added')
        
        # clean_min_ = np.quantile(clean,0.01)
        # 
        # recov_im = (((recov_im - recov_im.min())/(recov_im.max()-recov_im.min()))*(clean.max()-clean_min_))+clean_min_
        
        recov_im = match_histograms(recov_im,clean_b)
        
#         fig,ax = plt.subplots(1,2,sharex=True,sharey=True)
#         ax[0].imshow(clean_b,cmap="gray")
#         ax[0].set_title("Clean")
#                  
#         ax[1].imshow(recov_im,cmap="gray")
#         ax[1].set_title("Recovered_image")
#                  
#         plt.show()
          
          
#         plt.hist(recov_im.ravel(),bins=1024,label='stretched ')
#         plt.legend()
#         plt.show()
#         exit(0)
        recovered_list.append(recov_im)
     
    reconstructed_image = np.stack(recovered_list, axis=-1)
#     print(f"Min val: {reconstructed_image.min()} and Max val: {reconstructed_image.max()}")
#     exit(0)
    print(f"Recovered Image Shape : {reconstructed_image.shape}")
#     return reconstructed_image

    save_dir = "/home/sac/saptadeep/Hyperspectral_Denoising/New_clean/"
      
    full_path = os.path.join(save_dir, img_nm)
      
    # Save the numpy array
    np.save(full_path, reconstructed_image)
    print(f"Saved file to {full_path}")
     
#     diff = clean - reconstructed_image
#         
#         
#         
#     fig,ax = plt.subplots(1,4,sharex=True,sharey=True)
#     ax[0].imshow(clean[:,:,show_band],cmap="gray")
#     ax[0].set_title("Clean")
#       
#     ax[1].imshow(reconstructed_image[:,:,show_band],cmap="gray")
#     ax[1].set_title("Recovered_image")
#       
#     ax[2].imshow(diff[:,:,show_band],cmap="gray")
#     ax[2].set_title("Difference")
#       
#     ax[3].imshow(noise[:,:,show_band],cmap="gray")
#     ax[3].set_title("Noise")    
#       
#     plt.suptitle(f"Band:{show_band}")         
#     plt.show()

# def contrast_enhancement(local_im):
#     non_zero_min = np.min(local_im[local_im > 0])
#     im_max_ = np.quantile(local_im[local_im > 0],0.995)
#     local_im = (local_im - non_zero_min) / (im_max_  - non_zero_min)
#     local_im[local_im < 0] = 0
#     local_im[local_im > 1] = 1
#     return local_im

def normalize(img):
    img_min = np.min(img[img!=0]) 
    img_max = np.quantile(img,0.99) 
    
    img = (img - img_min) / (img_max - img_min)
    img[img < 0] = 0
    img[img > 1] = 1
    return img

## Clean

directory = "/home/sac/saptadeep/Hyperspectral_Denoising/New_Dataset/norm_filter/"
filenames = sorted([f for f in os.listdir(directory) if f.endswith(('.npy'))])
clean = np.load("/home/sac/saptadeep/Hyperspectral_Denoising/New_clean/Filtered_new_test_clean.npy")#.transpose()
# clean = clean[:,:,:]#21084:
print(f"Clean shape : {clean.shape}")
# exit(0)
# Clean shape: (128,128,22922)

## Noisy
noise_fp = "/home/sac/saptadeep/Hyperspectral_Denoising/Dataset/128_window_channels/noisy/"
noisy = np.load(noise_fp+"all_noisy_patches.npy")
noisy_norm = normalize(noisy)

noisy_low = noisy_norm[0,:,:,:101]
noisy_mid = noisy_norm[1,:,:,:101]
noisy_high = noisy_norm[2,:,:,:clean.shape[2]]#21084:

img_nm = "new_test_noisy"
# print(f"max: {noisy_high.max()}, min: {noisy_high.min()}")

modif_img = wavelet_fusion(clean, noisy_high, img_nm, 5 )

# print(modif_img)

# plt.figure()
# plt.imshow(mod_img[:,:,100], cmap = "gray")
# plt.show()










