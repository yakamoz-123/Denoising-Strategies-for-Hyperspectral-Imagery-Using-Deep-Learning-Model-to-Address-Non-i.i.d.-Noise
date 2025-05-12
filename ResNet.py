'''
Created on May 1, 2025

@author: sac
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

def check_tensor_validity(tensor, name="Tensor"):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise ValueError(f"{name} contains NaNs or Infs")

###########################################
# 1. Custom Dataset Loader for HSI Denoising
###########################################
class HSIDenoiseDataset(Dataset):
    def __init__(self, clean_path, noisy_path):
        if not os.path.exists(clean_path):
            raise FileNotFoundError(f"Clean file not found: {clean_path}")
        if not os.path.exists(noisy_path):
            raise FileNotFoundError(f"Noisy file not found: {noisy_path}")

#         self.clean = np.load(clean_path).astype(np.float32)[:3000]
#         self.noisy = np.load(noisy_path).astype(np.float32)[:3000]
        self.clean = np.load(clean_path).astype(np.float32)#[:,:,3000:]
        print(self.clean.shape)
        self.noisy = np.load(noisy_path).astype(np.float32)#[:,:,3000:]
        

        if self.clean.shape != self.noisy.shape:
            raise ValueError("Clean and noisy arrays must have the same shape")

        if self.clean.max() > 1.0:
            max_val = np.iinfo(np.uint16).max
            self.clean /= max_val
            self.noisy /= max_val

    def __len__(self):
        return self.clean.shape[-1]

    def __getitem__(self, idx):
        clean_band = self.clean[:, :, idx]
        noisy_band = self.noisy[:, :, idx]
        return (torch.tensor(noisy_band, dtype=torch.float32).unsqueeze(0),
                torch.tensor(clean_band, dtype=torch.float32).unsqueeze(0))

###########################################
# 2. ResNet Denoising Model Definition (Stronger Swish version)
###########################################
def swish(x):
    return x * torch.sigmoid(x)

class Swish(nn.Module):
    def forward(self, x):
        return swish(x)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            Swish(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        out = self.conv_block(x)
        check_tensor_validity(out, "ResBlock output")
        return x + out

class DenoiseResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=5, padding=2),
            Swish(),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            nn.Conv2d(128, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        out = self.net(x)
        check_tensor_validity(out, "Model output")
        return out

###########################################
# 3. Metrics: Manual PSNR, SSIM, SAM, ERGAS
###########################################
def compute_psnr(gt, pred):
    mse = np.mean((gt - pred) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))

def compute_ssim(img1, img2):
    C1, C2 = 0.01**2, 0.03**2
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1, sigma2 = img1.std(), img2.std()
    cov = ((img1 - mu1) * (img2 - mu2)).mean()
    return ((2*mu1*mu2 + C1)*(2*cov + C2)) / ((mu1**2 + mu2**2 + C1)*(sigma1**2 + sigma2**2 + C2))

def compute_metrics(clean, denoised):
    bands = clean.shape[2]
    psnr_val = ssim_val = 0.0
    for b in range(bands):
        c, d = clean[:, :, b], denoised[:, :, b]
        psnr_val += compute_psnr(c, d)
        ssim_val += compute_ssim(c, d)
        
    return psnr_val/bands, ssim_val/bands, 

def compute_ergas(clean, denoised):
    bands = clean.shape[2]
    err = 0.0
    for b in range(bands):
        mse = np.mean((clean[:, :, b] - denoised[:, :, b])**2)
        mean_ref = np.mean(clean[:, :, b])
        err += mse / (mean_ref**2 + 1e-8)
    return 100 * np.sqrt(err / bands)

###########################################
# 4. Training Loop
###########################################
# def train_model(clean_path, noisy_path, batch_size=16, lr=1e-3, epochs=5):
#     dataset = HSIDenoiseDataset(clean_path, noisy_path)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
#                         num_workers=4, pin_memory=True)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = DenoiseResNet().to(device)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
# 
#     for ep in range(1, epochs+1):
#         
#         model.train()
#         total_loss = 0.0
#         for noisy, clean in loader:
#             noisy, clean = noisy.to(device), clean.to(device)
#             check_tensor_validity(noisy, "Input noisy")
#             check_tensor_validity(clean, "Input clean")
#             optimizer.zero_grad()
#             out = model(noisy)
#             loss = criterion(out, clean)
#             check_tensor_validity(loss, "Loss")
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         
#         print(f"Epoch {ep}/{epochs} - Loss: {total_loss/len(loader):.6f}")
#         if ep%10 == 0:
#             torch.save(model.state_dict(), "/home/sac/saptadeep/LTC_noise_compo/New_Dataset/New/model_weights.pth")
#     return model, dataset

###########################################
# 5. Testing & Visualization
###########################################
def test_and_visualize(model, dataset, bands_to_show=[0,50,100,200,500]):
    device = next(model.parameters()).device
    model.load_state_dict(torch.load("/home/sac/saptadeep/LTC_noise_compo/New_Dataset/New/model_weights.pth"))
    model.eval()
    print("hiiiiiiiiiii")
#     h, w, b = dataset.clean.shape
#     denoised = np.zeros((h, w, b), dtype=np.float32)
    with torch.no_grad():
        
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            out = model(noisy)
            print(out.shape)
            #     Compute metrics
            
            clean_l = clean.cpu().squeeze().numpy()
            noisy_l = noisy.cpu().squeeze().numpy()
            pred_l = out.cpu().squeeze().numpy()
            print(clean_l.shape,"  ",pred_l.shape)
            
            p, s = compute_metrics(clean_l,pred_l)
            e = compute_ergas(clean_l,pred_l)
            print(f"MPSNR: {p:.4f}, MSSIM: {s:.4f}, ERGAS: {e:.4f}")
            

            
            # For vizualisation
            for batch_id in range(out.shape[0]):
                fig,ax = plt.subplots(1,3,sharex=True,sharey=True)
                pred_ = out[batch_id].cpu().squeeze().numpy()
                clean_ = clean[batch_id].cpu().squeeze().numpy()
                noisy_ = noisy[batch_id].cpu().squeeze().numpy()
                
                print(clean_.shape,"  ",pred_.shape)
                
#                 p, s = compute_metrics(clean_,pred_)
#                 e = compute_ergas(clean_,pred_)
#                 print(f"MPSNR: {p:.4f}, MSSIM: {s:.4f}, ERGAS: {e:.4f}")
                print(pred_.shape,"   ",clean_.shape)
                ax[0].imshow(pred_,cmap="gray")
                ax[1].imshow(noisy_,cmap="gray")
                ax[2].imshow(clean_,cmap="gray")
                plt.show()
                
            
            
    print("completed")
#         for i in range(b):
#             noisy, _ = dataset[i]
#             print(dataset.shape)
#             out = model(noisy)
# #             out = model(noisy.unsqueeze(0).to(device)).cpu().squeeze().numpy()
#             if out.shape != denoised[:, :, i].shape:
#                 print(f"Shape mismatch: out {out.shape} vs denoised[:, :, {i}] {denoised[:, :, i].shape}")
#                 continue
#             denoised[:, :, i] = out

# 
#     # Plot results
#     for b in bands_to_show:
#         if b < denoised.shape[2]:
#             plt.figure(figsize=(12,4))
#             for idx, (img, title) in enumerate([
#                 (dataset.clean[:,:,b], f"Clean (Band {b})"),
#                 (dataset.noisy[:,:,b], f"Noisy (Band {b})"),
#                 (denoised[:,:,b], f"Denoised (Band {b})"),
#                 (np.abs(dataset.clean[:,:,b]-denoised[:,:,b]), f"Diff (Band {b})")]):
#                 plt.subplot(1,4,idx+1)
#                 plt.title(title)
#                 plt.imshow(img, cmap='gray')
#                 plt.axis('off')
#             plt.tight_layout()
#             plt.show()
#             plt.savefig("sample_band_visualization.png")

# np.save('/home/sac/saptadeep/LTC_noise_compo/New_Dataset/New/Denoised_npy/sample_clean.npy', sample_clean_np)
# np.save('/home/sac/saptadeep/LTC_noise_compo/New_Dataset/New/Denoised_npy/sample_noisy.npy', sample_noisy_np)
# np.save('/home/sac/saptadeep/LTC_noise_compo/New_Dataset/New/Denoised_npy/sample_denoised.npy', sample_denoised_np)

###########################################
# 6. Main: Sample Run
###########################################
if __name__ == '__main__':
    CLEAN_FILE = '/home/sac/saptadeep/LTC_noise_compo/New_Dataset/New_clean/Filtered_new_test_clean.npy'
    NOISY_FILE = '/home/sac/saptadeep/LTC_noise_compo/New_Dataset/New_clean/new_test_noisy.npy'
    
    
    
    try:
#         model, ds = train_model(CLEAN_FILE, NOISY_FILE,
#                                 batch_size=8, lr=1e-3, epochs=1)
        print("data loading start ")
        dataset = HSIDenoiseDataset(CLEAN_FILE, NOISY_FILE)
        print("data loaded")
        loader = DataLoader(dataset, batch_size=8, shuffle=True,
                            num_workers=4, pin_memory=True)
        print("data loaded")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DenoiseResNet().to(device)
        test_and_visualize(model, loader)



    except Exception as ex:
        print(f"Error during processing: {ex}")
        
        
        
        
        
        
        
        
