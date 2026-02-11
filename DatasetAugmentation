import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import shutil

# --- 1. CONFIG ---
CHECKPOINT_PATH = "/kaggle/working/fire_gan_checkpoint.pth"
SOURCE_ROOT = '/kaggle/input/segmentation-of-burned-areas/FireSR/dataset'
WORKING_DIR = '/kaggle/working/paper_experiment'
AUG_FACTOR = 3  
START_INDEX = 200
NUM_TO_EXTRACT = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(WORKING_DIR): shutil.rmtree(WORKING_DIR)
for d in ['synthetic_raw', 'visual_grids']: os.makedirs(os.path.join(WORKING_DIR, d), exist_ok=True)

# --- 2. MODELS ---
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Flatten() 
        )
        
        self.fc = nn.Linear(128 * 64 * 64, 256 * 64 * 64)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh()
        )
    def forward(self, pre, mask):
        x = torch.cat([pre, mask], dim=1)
        x = self.enc(x)
        x = self.fc(x).view(-1, 256, 64, 64)
        return self.dec(x)

class Refiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(True), nn.Conv2d(32, 3, 3, 1, 1), nn.Tanh())
    def forward(self, coarse, mask):
        gate = (mask + 1) / 2
        return torch.clamp(coarse * (1 - gate) + self.net(coarse) * gate, -1, 1)

# --- 3. NORMALIZATION ---
def to_vis(tensor):
    """Safely converts model output [-1, 1] to [0, 1] for saving"""
    img = tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    img = (img + 1.0) / 2.0
    return np.clip(img, 0, 1)

def read_tif_strict(path, is_mask=False):
    with rasterio.open(path) as src:
        data = src.read([1, 2, 3] if src.count >= 3 and not is_mask else [1]).astype(np.float32)
        if not is_mask:
            # Robust scaling to avoid extreme values
            p2, p98 = np.percentile(data, (2, 98))
            data = np.clip((data - p2) / (p98 - p2 + 1e-6), 0, 1)
        else:
            data = (data > 0).astype(np.float32)
        
        t = torch.from_numpy(data).float().unsqueeze(0)
        t = F.interpolate(t, size=(256, 256), mode='bilinear')
        return (t * 2.0 - 1.0).to(DEVICE)

# --- 4. RUN ---
gen = nn.DataParallel(Generator().to(DEVICE))
refiner = nn.DataParallel(Refiner().to(DEVICE))
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
gen.load_state_dict(ckpt['gen']); refiner.load_state_dict(ckpt['refiner'])
gen.eval(); refiner.eval()

all_files = sorted(list(set(os.listdir(os.path.join(SOURCE_ROOT, 'S2/pre'))) & 
                        set(os.listdir(os.path.join(SOURCE_ROOT, 'mask/post')))))[START_INDEX:START_INDEX+NUM_TO_EXTRACT]

print(f"Generating 150 images...")
for filename in all_files:
    pre_t = read_tif_strict(os.path.join(SOURCE_ROOT, 'S2/pre', filename))
    mask_t = read_tif_strict(os.path.join(SOURCE_ROOT, 'mask/post', filename), is_mask=True)
    real_post_t = read_tif_strict(os.path.join(SOURCE_ROOT, 'S2/post', filename))

    for i in range(AUG_FACTOR):
        with torch.no_grad():
            # Only add jitter to the PRE-fire image, NOT the mask
            # Reduced jitter magnitude to avoid "White out"
            jitter = torch.clamp(pre_t + (torch.randn_like(pre_t) * 0.01), -1, 1)
            
            # Forward pass
            synth = refiner(gen(jitter, mask_t), mask_t)
            
            # Save Raw
            res_img = to_vis(synth)
            plt.imsave(os.path.join(WORKING_DIR, 'synthetic_raw', f"{filename[:-4]}_v{i}.png"), res_img)
            
    # Save 1 Grid per set to verify colors
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(to_vis(pre_t)); ax[0].set_title("Pre")
    ax[1].imshow(to_vis(mask_t), cmap='gray'); ax[1].set_title("Mask")
    ax[2].imshow(to_vis(real_post_t)); ax[2].set_title("Real Post")
    ax[3].imshow(res_img); ax[3].set_title("Synthetic (v2)")
    for a in ax: a.axis('off')
    plt.savefig(os.path.join(WORKING_DIR, 'visual_grids', f"{filename[:-4]}_check.png"))
    plt.close()

shutil.make_archive('/kaggle/working/Augmented_Final_Fix', 'zip', WORKING_DIR)
print("Done. Check Augmented_Final_Fix.zip")
