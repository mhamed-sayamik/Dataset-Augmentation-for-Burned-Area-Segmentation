import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from math import log10


def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0: return 100
    return 20 * log10(1.0 / torch.sqrt(mse).item())

# 1. CONFIGURATION & DIRECTORIES
# ---------------------------------------------------------
GPU_COUNT = torch.cuda.device_count()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = '/kaggle/working/data'
IMG_SIZE = 256
BATCH_SIZE = 16 * GPU_COUNT
LR = 0.0001
EPOCHS = 101

# Paths for saving outputs
CHECKPOINT_PATH = "fire_gan_checkpoint.pth"
VIS_DIR = "visualizations"
os.makedirs(VIS_DIR, exist_ok=True)

print(f"🚀 Dual-T4 Pipeline Active. Using {GPU_COUNT} GPUs.")

# 2. DATASET
# ---------------------------------------------------------
class FireDataset(Dataset):
    def __init__(self, root):
        self.root = root
        pre_dir = os.path.join(root, 'S2/pre')
        post_dir = os.path.join(root, 'S2/post')
        mask_dir = os.path.join(root, 'mask/post')
        common = set(os.listdir(pre_dir)) & set(os.listdir(post_dir)) & set(os.listdir(mask_dir))
        self.ids = sorted(list(common))

    def __len__(self): return len(self.ids)

    def read_tif(self, path, is_mask=False):
        with rasterio.open(path) as src:
            data = src.read([1, 2, 3] if src.count >= 3 and not is_mask else [1]).astype(np.float32)
            data = np.nan_to_num(data)
            if not is_mask:
                p2, p98 = np.percentile(data, (2, 98))
                if p98 > p2: data = (data - p2) / (p98 - p2)
            else:
                max_v = np.max(data)
                data = data / (max_v if max_v > 0 else 1.0)
            data = np.clip(data, 0, 1)
            tensor = torch.from_numpy(data).float()
            return F.interpolate(tensor.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode='bilinear').squeeze(0)

    def __getitem__(self, idx):
        name = self.ids[idx]
        pre = self.read_tif(os.path.join(self.root, 'S2/pre', name))
        post = self.read_tif(os.path.join(self.root, 'S2/post', name))
        mask = self.read_tif(os.path.join(self.root, 'mask/post', name), is_mask=True)
        return (pre * 2 - 1), (post * 2 - 1), (mask * 2 - 1)

# 3. MODELS
# ---------------------------------------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, True)
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh()
        )
    def forward(self, pre, mask): return self.dec(self.enc(torch.cat([pre, mask], dim=1)))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def sn_conv(in_c, out_c): return nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, 4, 2, 1))
        self.net = nn.Sequential(
            sn_conv(3, 64), nn.LeakyReLU(0.2, True),
            sn_conv(64, 128), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 1, 4, 1, 0)
        )
    def forward(self, x): return self.net(x)

class Refiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(True), nn.Conv2d(32, 3, 3, 1, 1), nn.Tanh())
    def forward(self, coarse, mask):
        gate = (mask + 1) / 2
        return coarse * (1 - gate) + self.net(coarse) * gate

# 4. INITIALIZATION & RESUME LOGIC
# ---------------------------------------------------------
dataset = FireDataset(DATA_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

gen = nn.DataParallel(Generator().to(DEVICE))
disc = nn.DataParallel(Discriminator().to(DEVICE))
refiner = nn.DataParallel(Refiner().to(DEVICE))

opt_G = optim.Adam(list(gen.parameters()) + list(refiner.parameters()), lr=LR, betas=(0.5, 0.999))
opt_D = optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))

scaler = torch.amp.GradScaler('cuda')
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()

start_epoch = 0
if os.path.exists(CHECKPOINT_PATH):
    print("📂 Loading checkpoint...")
    ckpt = torch.load(CHECKPOINT_PATH)
    gen.load_state_dict(ckpt['gen'])
    disc.load_state_dict(ckpt['disc'])
    refiner.load_state_dict(ckpt['refiner'])
    opt_G.load_state_dict(ckpt['opt_G'])
    opt_D.load_state_dict(ckpt['opt_D'])
    start_epoch = ckpt['epoch'] + 1
    print(f"▶️ Resuming from Epoch {start_epoch}")

# 5. TRAINING LOOP
# ---------------------------------------------------------
for epoch in range(start_epoch, EPOCHS):
    epoch_psnr = []
    
    for pre, post, mask in loader:
        pre, post, mask = pre.to(DEVICE).float(), post.to(DEVICE).float(), mask.to(DEVICE).float()
        
        # --- TRAIN DISCRIMINATOR ---
        opt_D.zero_grad()
        with torch.amp.autocast('cuda'):
            fake = refiner(gen(pre, mask), mask)
            d_real = disc(post)
            d_fake = disc(fake.detach())
            loss_D = criterion_GAN(d_real, torch.ones_like(d_real)) + \
                     criterion_GAN(d_fake, torch.zeros_like(d_fake))
        scaler.scale(loss_D).backward()
        scaler.step(opt_D)

        # --- TRAIN GENERATOR ---
        opt_G.zero_grad()
        with torch.amp.autocast('cuda'):
            d_fake_g = disc(fake)
            loss_G = criterion_GAN(d_fake_g, torch.ones_like(d_fake_g)) + (100 * criterion_L1(fake, post))
        scaler.scale(loss_G).backward()
        scaler.step(opt_G)
        scaler.update()
        
        # Batch Metric calculation
        with torch.no_grad():
            # Normalized [-1,1] to [0,1] for metric calculation
            psnr = calculate_psnr((fake + 1) / 2, (post + 1) / 2)
            epoch_psnr.append(psnr)

    # 6. SAVE & VISUALIZE
    # ---------------------------------------------------------
    avg_psnr = sum(epoch_psnr) / len(epoch_psnr)
    print(f"Epoch {epoch} | G: {loss_G.item():.4f} | D: {loss_D.item():.4f} | PSNR: {avg_psnr:.2f}dB")

    # Save Checkpoint
    torch.save({
        'epoch': epoch,
        'gen': gen.state_dict(),
        'disc': disc.state_dict(),
        'refiner': refiner.state_dict(),
        'opt_G': opt_G.state_dict(),
        'opt_D': opt_D.state_dict(),
    }, CHECKPOINT_PATH)

    if epoch % 5 == 0:
        with torch.no_grad():
            fig, ax = plt.subplots(1, 4, figsize=(20, 5))
            samples = [pre[0], mask[0], post[0], fake[0]]
            titles = ["Pre-fire (Input)", "Fire Mask (Input)", "Real Post (GT)", f"Synthetic (PSNR: {avg_psnr:.1f})"]
            for i, img in enumerate(samples):
                img = ((img.cpu().permute(1,2,0).numpy() + 1) / 2).clip(0, 1)
                ax[i].imshow(img.squeeze(), cmap='gray' if i==1 else None)
                ax[i].set_title(titles[i], fontsize=10)
                ax[i].axis('off')
            
            # Save visual to disk so it's not lost
            plt.savefig(f"{VIS_DIR}/epoch_{epoch}.png", bbox_inches='tight')
            plt.show()
            plt.close() # Free memory

print("🏁 Training complete. Check the 'visualizations' folder for your article figures.")
