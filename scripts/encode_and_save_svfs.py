# =============================================================================
# File: scripts/encode_and_save_svfs.py
# Purpose: Support or experiment file within SYNBADD repository.
# Prerequisites: Depends on imports and paths defined in this file.
# Usually Run Next: Typically called by a train/test/eval script in this repo.
# Model Components Used: See imports in this file for exact model/loss dependencies.
# =============================================================================

import os
import torch
from tqdm import tqdm
from models import VAE3D_UNetSkip
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

# --- Config ---
input_dir = "/scratch/anirudhkaushik/svf_outputs"
output_dir = "/scratch/anirudhkaushik/svf_encoded"
vae_weights = "/scratch/anirudhkaushik/SYNBADD/SVF Encoder VAE Training/best_model.pth"
batch_size = 8
num_workers = 4
device = torch.device("cuda:0")

os.makedirs(output_dir, exist_ok=True)

# --- Dataset ---
class SVFDataset(Dataset):
    def __init__(self, root_dir):
        self.files = [f for f in os.listdir(root_dir) if f.endswith(".pt")]
        self.files.sort()
        self.root_dir = root_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.files[idx])
        svf = torch.load(path).float()  # [3, 128, 128, 128]
        return svf, self.files[idx]

dataset = SVFDataset(input_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# --- Load VAE in DataParallel ---
vae = VAE3D_UNetSkip().to(device)
state_dict = torch.load(vae_weights, map_location=device)
new_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
vae.load_state_dict(new_state_dict)
vae = torch.nn.DataParallel(vae, device_ids=list(range(torch.cuda.device_count())))
vae.eval()

# --- Encode in batches ---
# with torch.no_grad():
#     for svf_batch, filenames in tqdm(dataloader, desc="Encoding SVFs"):
#         svf_batch = svf_batch.to(device)  # [B, 3, 128, 128, 128]
#         mu_batch, _, skips_batch = vae.module.encode(svf_batch)  # [B, 512], [B, skip_shapes...]

#         for i, fname in enumerate(filenames):
#             mu = mu_batch[i].cpu()
#             skips = [s[i].cpu() for s in skips_batch]
#             encoded = {"mu": mu, "skips": skips}
#             torch.save(encoded, os.path.join(output_dir, fname))

zero_svf = torch.zeros(1, 3, 128, 128, 128).to(device)
with torch.no_grad():
    mu, _, skips = vae.module.encode(zero_svf)

# save mu and skips
mu = mu.cpu()
skips = [s.cpu() for s in skips]
encoded = {"mu": mu, "skips": skips}
torch.save(encoded, os.path.join(output_dir, "zero_svf.pt"))