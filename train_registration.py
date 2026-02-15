# =============================================================================
# File: train_registration.py
# Purpose: Training entrypoint for a specific experiment variant or legacy branch.
# Prerequisites: Relevant dataset metadata, supporting modules (datasets/generator/models/losses/utils), and configured /scratch output paths.
# Usually Run Next: Run corresponding test/evaluation scripts to generate outputs and comparison metrics.
# Model Components Used: See imports/CLI in this file for exact model stack; active latest flagship training is train_VAE_SYNBADD_svf.py.
# =============================================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from voxelmorph.torch import losses
from voxelmorph.torch.networks import VxmDense
from datasets import ADNIVxmTraining
import matplotlib.pyplot as plt

def jacobian_determinant_torch(flow):
    """
    Compute the Jacobian determinant of a 3D flow field (batched).
    Args:
        flow: tensor of shape (B, 3, H, W, D) — displacement field
    Returns:
        jacobian_dets: tensor of shape (B, H, W, D)
    """
    B, C, H, W, D = flow.shape
    dy = flow[:, :, 1:, :-1, :-1] - flow[:, :, :-1, :-1, :-1]
    dx = flow[:, :, :-1, 1:, :-1] - flow[:, :, :-1, :-1, :-1]
    dz = flow[:, :, :-1, :-1, 1:] - flow[:, :, :-1, :-1, :-1]

    J = torch.stack([dx, dy, dz], dim=-1)  # (B, 3, H-1, W-1, D-1, 3)
    J = J.permute(0, 2, 3, 4, 1, 5)         # (B, H-1, W-1, D-1, 3, 3)

    # Compute determinant of the 3×3 Jacobian matrix
    jacobian_det = (
        J[..., 0, 0] * (J[..., 1, 1] * J[..., 2, 2] - J[..., 1, 2] * J[..., 2, 1]) -
        J[..., 0, 1] * (J[..., 1, 0] * J[..., 2, 2] - J[..., 1, 2] * J[..., 2, 0]) +
        J[..., 0, 2] * (J[..., 1, 0] * J[..., 2, 1] - J[..., 1, 1] * J[..., 2, 0])
    )
    return jacobian_det  # (B, H-1, W-1, D-1)

# --- Config ---
csv_file = "/scratch/anirudhkaushik/ADNI1_Complete_DataSplits.csv"
batch_size = 16
lr = 1e-4
epochs = 100
lambda_smooth = 0.01
fold = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
save_dir = "/scratch/anirudhkaushik/vxm_checkpoints"
os.makedirs(save_dir, exist_ok=True)

# --- W&B Init ---
wandb.init(
    project="voxelmorph-adni_128",
    name="vxm-longitudinal",
    config={
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "lambda_jacobian": lambda_smooth,
        "image_size": (128, 128, 128),
        "fold": fold
    }
)

# --- Model + Optimizer ---
vxm_model = VxmDense(
    inshape=(128, 128, 128),
    nb_unet_features=[[16, 32, 32, 32], [32, 32, 32, 32]],
    int_downsize=1,
)
vxm_model = nn.DataParallel(vxm_model).to(device)
optimizer = torch.optim.Adam(vxm_model.parameters(), lr=lr)

# --- Dataset + Loader ---
timepoints = ['sc', 'm06', 'm12', 'm18', 'm24', 'm36']
dataset = ADNIVxmTraining(csv_file, subset='train', fold=1, aug_p=0, 
                                      timepoints=timepoints, target_timepoint="m18", use_val_dataset=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# --- Training Loop ---
def train_vxm(dataloader, model, optimizer, epochs=100, device="cuda"):
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_ncc_loss = 0.0
        total_jac_loss = 0.0
        total_valid_targets = 0

        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch in dataloader:
                latest = batch["latest_scan"].to(device)         # [B, 1, H, W, D]
                stacked = batch["stacked_scan"].to(device)       # [B, T, 1, H, W, D]
                masks = batch["mask"].to(device)                 # [B, T]

                B, T, C, H, W, D = stacked.shape
                targets = stacked[:, 1:]                         # [B, T-1, 1, H, W, D]
                masks = masks[:, 1:]                             # [B, T-1]
                n_targets = targets.shape[1]

                optimizer.zero_grad()
                batch_ncc = 0.0
                batch_jac = 0.0
                valid_count = 0

                for t in range(n_targets):
                    target = targets[:, t]
                    valid_mask = masks[:, t].view(-1, 1, 1, 1, 1)
                    valid_indices = (valid_mask.squeeze() > 0).nonzero(as_tuple=True)[0]

                    if len(valid_indices) == 0:
                        continue

                    valid_latest = latest[valid_indices]
                    valid_target = target[valid_indices]

                    warped, flow = model(valid_latest, valid_target)
                    # warped = model.module.transformer(valid_latest, flow)

                    ncc = F.mse_loss(warped, valid_target)
                    jac_det = jacobian_determinant_torch(flow)  # shape: (B, H-1, W-1, D-1)
                    jac = torch.mean(torch.abs(jac_det - 1.0))  # scalar loss

                    batch_ncc += ncc
                    batch_jac += jac
                    valid_count += 1

                if valid_count > 0:
                    loss = batch_ncc / valid_count + lambda_smooth * batch_jac / valid_count
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total_ncc_loss += batch_ncc.item() / valid_count
                    total_jac_loss += batch_jac.item() / valid_count
                    total_valid_targets += valid_count

                pbar.set_postfix({
                    "Loss": total_loss / (total_valid_targets or 1),
                    "NCC": total_ncc_loss / (total_valid_targets or 1),
                    "Jac": total_jac_loss / (total_valid_targets or 1)
                })
                pbar.update(1)

        avg_loss = total_loss / (total_valid_targets or 1)
        avg_ncc = total_ncc_loss / (total_valid_targets or 1)
        avg_jac = total_jac_loss / (total_valid_targets or 1)

        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | NCC: {avg_ncc:.4f} | Jacobian: {avg_jac:.4f}")
        wandb.log({
            "Epoch": epoch + 1,
            "Total Loss": avg_loss,
            "NCC Loss": avg_ncc,
            "Jacobian Loss": avg_jac
        })

         # --- Optional Visual Logging ---
        model.eval()
        with torch.no_grad():
            # Get a sample batch
            sample_batch = next(iter(dataloader))
            latest_full = sample_batch["latest_scan"].to(device)     # [B, 1, H, W, D]
            stacked_sample = sample_batch["stacked_scan"].to(device)
            mask_sample = sample_batch["mask"].to(device)

            # Exclude baseline, loop to find first valid target
            targets_sample = stacked_sample[:, 1:]
            mask_targets = mask_sample[:, 1:]

             # Find a valid sample and timepoint
            valid_sample_idx = None
            valid_timepoint = None
            
            for sample_idx in range(mask_targets.shape[0]):
                for t in range(mask_targets.shape[1]):
                    if mask_targets[sample_idx, t] == 1:
                        valid_sample_idx = sample_idx
                        valid_timepoint = t
                        break
                if valid_sample_idx is not None:
                    break
            
            if valid_sample_idx is None:
                print("⚠️ No valid target scan found for visual logging.")
                continue
            
            # Extract only the valid sample for both latest and target
            latest_sample = latest_full[valid_sample_idx:valid_sample_idx+1]         # [1, 1, H, W, D]
            target_sample = targets_sample[valid_sample_idx, valid_timepoint].unsqueeze(0)  # [1, 1, H, W, D]
            
            warped, flow = model(latest_sample, target_sample)
            warped_sample = warped[0]

            middle_slice = latest_sample.shape[-1] // 2

            def slice_image(img):
                img = img.detach().cpu()
                if img.dim() == 5:
                    return img[0, 0, :, :, middle_slice].numpy()
                elif img.dim() == 4:
                    return img[0, :, :, middle_slice].numpy()
                else:
                    raise ValueError("Unexpected tensor shape")

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(slice_image(latest_sample), cmap='gray')
            axes[0].set_title("Latest (Moving)")
            axes[1].imshow(slice_image(target_sample), cmap='gray')
            axes[1].set_title("Target (Fixed)")
            axes[2].imshow(slice_image(warped_sample), cmap='gray')
            axes[2].set_title("Warped (Registered)")

            for ax in axes:
                ax.axis("off")

            wandb.log({f"Visual Epoch {epoch+1}": wandb.Image(fig)})
            plt.close(fig)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_vxm_model_128.pth"))
            wandb.run.summary["best_loss"] = best_loss

# --- Run Training ---
if __name__ == "__main__":
    train_vxm(dataloader, vxm_model, optimizer, epochs=epochs, device=device)
