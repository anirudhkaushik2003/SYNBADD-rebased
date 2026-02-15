# =============================================================================
# File: train_SYNBADD_SVF.py
# Purpose: Training entrypoint for a specific experiment variant or legacy branch.
# Prerequisites: Relevant dataset metadata, supporting modules (datasets/generator/models/losses/utils), and configured /scratch output paths.
# Usually Run Next: Run corresponding test/evaluation scripts to generate outputs and comparison metrics.
# Model Components Used: See imports/CLI in this file for exact model stack; active latest flagship training is train_VAE_SYNBADD_svf.py.
# =============================================================================

import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import argparse
from matplotlib import pyplot as plt
from torch.autograd import Variable

from ViVit import ViViT
from datasets import  SVFRandomMaskedSequenceDataset
from generator import SVFContextDiffeoGenerator
from losses import compute_metrics, smooth_loss_l2
from utils import get_optimizers, get_data_loaders, str2bool, plot_slices, get_run_name
from test_SYNBADD_SVF import test, validate
from monai.inferers import sliding_window_inference

def spatial_patchify_tensor(x, patch_size, stride):
    """
    x: [B, T, C, H, W, D] tensor
    patch_size: (pH, pW, pD)
    stride: (sH, sW, sD)

    Returns:
        - List of patches [B, T, C, pH, pW, pD]
        - List of indices as tuples (b, z, y, x)
    """
    B, T, C, H, W, D = x.shape
    pH, pW, pD = patch_size
    sH, sW, sD = stride

    patches = []
    indices = []

    for b in range(B):
        for z in range(0, H - pH + 1, sH):
            for y in range(0, W - pW + 1, sW):
                for x_ in range(0, D - pD + 1, sD):
                    patch = x[b:b+1, :, :, z:z + pH, y:y + pW, x_:x_ + pD]  # [1, T, C, pH, pW, pD]
                    patches.append(patch)
                    indices.append((b, z, y, x_))

    return patches, indices

def make_3d_gaussian(pH, pW, pD, sigma=1.0, device="cpu"):
    """Create a normalized 3D Gaussian mask of size (1, 1, pH, pW, pD)"""
    z = torch.linspace(-1, 1, pH)
    y = torch.linspace(-1, 1, pW)
    x = torch.linspace(-1, 1, pD)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    kernel = kernel / kernel.max()
    return kernel.unsqueeze(0).unsqueeze(0).to(device)

def blend_and_stitch(patches, indices, output_shape, patch_size):
    """
    Stitch spatial patches (no temporal dim) into full-sized 3D volumes using weighted averaging.

    Args:
        patches (torch.Tensor): shape [N, C, pH, pW, pD] - processed patches
        indices (List[Tuple[int, int, int, int]]): list of tuples (batch_idx, z, y, x) for patch placement
        output_shape (Tuple[int, int, int, int, int]): (B, C, H, W, D) - shape of full output volume
        patch_size (Tuple[int, int, int]): (pH, pW, pD)

    Returns:
        torch.Tensor: Stitched output of shape [B, C, H, W, D]
    """
    B, C, H, W, D = output_shape
    pH, pW, pD = patch_size

    if isinstance(patches, list):
        patches = torch.stack(patches, dim=0)  # [N, C, pH, pW, pD]

    device = patches.device

    # Initialize output tensors
    stitched = torch.zeros((B, C, H, W, D), dtype=patches.dtype, device=device)
    weight_map = torch.zeros((B, 1, H, W, D), dtype=patches.dtype, device=device)

    # Create a single patch weighting mask (linear blend)
    blend_mask = make_3d_gaussian(pH, pW, pD, sigma=0.5, device=device)

    for i, (b_idx, z, y, x) in enumerate(indices):
        print(stitched.size(), patches[i].size())
        print(blend_mask.size())
        stitched[b_idx, :, z:z+pH, y:y+pW, x:x+pD] += patches[i] * blend_mask
        weight_map[b_idx, :, z:z+pH, y:y+pW, x:x+pD] += blend_mask

    # Avoid division by zero
    stitched = stitched / (weight_map + 1e-8)

    return stitched

random.seed(1337)

def train(opt):
   
    run_name = get_run_name(opt)

    if not opt.debug:
        wandb.init(project=f"SYNBADD_{opt.dataset}_5fcrossVal", name=run_name)
        wandb.config.update(vars(opt))

    image_size = (opt.img_height//2, opt.img_height//2, opt.img_height//2) 
    patch_size = (opt.img_height//opt.patch_size_factor, opt.img_height//opt.patch_size_factor, opt.img_height//opt.patch_size_factor)
    print(f"Image Size: {image_size}, Patch Size: {patch_size}\n")
    os.makedirs("/scratch/anirudhkaushik/SYNBADD/", exist_ok=True)

    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.float32)

    lambda_gp = 1
    accumulation_steps = opt.effective_batch_size // opt.batch_size

    nb_features = [
        [16, 32, 32, 32, 32],
        [32, 32, 32, 32, 32, 32, 16, 16]
    ]

    generator = SVFContextDiffeoGenerator(inshape=(opt.img_height, opt.img_height, opt.img_height),
                                c_dim=opt.c_dim,
                                use_probs=False,
                                int_downsize=2,
                                in_channels=3,
                                int_steps=7,
                                nb_unet_features=nb_features).to(device)
    vivit_model = ViViT(image_size, patch_size, opt.n_frames, num_channels=3).to(device)

    if opt.n_gpu > 1:
        generator = nn.DataParallel(generator, list(range(opt.n_gpu)))
        vivit_model = nn.DataParallel(vivit_model, list(range(opt.n_gpu)))

    optimizer_G, optimizer_ViViT, scheduler, scheduler_vivit, criterion_lddmm = get_optimizers(opt, generator, vivit_model, device=device)
    
    svf_file = "/scratch/anirudhkaushik/svf_metadata.csv"
    if opt.dataset == "ADNI":
        timepoints = ['sc', 'm06', 'm12', 'm18', 'm24', 'm36']
        csv_file = "/scratch/anirudhkaushik/ADNI1_Complete_DataSplits.csv"
    elif opt.dataset == "Toy":
        timepoints = ['sc', 'm06', 'm12', 'm18', 'm24', 'm30', 'm36']
        csv_file = "./metadata.csv"

    # Create train, validation, and test datasets
    train_dataset = SVFRandomMaskedSequenceDataset(csv_file, svf_file, subset='train', fold=opt.fold, aug_p=opt.aug_p, 
                                      timepoints=timepoints, target_timepoint=opt.target_timepoint, use_val_dataset=opt.use_val_dataset)
    val_dataset = SVFRandomMaskedSequenceDataset(csv_file, svf_file, subset='val', fold=opt.fold, aug_p=0.0, 
                                    timepoints=timepoints, target_timepoint=opt.target_timepoint, use_val_dataset=opt.use_val_dataset)
    test_dataset = SVFRandomMaskedSequenceDataset(csv_file, svf_file, subset='test', fold=opt.fold, aug_p=0.0, 
                                     timepoints=timepoints, target_timepoint=opt.target_timepoint, use_val_dataset=opt.use_val_dataset)
    
    # get dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_data_loaders(opt, train_dataset, val_dataset, test_dataset)

    # Log dataset sizes
    if not opt.debug:
        wandb.log({
            "Train Size": len(train_dataset),
            "Validation Size": len(val_dataset),
            "Test Size": len(test_dataset)
        })

    best_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = opt.patience

    for epoch in range(opt.n_epochs):
        # Training phase
        generator.train()
        vivit_model.train()
        optimizer_G.zero_grad()
        optimizer_ViViT.zero_grad()
        epoch_loss = 0
        align_losses, mean_jacobian_losses, std_jacobian_losses, mse_losses, psnr_losses, ssim_losses = [], [], [], [], [], []

        with tqdm(total=len(train_dataloader.dataset), desc=f"Epoch {epoch+1}/{opt.n_epochs}") as progress_bar:
            for i, batch in enumerate(train_dataloader):
                stacked_svf, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint = [batch[k].to(device) for k in ["stacked_svf", "latest_scan", "target_scan", "class_label", "reg_label", "target_timepoint", "latest_timepoint"]]
                x_real, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint = [Variable(v.type(torch.float32)).to(device) for v in [stacked_svf, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint]]
                target_labels = torch.cat([latest_scan_timepoint, class_labels, reg_labels, target_timepoint], dim=1)
                
                if opt.use_past_data:
                    patch_size = (64, 64, 64)
                    stride = (32, 32, 32)
                    patches, indices = spatial_patchify_tensor(x_real, patch_size, stride)
                    contextualized_patches = [vivit_model(patch) for patch in patches]
                    output_shape = (x_real.shape[0], x_real.shape[2], x_real.shape[3], x_real.shape[4], x_real.shape[5]) # B, T, C, H, W, D = x_real.shape -> B, C, H, W, D 
                    contextualized_scan = blend_and_stitch(contextualized_patches, indices, output_shape, patch_size)
                    contextualized_scan = contextualized_scan.view(x_real.shape)
                else:
                    contextualized_scan = latest_scan

                y_synthetic, synthetic_field = generator(latest_scan, contextualized_scan, target_labels)
                align_loss = smooth_loss_l2(y_synthetic, target_scan)
                jacobian_loss = criterion_lddmm(target_scan, y_synthetic)

                total_loss = align_loss + opt.jacobian_coeff*jacobian_loss
                total_loss.backward()

                mse_loss, psnr_value, ssim_value, mean_jacobian, std_jacobian, jacobian_det = compute_metrics(target_scan, y_synthetic, criterion_lddmm, device)

                align_losses.append(align_loss.item())
                mean_jacobian_losses.append(mean_jacobian)
                std_jacobian_losses.append(std_jacobian)
                mse_losses.append(mse_loss)
                psnr_losses.append(psnr_value)
                ssim_losses.append(ssim_value)

                epoch_loss += total_loss.item()

                if (i + 1) % accumulation_steps == 0:
                    optimizer_G.step()
                    optimizer_ViViT.step()
                    optimizer_G.zero_grad()
                    optimizer_ViViT.zero_grad()

                progress_bar.update(x_real.size(0))
                progress_bar.set_postfix({
                    "Smooth l12 Loss": align_loss.item(),
                    "LDDMM Loss": jacobian_loss.item(),
                    "Mse": mse_loss,
                    "PSNR": psnr_value,
                    "SSIM": ssim_value,
                    "Total Loss": total_loss.item(),
                    "Mean Jac": mean_jacobian,
                    "Std Jac": std_jacobian,
                })

        epoch_loss /= len(train_dataloader)
        avg_align_loss = np.mean(align_losses)
        avg_mean_jacobian = np.mean(mean_jacobian_losses)
        avg_std_jacobian = np.mean(std_jacobian_losses)
        avg_psnr = np.mean(psnr_losses)
        avg_ssim = np.mean(ssim_losses)
        avg_mse = np.mean(mse_losses)
        
        # Validation phase
        if opt.use_val_dataset:
            val_loss = validate(generator, vivit_model, test_dataloader, criterion_lddmm, device, opt)
        else:
            val_loss = epoch_loss
        
        scheduler.step(val_loss)  # Using validation loss for scheduler
        scheduler_vivit.step(val_loss)
        
        if not opt.debug:
            wandb.log({
                "Epoch": epoch,
                "Train Loss": epoch_loss, 
                "Train Smooth L12 Loss": avg_align_loss,
                "Train Validation Loss": val_loss,
                "Train Learning Rate": optimizer_G.param_groups[0]['lr'],
                "Train Mean Jac": avg_mean_jacobian,
                "Train Std Jac": avg_std_jacobian,
                "Train PSNR": avg_psnr,
                "Train SSIM": avg_ssim,
                "Train MSE": avg_mse
            })

        print(f"Epoch {epoch+1}/{opt.n_epochs}: Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            group_map = {0.0: 'CN', 1.0: 'MCI', 2.0: 'AD'}
            best_loss = val_loss
            patience_counter = 0
            
            # Save best model
            model_save_path = f"/scratch/anirudhkaushik/SYNBADD/{run_name}.pth"
            torch.save(generator.state_dict(), model_save_path)
            torch.save(vivit_model.state_dict(), f"/scratch/anirudhkaushik/SYNBADD/{run_name}_vivit.pth")
            
            print(f"Saved best model at epoch {epoch+1} with validation loss: {val_loss:.4f}")

            if not opt.debug:
                # Visualize best validation result
                with torch.no_grad():
                    val_batch = next(iter(test_dataloader))
                    stacked_svf, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint = [val_batch[k].to(device) for k in ["stacked_svf", "latest_scan", "target_scan", "class_label", "reg_label", "target_timepoint", "latest_timepoint"]]
                    x_real, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint = [Variable(v.type(torch.float32)).to(device) for v in [stacked_svf, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint]]
                    target_labels = torch.cat([latest_scan_timepoint, class_labels, reg_labels, target_timepoint], dim=1)

                    if opt.use_past_data:
                        contextualized_scan = vivit_model(stacked_svf)
                    else:
                        contextualized_scan = latest_scan
                        
                    y_synthetic, synthetic_field = generator(latest_scan, contextualized_scan, target_labels)

                    mse_loss, psnr_value, ssim_value, mean_jacobian, std_jacobian, jacobian_det = compute_metrics(target_scan, y_synthetic, criterion_lddmm, device)
                    
                    loss_map = {"mse": mse_loss,"psnr": psnr_value,"ssim": ssim_value,"mean jacobian": mean_jacobian, "std jacobian": std_jacobian}
                    
                    fig = plot_slices(latest_scan, target_scan, y_synthetic, jacobian_det, loss_map,
                                     reg_labels[0].item(), 
                                     group_map[class_labels[0].item()], target_timepoint[0].item(), latest_scan_timepoint[0].item())
                    wandb.log({"Best Validation Result": wandb.Image(fig)})
                    plt.close(fig)
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Log best validation loss
    if not opt.debug:
        wandb.log({"Best Validation Loss": best_loss})

    # Load best model for testing
    print("Loading best model for testing...")
    generator.load_state_dict(torch.load(f"/scratch/anirudhkaushik/SYNBADD/{run_name}.pth"))
    vivit_model.load_state_dict(torch.load(f"/scratch/anirudhkaushik/SYNBADD/{run_name}_vivit.pth"))
    
    # Test phase
    test_loss = test(generator, vivit_model, test_dataloader, criterion_lddmm, device, opt, run_name)
    print(f"Final test loss: {test_loss:.4f}")
    
    if not opt.debug:
        wandb.log({"Final Test Loss": test_loss})
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=10, help="size of the batches")
    parser.add_argument("--fold", type=int, default=1, help="fold number")
    parser.add_argument("--effective_batch_size", type=int, default=20, help="effective batch size after gradient accumulation")
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--aug_p", type=float, default=0.8, help="amount of augmentation to use")
    parser.add_argument("--n_cpu", type=int, default=10, help="number of cpu threads to use during batch generation")
    parser.add_argument("--n_gpu", type=int, default=4, help="number of gpus on machine to use")
    parser.add_argument("--img_height", type=int, default=128, help="size of image height")
    parser.add_argument("--c_dim", type=int, default=4, help="number of dimensions for condition (target_labels)")
    parser.add_argument("--n_critic", type=int, default=1, help="number of training iterations for discriminator")
    parser.add_argument("--n_frames", type=int, default=6, help="number of frames for ViVit")
    parser.add_argument("--jacobian_coeff", type=float, default=1.0, help="weight for lddmm loss")
    parser.add_argument("--sim_weight", type=float, default=0.5, help="weight for similarity loss")
    parser.add_argument("--use_past_data", type=str2bool, nargs='?', const=True, default=True, help="use past data for ViVit")
    parser.add_argument("--use_future_data", type=str2bool, nargs='?', const=True, default=False, help="use future data for ViVit")
    parser.add_argument("--dataset", type=str, default="ADNI", help="which dataset to use for training", choices=["ADNI", "Toy"])
    parser.add_argument("--target_timepoint", type=str, default="m18", help="timepoint to predict")
    parser.add_argument("--patch_size_factor", type=int, default=4, help="patch size factor for ViVit")
    parser.add_argument("--use_deformation_field", type=str2bool, nargs='?', const=True, default=False, help="use deformation field for alignment loss")
    parser.add_argument("--patience", type=int, default=20, help="early stopping patience")
    parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=False, help="run code in debug mode")
    parser.add_argument("--specials", type=str, default="DAFT", help="special run parameters")
    parser.add_argument("--use_val_dataset", type=str2bool, nargs='?', const=True, default=False, help="use validation dataset for training")
    parser.add_argument("--jacobian_factor", type=float, default=0.1, help="weight for jacobian regularization")

    opt = parser.parse_args()
    train(opt)