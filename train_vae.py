# =============================================================================
# File: train_vae.py
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
from datasets import  VAEdeformationDataset
from generator import SVFTransformerDiffeoGenerator
from utils import get_data_loaders
from test_resnet_SYNBADD_svf import test, validate
from utils import str2bool
from monai.inferers import sliding_window_inference
#import 3d resnet
from losses import jacobian_determinant_torch
# torch dataloader
from torch.utils.data import DataLoader
from models import VAE3D_UNetSkip



def vae_loss(recon, x, mu, logvar, epe_weight=0.05, angular_weight=0.05):
    """
    Compute VAE loss with additional channel-wise EPE and angular error terms
    Args:
        recon: Reconstructed SVF tensor [B, 3, H, W, D]
        x: Input SVF tensor [B, 3, H, W, D]
        mu: Mean vector from encoder
        logvar: Log variance vector from encoder
        epe_weight: Weight for EPE loss term
        angular_weight: Weight for angular error loss term
    Returns:
        Total loss
    """
    # Original VAE loss components
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + kld
    
    # Add channel-wise EPE loss if weight > 0
    if epe_weight > 0.0:
        # Calculate per-channel EPE (x, y, z)
        epe_x = F.mse_loss(recon[:, 0], x[:, 0], reduction='mean').sqrt()
        epe_y = F.mse_loss(recon[:, 1], x[:, 1], reduction='mean').sqrt()
        epe_z = F.mse_loss(recon[:, 2], x[:, 2], reduction='mean').sqrt()
        
        # Use the average of channel-wise EPEs as the loss component
        epe_loss = (epe_x + epe_y + epe_z) / 3.0
        total_loss += epe_weight * epe_loss
    
    # Add angular error loss if weight > 0
    if angular_weight > 0.0:
        angular_loss = compute_angular_error_loss(recon, x)
        total_loss += angular_weight * angular_loss
    
    return total_loss


def compute_epe(predicted, target):
    """
    Compute End Point Error (EPE) for each channel
    Args:
        predicted: Predicted SVF tensor [B, 3, H, W, D]
        target: Target SVF tensor [B, 3, H, W, D]
    Returns:
        Dictionary of EPE values for each channel
    """
    # Calculate per-channel EPE (x, y, z)
    epe_x = F.mse_loss(predicted[:, 0], target[:, 0], reduction='mean').sqrt()
    epe_y = F.mse_loss(predicted[:, 1], target[:, 1], reduction='mean').sqrt()
    epe_z = F.mse_loss(predicted[:, 2], target[:, 2], reduction='mean').sqrt()
    
    # Calculate overall EPE
    epe_total = F.mse_loss(predicted, target, reduction='mean').sqrt()
    
    return {
        'epe_x': epe_x.item(),
        'epe_y': epe_y.item(),
        'epe_z': epe_z.item(),
        'epe_total': epe_total.item()
    }


def compute_angular_error(predicted, target):
    """
    Compute angular error between predicted and target SVF vectors
    Args:
        predicted: Predicted SVF tensor [B, 3, H, W, D]
        target: Target SVF tensor [B, 3, H, W, D]
    Returns:
        Mean angular error in degrees
    """
    # Reshape to [B, 3, H*W*D]
    B, C, H, W, D = predicted.shape
    pred_flat = predicted.reshape(B, C, -1)
    target_flat = target.reshape(B, C, -1)
    
    # Compute dot product between predicted and target vectors
    dot_product = torch.sum(pred_flat * target_flat, dim=1)  # [B, H*W*D]
    
    # Compute magnitudes
    pred_mag = torch.norm(pred_flat, dim=1)  # [B, H*W*D]
    target_mag = torch.norm(target_flat, dim=1)  # [B, H*W*D]
    
    # Avoid division by zero
    pred_mag = torch.clamp(pred_mag, min=1e-8)
    target_mag = torch.clamp(target_mag, min=1e-8)
    
    # Compute cosine of angle
    cos_angle = dot_product / (pred_mag * target_mag)
    cos_angle = torch.clamp(cos_angle, min=-1.0, max=1.0)
    
    # Convert to angles in degrees
    angles = torch.acos(cos_angle) * 180.0 / np.pi
    
    # Return mean angular error
    return angles.mean().item()


def compute_angular_error_loss(predicted, target):
    """
    Compute angular error loss between predicted and target SVF vectors
    This version returns a tensor for backpropagation, unlike compute_angular_error
    Args:
        predicted: Predicted SVF tensor [B, 3, H, W, D]
        target: Target SVF tensor [B, 3, H, W, D]
    Returns:
        Angular error loss tensor
    """
    # Reshape to [B, 3, H*W*D]
    B, C, H, W, D = predicted.shape
    pred_flat = predicted.reshape(B, C, -1)
    target_flat = target.reshape(B, C, -1)
    
    # Compute dot product between predicted and target vectors
    dot_product = torch.sum(pred_flat * target_flat, dim=1)  # [B, H*W*D]
    
    # Compute magnitudes
    pred_mag = torch.norm(pred_flat, dim=1)  # [B, H*W*D]
    target_mag = torch.norm(target_flat, dim=1)  # [B, H*W*D]
    
    # Avoid division by zero
    pred_mag = torch.clamp(pred_mag, min=1e-8)
    target_mag = torch.clamp(target_mag, min=1e-8)
    
    # Compute cosine of angle
    cos_angle = dot_product / (pred_mag * target_mag)
    cos_angle = torch.clamp(cos_angle, min=-1.0, max=1.0)
    
    # Use 1 - cos(angle) as the loss (smaller angles have lower loss)
    # This avoids numerical issues with acos near +/-1
    angular_loss = (1.0 - cos_angle).mean()
    
    return angular_loss


def plot_sample(input_sample, recon_sample, saved):
    input_sample = jacobian_determinant_torch(input_sample)
    recon_sample = jacobian_determinant_torch(recon_sample)

    input_sample = input_sample[0].cpu().detach().numpy()
    recon_sample = recon_sample[0].cpu().detach().numpy()

    mid_slice = input_sample.shape[2] // 2

    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    axes[0].imshow(input_sample[:, :, mid_slice], cmap='jet')
    axes[0].set_title("Input SVF (Jacobian Det)")
    axes[1].imshow(recon_sample[:, :, mid_slice], cmap='jet')
    axes[1].set_title("Reconstructed SVF (Jacobian Det)")
    axes[0].axis('off')
    axes[1].axis('off')

    img_name = f"SVF_Reconstruction_{saved}"
    return wandb.Image(fig, caption=img_name)


def plot_channel_errors(input_sample, recon_sample, idx=0):
    """
    Plot per-channel error maps
    """
    # Get the middle slice for visualization
    mid_slice = input_sample.shape[3] // 2
    
    # Calculate error for each channel
    error_x = torch.abs(input_sample[idx, 0, :, :, mid_slice] - recon_sample[idx, 0, :, :, mid_slice])
    error_y = torch.abs(input_sample[idx, 1, :, :, mid_slice] - recon_sample[idx, 1, :, :, mid_slice])
    error_z = torch.abs(input_sample[idx, 2, :, :, mid_slice] - recon_sample[idx, 2, :, :, mid_slice])
    
    # Convert to numpy for plotting
    error_x = error_x.cpu().detach().numpy()
    error_y = error_y.cpu().detach().numpy()
    error_z = error_z.cpu().detach().numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot error maps
    im0 = axes[0].imshow(error_x, cmap='hot')
    axes[0].set_title("X-Channel Error")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    im1 = axes[1].imshow(error_y, cmap='hot')
    axes[1].set_title("Y-Channel Error")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    im2 = axes[2].imshow(error_z, cmap='hot')
    axes[2].set_title("Z-Channel Error")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    return wandb.Image(fig, caption="Channel-wise Error Maps")


def test(model, test_dataloader, device, opt):
    model.eval()
    total_test_loss = 0.0
    
    # Initialize metrics
    total_epe_x = 0.0
    total_epe_y = 0.0
    total_epe_z = 0.0
    total_epe = 0.0
    total_angular_error = 0.0

    # Optional: save a few sample reconstructions to wandb
    save_samples = 3
    saved = 0
    sample_images = []
    error_maps = []

    with torch.no_grad():
        for i, svf in enumerate(test_dataloader):
            svf = svf.to(device).type(torch.float32)  # [B, 3, 128, 128, 128]

            recon, mu, logvar = model(svf)
            loss = vae_loss(recon, svf, mu, logvar, 
                         epe_weight=opt.epe_weight, 
                         angular_weight=opt.angular_weight)
            total_test_loss += loss.item()
            
            # Compute EPE and angular error
            epe_metrics = compute_epe(recon, svf)
            angular_err = compute_angular_error(recon, svf)
            
            # Accumulate metrics
            total_epe_x += epe_metrics['epe_x']
            total_epe_y += epe_metrics['epe_y']
            total_epe_z += epe_metrics['epe_z']
            total_epe += epe_metrics['epe_total']
            total_angular_error += angular_err

            # Save sample reconstructions and error maps
            if saved < save_samples:
                # Save reconstruction comparison
                sample_images.append(plot_sample(svf, recon, saved))
                
                # Save channel-wise error maps
                error_maps.append(plot_channel_errors(svf, recon, idx=0))
                
                saved += 1

    # Average the metrics
    num_batches = len(test_dataloader)
    avg_test_loss = total_test_loss / num_batches
    avg_epe_x = total_epe_x / num_batches
    avg_epe_y = total_epe_y / num_batches
    avg_epe_z = total_epe_z / num_batches
    avg_epe = total_epe / num_batches
    avg_angular_error = total_angular_error / num_batches

    print(f"\n🔍 Test Loss: {avg_test_loss:.4f}")
    print(f"📏 EPE - X: {avg_epe_x:.4f}, Y: {avg_epe_y:.4f}, Z: {avg_epe_z:.4f}, Total: {avg_epe:.4f}")
    print(f"🔄 Angular Error: {avg_angular_error:.2f}°")
    
    if not opt.debug:
        wandb.log({
            "Test Loss": avg_test_loss,
            "EPE_X": avg_epe_x,
            "EPE_Y": avg_epe_y,
            "EPE_Z": avg_epe_z,
            "EPE_Total": avg_epe,
            "Angular_Error": avg_angular_error,
            "SVF Reconstructions": sample_images,
            "Error Maps": error_maps
        })
    
    return avg_test_loss, avg_epe, avg_angular_error

random.seed(1337)

def train(opt):
   
    run_name = opt.run_name
    project_name = opt.project_name

    if not opt.debug:
        wandb.init(project=project_name, name=run_name)
        wandb.config.update(vars(opt))

    image_size = (opt.img_height, opt.img_height, opt.img_height) 

    save_directory = project_name

    print(f"Image Size: {image_size}, Number of Transformer Layers: {opt.num_layers}, Sequence Length: {opt.seq_len}\n")
    print(f"Using EPE loss: {opt.epe_weight > 0}, EPE weight: {opt.epe_weight}")
    print(f"Using Angular loss: {opt.angular_weight > 0}, Angular weight: {opt.angular_weight}")
    
    os.makedirs(f"/scratch/anirudhkaushik/SYNBADD/{save_directory}", exist_ok=True)

    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.float32)

    lambda_gp = 1
    accumulation_steps = opt.effective_batch_size // opt.batch_size

    nb_features = [
        [16, 32, 32, 32, 32],
        [32, 32, 32, 32, 32, 32, 16, 16]
    ]

    model = VAE3D_UNetSkip().to(device)


    if opt.n_gpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(opt.n_gpu)))

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=1e-5)
    # reduce lr on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=opt.patience, verbose=True)

    svf_file = "/scratch/anirudhkaushik/svf_metadata.csv"
    if opt.dataset == "ADNI":
        timepoints = ['sc', 'm06', 'm12', 'm18', 'm24', 'm36']
        csv_file = "/scratch/anirudhkaushik/ADNI1_Complete_DataSplits.csv"
    elif opt.dataset == "Toy":
        timepoints = ['sc', 'm06', 'm12', 'm18', 'm24', 'm30', 'm36']
        csv_file = "./metadata.csv"

    synthetic_dir="/scratch/anirudhkaushik/synthetic_deformations"

    # Create train, validation, and test datasets
    train_dataset = VAEdeformationDataset(
        svf_file=svf_file,
        csv_file=csv_file,
        fold=opt.fold,
        subset='train',
        synthetic_dir=synthetic_dir
    )
    val_dataset = VAEdeformationDataset(
        svf_file=svf_file,
        csv_file=csv_file,
        fold=opt.fold,
        subset='val',
        synthetic_dir=synthetic_dir
    )
    test_dataset = VAEdeformationDataset(
        svf_file=svf_file,
        csv_file=csv_file,
        fold=opt.fold,
        subset='test',
        synthetic_dir=synthetic_dir
    )
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.n_cpu,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=opt.batch_size,
                                shuffle=False,
                                num_workers=opt.n_cpu,
                                pin_memory=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.n_cpu,
                                 pin_memory=True)

    # Log dataset sizes
    if not opt.debug:
        wandb.log({
            "Train Size": len(train_dataset),
            "Validation Size": len(val_dataset),
            "Test Size": len(test_dataset)
        })

    best_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = opt.patience * 2

    train_losses = []
    val_losses = []
    
    # Track metrics during training
    train_epe_history = []
    train_angular_history = []
    val_epe_history = []
    val_angular_history = []

    for epoch in range(opt.n_epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_epe_x = 0.0
        epoch_train_epe_y = 0.0
        epoch_train_epe_z = 0.0
        epoch_train_epe = 0.0
        epoch_train_angular = 0.0

        with tqdm(total=len(train_dataloader.dataset), desc=f"Epoch {epoch+1}/{opt.n_epochs}") as progress_bar:
            for i, batch in enumerate(train_dataloader):
                svf = batch.to(device)  # [B, 3, 128, 128, 128]
                svf = Variable(svf.type(torch.float32)).to(device)

                optimizer.zero_grad()
                recon, mu, logvar = model(svf)
                
                # Use the updated loss function with EPE and angular error components
                loss = vae_loss(recon, svf, mu, logvar, 
                             epe_weight=opt.epe_weight, 
                             angular_weight=opt.angular_weight)
                loss.backward()
                optimizer.step()

                # Compute metrics (no gradients needed)
                with torch.no_grad():
                    epe_metrics = compute_epe(recon, svf)
                    angular_err = compute_angular_error(recon, svf)
                    
                    # Update epoch metrics
                    epoch_train_epe_x += epe_metrics['epe_x']
                    epoch_train_epe_y += epe_metrics['epe_y']
                    epoch_train_epe_z += epe_metrics['epe_z']
                    epoch_train_epe += epe_metrics['epe_total']
                    epoch_train_angular += angular_err

                epoch_train_loss += loss.item()
                progress_bar.update(svf.size(0))
                progress_bar.set_postfix({"Train Loss": f"{loss.item():.4f}"})

        # Calculate average metrics for the epoch
        num_batches = len(train_dataloader)
        avg_train_loss = epoch_train_loss / num_batches
        avg_train_epe_x = epoch_train_epe_x / num_batches
        avg_train_epe_y = epoch_train_epe_y / num_batches
        avg_train_epe_z = epoch_train_epe_z / num_batches
        avg_train_epe = epoch_train_epe / num_batches
        avg_train_angular = epoch_train_angular / num_batches
        
        train_losses.append(avg_train_loss)
        train_epe_history.append(avg_train_epe)
        train_angular_history.append(avg_train_angular)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_epe_x = 0.0
        epoch_val_epe_y = 0.0
        epoch_val_epe_z = 0.0
        epoch_val_epe = 0.0
        epoch_val_angular = 0.0
        
        with torch.no_grad():
            for val_svf in val_dataloader:
                val_svf = val_svf.to(device).type(torch.float32)
                recon, mu, logvar = model(val_svf)
                
                # Use the updated loss function with EPE and angular error components
                val_loss = vae_loss(recon, val_svf, mu, logvar, 
                                 epe_weight=opt.epe_weight, 
                                 angular_weight=opt.angular_weight)
                
                # Compute metrics
                epe_metrics = compute_epe(recon, val_svf)
                angular_err = compute_angular_error(recon, val_svf)
                
                # Update epoch metrics
                epoch_val_epe_x += epe_metrics['epe_x']
                epoch_val_epe_y += epe_metrics['epe_y']
                epoch_val_epe_z += epe_metrics['epe_z']
                epoch_val_epe += epe_metrics['epe_total']
                epoch_val_angular += angular_err
                epoch_val_loss += val_loss.item()

        # Calculate average validation metrics
        num_val_batches = len(val_dataloader)
        avg_val_loss = epoch_val_loss / num_val_batches
        avg_val_epe_x = epoch_val_epe_x / num_val_batches
        avg_val_epe_y = epoch_val_epe_y / num_val_batches
        avg_val_epe_z = epoch_val_epe_z / num_val_batches
        avg_val_epe = epoch_val_epe / num_val_batches
        avg_val_angular = epoch_val_angular / num_val_batches
        
        val_losses.append(avg_val_loss)
        val_epe_history.append(avg_val_epe)
        val_angular_history.append(avg_val_angular)
        
        scheduler.step(avg_val_loss)

        if not opt.debug:
            wandb.log({
                "Epoch": epoch + 1,
                "Train Loss": avg_train_loss,
                "Train EPE_X": avg_train_epe_x,
                "Train EPE_Y": avg_train_epe_y,
                "Train EPE_Z": avg_train_epe_z,
                "Train EPE_Total": avg_train_epe,
                "Train Angular_Error": avg_train_angular,
                "Validation Loss": avg_val_loss,
                "Validation EPE_X": avg_val_epe_x,
                "Validation EPE_Y": avg_val_epe_y,
                "Validation EPE_Z": avg_val_epe_z,
                "Validation EPE_Total": avg_val_epe,
                "Validation Angular_Error": avg_val_angular,
                "Learning Rate": optimizer.param_groups[0]["lr"]
            })

        print(f"Epoch [{epoch+1}/{opt.n_epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  Train EPE: {avg_train_epe:.4f} | Val EPE: {avg_val_epe:.4f}")
        print(f"  Train Angular Error: {avg_train_angular:.2f}° | Val Angular Error: {avg_val_angular:.2f}°")

        # Early stopping logic - now considering both loss and EPE
        if 1e-4 <= best_loss - avg_val_loss :
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"/scratch/anirudhkaushik/SYNBADD/{save_directory}/best_model.pth")

            # Save a few sample reconstructions and error maps to wandb
            if not opt.debug:
                sample_images = []
                error_maps = []
                
                for i in range(min(3, val_svf.shape[0])):
                    # Save reconstruction comparison
                    input_jac = jacobian_determinant_torch(val_svf)
                    recon_jac = jacobian_determinant_torch(recon)

                    input_sample = input_jac[i].cpu().detach().numpy()
                    recon_sample = recon_jac[i].cpu().detach().numpy()

                    mid_slice = input_sample.shape[2] // 2

                    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
                    axes[0].imshow(input_sample[:, :, mid_slice], cmap='jet')
                    axes[0].set_title("Input SVF (Jacobian Det)")
                    axes[1].imshow(recon_sample[:, :, mid_slice], cmap='jet')
                    axes[1].set_title("Reconstructed SVF (Jacobian Det)")
                    axes[0].axis('off')
                    axes[1].axis('off')

                    img_name = f"SVF_Reconstruction_{i}"
                    sample_images.append(wandb.Image(fig, caption=img_name))
                    plt.close()
                    
                    # Add error maps
                    error_maps.append(plot_channel_errors(val_svf, recon, idx=i))

                wandb.log({
                    "Best Model Reconstructions": sample_images,
                    "Best Model Error Maps": error_maps
                })
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(f"/scratch/anirudhkaushik/SYNBADD/{project_name}/best_model.pth"))
    test_loss, test_epe, test_angular = test(model, test_dataloader, device, opt)
    
    # Plot and log performance history
    if not opt.debug:
        # Create loss history plot
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax.set_title('Loss History')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        wandb.log({"Loss History": wandb.Image(fig)})
        plt.close()
        
        # Create EPE history plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_epe_history, 'b-', label='Training EPE')
        ax.plot(epochs, val_epe_history, 'r-', label='Validation EPE')
        ax.set_title('EPE History')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('EPE')
        ax.legend()
        ax.grid(True)
        wandb.log({"EPE History": wandb.Image(fig)})
        plt.close()
        
        # Create Angular Error history plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_angular_history, 'b-', label='Training Angular Error')
        ax.plot(epochs, val_angular_history, 'r-', label='Validation Angular Error')
        ax.set_title('Angular Error History')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Angular Error (degrees)')
        ax.legend()
        ax.grid(True)
        wandb.log({"Angular Error History": wandb.Image(fig)})
        plt.close()
        
        # Log final test metrics
        wandb.log({
            "Final Test Loss": test_loss,
            "Final Test EPE": test_epe,
            "Final Test Angular Error": test_angular
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a 3D VAE for SVF data")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--fold", type=int, default=1, help="fold number")
    parser.add_argument("--effective_batch_size", type=int, default=32, help="effective batch size after gradient accumulation")
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--aug_p", type=float, default=0.8, help="amount of augmentation to use")
    parser.add_argument("--n_cpu", type=int, default=3, help="number of cpu threads to use during batch generation")
    parser.add_argument("--n_gpu", type=int, default=4, help="number of gpus on machine to use")
    parser.add_argument("--img_height", type=int, default=128, help="size of image height")
    parser.add_argument("--c_dim", type=int, default=3, help="number of dimensions for condition (target_labels)")
    parser.add_argument("--jacobian_coeff", type=float, default=0.1, help="weight for neg jacobian loss")
    parser.add_argument("--jac_dist_coeff", type=float, default=0.05, help="weight for jacobian distance loss")
    parser.add_argument("--disc_coeff", type=float, default=1.0, help="weight for adv loss")
    parser.add_argument("--smooth_coeff", type=float, default=0.05, help="weight for smooth loss")
    parser.add_argument("--lddmm_coeff", type=float, default=0.01, help="weight for lddmm loss")
    parser.add_argument("--use_past_data", type=str2bool, nargs='?', const=True, default=True, help="use past data for ViVit")
    parser.add_argument("--use_future_data", type=str2bool, nargs='?', const=True, default=True, help="use future data for ViVit")
    parser.add_argument("--dataset", type=str, default="ADNI", help="which dataset to use for training", choices=["ADNI", "Toy"])
    parser.add_argument("--target_timepoint", type=str, default="m24", help="timepoint to predict")
    parser.add_argument("--use_deformation_field", type=str2bool, nargs='?', const=True, default=True, help="use deformation field for alignment loss")
    parser.add_argument("--patience", type=int, default=5, help="scheduler patience")
    parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=False, help="run code in debug mode")
    parser.add_argument("--specials", type=str, default="DAFT_Resnet_SVF_1LayerTransformer_OnlyPast", help="special run parameters")
    parser.add_argument("--use_val_dataset", type=str2bool, nargs='?', const=True, default=False, help="use validation dataset for training")
    parser.add_argument("--in_channels", type=int, default=3, help="number of input channels to the generator")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers in the transformer model")
    parser.add_argument("--seq_len", type=int, default=5, help="number of frames in the transformer model")
    parser.add_argument("--use_discriminator", type=str2bool, nargs='?', const=True, default=True, help="use discriminator for training")
    parser.add_argument("--use_bidir_transformer", type=str2bool, nargs='?', const=True, default=True, help="use bidirectional transformer for training")
    parser.add_argument("--project_name", type=str, default="VAE_10_Aug_per_sample", help="wandb project name")
    parser.add_argument("--run_name", type=str, default="Vanilla_VAE_model", help="wandb run name")
    parser.add_argument("--epe_weight", type=float, default=0.05, help="weight for EPE loss")
    parser.add_argument("--angular_weight", type=float, default=0.05, help="weight for angular error loss")



    opt = parser.parse_args()
    
    train(opt)