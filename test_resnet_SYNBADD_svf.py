# =============================================================================
# File: test_resnet_SYNBADD_svf.py
# Purpose: Inference/evaluation entrypoint for a specific experiment variant.
# Prerequisites: Trained checkpoints and fold-specific dataset metadata.
# Usually Run Next: Run comparison analysis scripts or aggregate fold metrics.
# Model Components Used: Uses model modules imported in this file; latest subjectwise artifact generation uses test_model_subjectwise.py.
# =============================================================================

import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from matplotlib import pyplot as plt

from utils import plot_slices
from losses import compute_metrics, smooth_loss_l2, ssim_only_loss, jacobian_determinant_torch
import nibabel as nib
import pandas as pd

def validate(generator, resnet_encoder, transformer_model, discriminator_model, val_dataloader, criterion_lddmm, device, opt, run_name):
    project_name = opt.project_name
    save_directory =  project_name + "/val"
    os.makedirs("/scratch/anirudhkaushik/SYNBADD/" + save_directory, exist_ok=True)

    generator.eval()
    resnet_encoder.eval()
    transformer_model.eval()
    if opt.use_discriminator:
        discriminator_model.eval()
    
    val_loss = 0
    ssim_values, mean_jacobian_losses, std_jacobian_losses, mse_losses, psnr_losses, ssim_losses, avg_jacobian_losses, nrmse_losses = [], [], [], [], [], [], [], []
    disc_adv_losses = []

    with torch.no_grad():
        with tqdm(total=len(val_dataloader.dataset), desc="Validation") as progress_bar:
            for i, batch in enumerate(val_dataloader):
                stacked_svf, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint, target_timepoint_index, target_svf = [batch[k].to(device) for k in ["stacked_svf", "latest_scan", "target_scan", "class_label", "reg_label", "target_timepoint", "latest_timepoint", "target_timepoint_index", "target_svf"]]
                stacked_svf, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint, target_svf = [Variable(v.type(torch.float32)).to(device) for v in [stacked_svf, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint, target_svf]]
                target_labels = torch.cat([class_labels, reg_labels, target_timepoint], dim=1)


                if opt.use_past_data or opt.use_future_data:
                    # [B, T, C, H, W, D] Collapse batch dimension into time dimension
                    batch_size, temporal_dim, c_dim, h, w, d = stacked_svf.shape
                    stacked_svf = stacked_svf.view(-1, *stacked_svf.shape[2:])
                    # iterate and pass it through 3d resnet
                    contextualized_scan = resnet_encoder(stacked_svf)
                    # make stacked_svf normal sized again
                    stacked_svf = stacked_svf.view(batch_size, temporal_dim, c_dim, h, w, d)
                    # [B*T, C, H, W, D] -> [B*T, 512]
                    contextualized_scan = contextualized_scan.view(batch_size, temporal_dim, -1)
                    # [B*T, 512] -> [B, T, 512]
                    if opt.use_bidir_transformer:
                        contextualized_scan = transformer_model(contextualized_scan, target_timepoint_index)
                    else:
                        contextualized_scan = transformer_model(contextualized_scan)
                    # maxpooling over time dimension [B, T, 512] -> [B, 512]
                else:
                    contextualized_scan = latest_scan

                y_synthetic, synthetic_field = generator(latest_scan, contextualized_scan, target_labels)


                if opt.use_discriminator:
                    # Compute loss with real images
                    out_real = discriminator_model(latest_scan, target_labels)
                    d_loss_real = torch.mean(F.softplus(-out_real))

                    # Create fake images
                    out_fake = discriminator_model(y_synthetic.detach(), target_labels)
                    d_loss_fake = torch.mean(F.softplus(out_fake))
                    d_adv = d_loss_real + d_loss_fake
                else:
                    d_adv = torch.tensor(0.0).to(device)

                ssim_loss = ssim_only_loss(y_synthetic, target_scan)
                lddmm_loss = criterion_lddmm(target_scan, y_synthetic) # ensures Jac of 1 b/w target and warped meaning warped and target should have uniform velocity field b/w them
                smooth_loss = smooth_loss_l2(synthetic_field)

                jac_det = jacobian_determinant_torch(synthetic_field)  # shape: (B, H-1, W-1, D-1)
                jac_dist_loss = torch.mean((jac_det - 0.9).pow(2))  # Center around 0.9
                # jac = torch.mean(torch.abs(jac_det - 1.0))  # scalar loss
                ncc = F.mse_loss(y_synthetic, target_scan)

                # to penalize negative jacobian
                neg_jacobian_loss = torch.relu(-jac_det).mean()

                if opt.use_discriminator:
                    total_loss = ssim_loss + opt.disc_coeff*d_adv + opt.jacobian_coeff * neg_jacobian_loss + opt.smooth_coeff*smooth_loss + opt.lddmm_coeff*lddmm_loss + opt.jac_dist_coeff*jac_dist_loss
                else:
                    total_loss = ssim_loss + opt.jacobian_coeff*neg_jacobian_loss + opt.smooth_coeff*smooth_loss + opt.lddmm_coeff*lddmm_loss + opt.jac_dist_coeff*jac_dist_loss
                # total_loss = total_loss / 3
                val_loss += total_loss.item()

                mse_loss, psnr_value, ssim_value, mean_jacobian, std_jacobian, _, nrmse_loss = compute_metrics(target_scan, y_synthetic, synthetic_field, criterion_lddmm, device)
                jacobian_det = jacobian_determinant_torch(synthetic_field)  # shape: (B, H-1, W-1, D-1)
                target_jacobian_det = jacobian_determinant_torch(target_svf)  # shape: (B, H-1, W-1, D-1)

                # Store metrics for aggregation
                ssim_losses.append(ssim_loss.item())
                mean_jacobian_losses.append(mean_jacobian)
                std_jacobian_losses.append(std_jacobian)
                mse_losses.append(mse_loss)
                psnr_losses.append(psnr_value)
                ssim_values.append(ssim_value)
                avg_jacobian_losses.append(lddmm_loss.item())
                nrmse_losses.append(nrmse_loss)
                disc_adv_losses.append(d_adv.item())
                
                progress_bar.update(stacked_svf.size(0))
                progress_bar.set_postfix({
                    "Validation SSIM Loss": ssim_loss.item(),
                    "Validation Mse": mse_loss,
                    "Validation PSNR": psnr_value,
                    "Validation SSIM": ssim_value,
                    "Validation Total Loss": total_loss.item(),
                    "Validation Mean Jac": mean_jacobian,
                    "Validation Std Jac": std_jacobian,
                    "Validation LDDMM Loss": lddmm_loss.item(),
                    "Validation Adv Loss": d_adv.item(),
                })

                # Save visualization for a few test samples
                if i %2 == 0:
                    group_map = {0.0: 'CN', 1.0: 'MCI', 2.0: 'AD'}
                    loss_map = {"mse": mse_loss,"psnr": psnr_value,"ssim": ssim_value,"mean jacobian": mean_jacobian, "std jacobian": std_jacobian}

                    fig = plot_slices(latest_scan, target_scan, y_synthetic, jacobian_det, target_jacobian_det, loss_map,
                                     reg_labels[0].item(), group_map[class_labels[0].item()], target_timepoint[0].item(), latest_scan_timepoint[0].item())
                    
                    save_path = f"/scratch/anirudhkaushik/SYNBADD/{save_directory}/val_viz_{run_name}_{i}.png"
                    plt.savefig(save_path)
                    plt.close(fig)
                    
                    if not opt.debug:
                        wandb.log({f"Validation Sample {i}": wandb.Image(fig)})
    
    # Calculate average metrics
    avg_val_loss = val_loss / len(val_dataloader)
    avg_ssim_loss = np.mean(ssim_losses)
    avg_mean_jacobian = np.mean(mean_jacobian_losses)
    avg_std_jacobian = np.mean(std_jacobian_losses)
    avg_psnr = np.mean(psnr_losses)
    avg_ssim = np.mean(ssim_values)
    avg_mse = np.mean(mse_losses)
    avg_jacobian = np.mean(avg_jacobian_losses)
    avg_nrmse = np.mean(nrmse_losses)
    avg_disc_adv_loss = np.mean(disc_adv_losses)
    
    
    avg_metrics = {
        "Validation Loss": avg_val_loss, 
        "Validation SSIM Loss": avg_ssim_loss,
        "Validation Mean Jacobian": avg_mean_jacobian,
        "Validation Std Jacobian": avg_std_jacobian,
        "Validation PSNR": avg_psnr,
        "Validation SSIM": avg_ssim,
        "Validation MSE": avg_mse,
        "Validation LDDMM Loss": avg_jacobian,
        "Validation NRMSE": avg_nrmse,
        "Validation Adv Loss": avg_disc_adv_loss,
    }
    
    # Convert to DataFrame with just one row
    df = pd.DataFrame([avg_metrics])
    csv_path = f"/scratch/anirudhkaushik/SYNBADD/{save_directory}/{run_name}_validate.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Validation metrics saved to {csv_path}")

    print("\n\nValidation Loss: {:.4f}".format(avg_val_loss), " Validation Align Loss: {:.4f}".format(avg_ssim_loss), " Validation Mean Jacobian: {:.4f}".format(avg_mean_jacobian), " Validation Std Jacobian: {:.4f}".format(avg_std_jacobian), " Validation PSNR: {:.4f}".format(avg_psnr), " Validation SSIM: {:.4f}".format(avg_ssim), " Validation MSE: {:.4f}".format(avg_mse), "Validation LDDMM Loss: {:.4f}".format(avg_jacobian), " Validation NRMSE: {:.4f}".format(avg_nrmse), " Validation Adv Loss: {:.4f}".format(avg_disc_adv_loss), end="\n\n")

    if not opt.debug:
        wandb.log({
            "Validation Loss": avg_val_loss, 
            "Validation SSIM Loss": avg_ssim_loss,
            "Validation Mean Jacobian": avg_mean_jacobian,
            "Validation Std Jacobian": avg_std_jacobian,
            "Validation PSNR": avg_psnr,
            "Validation SSIM": avg_ssim,
            "Validation MSE": avg_mse,
            "Validation LDDMM Loss": avg_jacobian,
            "Validation NRMSE": avg_nrmse,
            "Validation Adv Loss": avg_disc_adv_loss,
        })
                
    return avg_val_loss

def test(generator, resnet_encoder, transformer_model, discriminator_model, test_dataloader, criterion_lddmm, device, opt, run_name):
    project_name = opt.project_name
    save_directory =  project_name + "/test"
    os.makedirs("/scratch/anirudhkaushik/SYNBADD/" + save_directory, exist_ok=True)
    
    generator.eval()
    resnet_encoder.eval()
    transformer_model.eval()
    if opt.use_discriminator:
        discriminator_model.eval()
    
    test_loss = 0
    ssim_values, mean_jacobian_losses, std_jacobian_losses, mse_losses, psnr_losses, ssim_losses, avg_jacobian_losses, nrmse_losses = [], [], [], [], [], [], [], []
    disc_adv_losses = []

    with torch.no_grad():
        with tqdm(total=len(test_dataloader.dataset), desc="Testing") as progress_bar:
             for i, batch in enumerate(test_dataloader):
                stacked_svf, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint, target_timepoint_index, target_svf = [batch[k].to(device) for k in ["stacked_svf", "latest_scan", "target_scan", "class_label", "reg_label", "target_timepoint", "latest_timepoint", "target_timepoint_index", "target_svf"]]
                stacked_svf, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint, target_svf = [Variable(v.type(torch.float32)).to(device) for v in [stacked_svf, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint, target_svf]]
                target_labels = torch.cat([class_labels, reg_labels, target_timepoint], dim=1)

                if opt.use_past_data or opt.use_future_data:
                    # [B, T, C, H, W, D] Collapse batch dimension into time dimension
                    batch_size, temporal_dim, c_dim, h, w, d = stacked_svf.shape
                    stacked_svf = stacked_svf.view(-1, *stacked_svf.shape[2:])
                    # iterate and pass it through 3d resnet
                    contextualized_scan = resnet_encoder(stacked_svf)
                    # make stacked_svf normal sized again
                    stacked_svf = stacked_svf.view(batch_size, temporal_dim, c_dim, h, w, d)
                    # [B*T, C, H, W, D] -> [B*T, 512]
                    contextualized_scan = contextualized_scan.view(batch_size, temporal_dim, -1)
                    # [B*T, 512] -> [B, T, 512]
                    if opt.use_bidir_transformer:
                        contextualized_scan = transformer_model(contextualized_scan, target_timepoint_index)
                    else:
                        contextualized_scan = transformer_model(contextualized_scan)
                    # maxpooling over time dimension [B, T, 512] -> [B, 512]
                else:
                    contextualized_scan = latest_scan

                y_synthetic, synthetic_field = generator(latest_scan, contextualized_scan, target_labels)
                
                if opt.use_discriminator:
                    # Compute loss with real images
                    out_real = discriminator_model(latest_scan, target_labels)
                    d_loss_real = torch.mean(F.softplus(-out_real))

                    # Create fake images
                    out_fake = discriminator_model(y_synthetic.detach(), target_labels)
                    d_loss_fake = torch.mean(F.softplus(out_fake))
                    d_adv = d_loss_real + d_loss_fake
                else:
                    d_adv = torch.tensor(0.0).to(device)

                ssim_loss = ssim_only_loss(y_synthetic, target_scan)
                lddmm_loss = criterion_lddmm(target_scan, y_synthetic) # ensures Jac of 1 b/w target and warped meaning warped and target should have uniform velocity field b/w them
                smooth_loss = smooth_loss_l2(synthetic_field)

                jac_det = jacobian_determinant_torch(synthetic_field)  # shape: (B, H-1, W-1, D-1)
                jac_dist_loss = torch.mean((jac_det - 0.9).pow(2))  # Center around 0.9
                # jac = torch.mean(torch.abs(jac_det - 1.0))  # scalar loss
                ncc = F.mse_loss(y_synthetic, target_scan)

                # to penalize negative jacobian
                neg_jacobian_loss = torch.relu(-jac_det).mean()

                if opt.use_discriminator:
                    total_loss = ssim_loss + opt.disc_coeff*d_adv + opt.jacobian_coeff * neg_jacobian_loss + opt.smooth_coeff*smooth_loss + opt.lddmm_coeff*lddmm_loss + opt.jac_dist_coeff*jac_dist_loss
                else:
                    total_loss = ssim_loss + opt.jacobian_coeff*neg_jacobian_loss + opt.smooth_coeff*smooth_loss + opt.lddmm_coeff*lddmm_loss + opt.jac_dist_coeff*jac_dist_loss
                # total_loss = total_loss / 3
                test_loss += total_loss.item()

                mse_loss, psnr_value, ssim_value, mean_jacobian, std_jacobian, _, nrmse_loss = compute_metrics(target_scan, y_synthetic, synthetic_field, criterion_lddmm, device)
                jacobian_det = jacobian_determinant_torch(synthetic_field)  # shape: (B, H-1, W-1, D-1)
                target_jacobian_det = jacobian_determinant_torch(target_svf)  # shape: (B, H-1, W-1, D-1)

                # Store metrics for aggregation
                ssim_losses.append(ssim_loss.item())
                mean_jacobian_losses.append(mean_jacobian)
                std_jacobian_losses.append(std_jacobian)
                mse_losses.append(mse_loss)
                psnr_losses.append(psnr_value)
                ssim_values.append(ssim_value)
                avg_jacobian_losses.append(lddmm_loss.item())
                nrmse_losses.append(nrmse_loss)
                disc_adv_losses.append(d_adv.item())

                progress_bar.update(stacked_svf.size(0))
                progress_bar.set_postfix({
                    "Test SSIM Loss": ssim_loss.item(),
                    "Test LDDMM Loss": lddmm_loss.item(),
                    "Test Mse": mse_loss,
                    "Test PSNR": psnr_value,
                    "Test SSIM": ssim_value,
                    "Test Total Loss": total_loss.item(),
                    "Test Mean Jac": mean_jacobian,
                    "Test Std Jac": std_jacobian,
                    "Test NRMSE": nrmse_loss,
                    "Test Adv Loss": d_adv.item(),
                })
                
                # Save visualization for a few test samples
                if i == 0:
                    group_map = {0.0: 'CN', 1.0: 'MCI', 2.0: 'AD'}
                    loss_map = {"mse": mse_loss,"psnr": psnr_value,"ssim": ssim_value,"mean jacobian": mean_jacobian, "std jacobian": std_jacobian}

                    fig = plot_slices(latest_scan, target_scan, y_synthetic, jacobian_det, target_jacobian_det, loss_map,
                                     reg_labels[0].item(), group_map[class_labels[0].item()], target_timepoint[0].item(), latest_scan_timepoint[0].item())
                    
                    save_path = f"/scratch/anirudhkaushik/SYNBADD/{save_directory}/test_viz_{run_name}_{i}.png"
                    plt.savefig(save_path)
                    plt.close(fig)
                    
                    if not opt.debug:
                        wandb.log({f"Test Sample": wandb.Image(fig)})
    
    # Calculate average metrics
    avg_test_loss = test_loss / len(test_dataloader)
    avg_ssim_loss = np.mean(ssim_losses)
    avg_mean_jacobian = np.mean(mean_jacobian_losses)
    avg_std_jacobian = np.mean(std_jacobian_losses)
    avg_psnr = np.mean(psnr_losses)
    avg_ssim = np.mean(ssim_values)
    avg_mse = np.mean(mse_losses)
    avg_jacobian = np.mean(avg_jacobian_losses)
    avg_nrmse = np.mean(nrmse_losses)
    avg_disc_adv_loss = np.mean(disc_adv_losses)
    
    avg_metrics = {
        "Test Loss": avg_test_loss, 
        "Test SSIM Loss": avg_ssim_loss,
        "Test Mean Jacobian": avg_mean_jacobian,
        "Test Std Jacobian": avg_std_jacobian,
        "Test PSNR": avg_psnr,
        "Test SSIM": avg_ssim,
        "Test MSE": avg_mse,
        "Test LDDMM Loss": avg_jacobian,
        "Test NRMSE": avg_nrmse,
        "Test Adv Loss": avg_disc_adv_loss,
    }
    
    # Convert to DataFrame with just one row
    df = pd.DataFrame([avg_metrics])
    csv_path = f"/scratch/anirudhkaushik/SYNBADD/{save_directory}/{run_name}_test.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Test metrics saved to {csv_path}")
                
    print("\n\nTest Loss: {:.4f}".format(avg_test_loss), " Test Align Loss: {:.4f}".format(avg_ssim_loss), " Test Mean Jacobian: {:.4f}".format(avg_mean_jacobian), " Test Std Jacobian: {:.4f}".format(avg_std_jacobian), " Test PSNR: {:.4f}".format(avg_psnr), " Test SSIM: {:.4f}".format(avg_ssim), " Test MSE: {:.4f}".format(avg_mse), "Test LDDMM Loss: {:.4f}".format(avg_jacobian), " Test NRMSE: {:.4f}".format(avg_nrmse), " Test Adv Loss: {:.4f}".format(avg_disc_adv_loss), end="\n\n")
    
    if not opt.debug:
        wandb.log({
            "Test Loss": avg_test_loss, 
            "Test SSIM Loss": avg_ssim_loss,
            "Test Mean Jacobian": avg_mean_jacobian,
            "Test Std Jacobian": avg_std_jacobian,
            "Test PSNR": avg_psnr,
            "Test SSIM": avg_ssim,
            "Test MSE": avg_mse,
            "Test LDDMM Loss": avg_jacobian,
            "Test NRMSE": avg_nrmse,
            "Test Adv Loss": avg_disc_adv_loss,
        })
    
    return avg_test_loss