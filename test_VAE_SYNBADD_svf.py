# =============================================================================
# File: test_VAE_SYNBADD_svf.py
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
from losses import compute_metrics, smooth_loss_l2, ssim_only_loss, jacobian_determinant_torch, channelwise_epe_loss, angular_error_loss
import nibabel as nib
import pandas as pd

def validate(generator, vae_model, regression_model, transformer_model, discriminator_model, val_dataloader, criterion_lddmm, device, opt, run_name):
    project_name = opt.project_name
    save_directory =  project_name + "/val"
    os.makedirs("/scratch/anirudhkaushik/SYNBADD/" + save_directory, exist_ok=True)

    generator.eval()
    vae_model.eval()
    if opt.use_transformer:
        transformer_model.eval()
    else:
        regression_model.eval()
    if opt.use_discriminator:
        discriminator_model.eval()
    
    val_loss = 0
    ssim_values, mean_jacobian_losses, std_jacobian_losses, mse_losses, psnr_losses, ssim_losses, avg_jacobian_losses, nrmse_losses, mssim_values = [], [], [], [], [], [], [], [], []
    disc_adv_losses = []
    epe_x_losses, epe_y_losses, epe_z_losses, angular_error_losses = [], [], [], []

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
                    
                    # with torch.no_grad():
                    #     contextualized_scan, logvar, skips = vae_model(stacked_svf, "encode", None)
                    #     contextualized_scan, logvar, skips = contextualized_scan.detach(), logvar.detach(), [s.detach() for s in skips]

                    # # take mean of skips which is a list
                    # for skip_ind in range(len(skips)):
                    #     s_b, s_c, s_h, s_w, s_d = skips[skip_ind].shape
                    #     skips[skip_ind] = skips[skip_ind].view(batch_size, temporal_dim, s_c, s_h, s_w, s_d)
                    #     # average each across the temporal dimension
                    #     skips[skip_ind] = skips[skip_ind].mean(dim=1)
                    
                    # contextualized_scan = contextualized_scan.view(batch_size, temporal_dim, -1)
                    # if opt.use_transformer:
                    #     contextualized_scan = transformer_model(contextualized_scan, target_timepoint_index)
                    # else:
                    #     contextualized_scan = regression_model(contextualized_scan)
                    # with torch.no_grad():
                    #     contextualized_scan = vae_model(contextualized_scan, "decode", skips)

                    # # make stacked_svf normal sized again
                    stacked_svf = stacked_svf.view(batch_size, temporal_dim, c_dim, h, w, d)
                    # [B*T, C, H, W, D] -> [B*T, 512]

                else:
                    contextualized_scan = latest_scan
                # do averaged imputation instead of generation
                contextualized_scan = stacked_svf.mean(dim=1) # take average along temporal dimension
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

                _, epe_x, epe_y, epe_z = channelwise_epe_loss(synthetic_field, target_svf)

                # Compute angular error loss
                angular_error = angular_error_loss(synthetic_field, target_svf)

                # Update the total_loss calculation to include the new losses
                if opt.use_discriminator:
                    total_loss = opt.ssim_coeff*ssim_loss + opt.disc_coeff*d_adv + opt.jacobian_coeff * neg_jacobian_loss + opt.smooth_coeff*smooth_loss + opt.jac_dist_coeff*jac_dist_loss + opt.lddmm_coeff*lddmm_loss + opt.epe_x_coeff*epe_x + opt.epe_y_coeff*epe_y + opt.epe_z_coeff*epe_z + opt.angular_coeff*angular_error
                else:
                    total_loss = opt.ssim_coeff*ssim_loss + opt.jacobian_coeff*neg_jacobian_loss + opt.smooth_coeff*smooth_loss + opt.jac_dist_coeff*jac_dist_loss + opt.lddmm_coeff*lddmm_loss + opt.epe_x_coeff*epe_x + opt.epe_y_coeff*epe_y + opt.epe_z_coeff*epe_z + opt.angular_coeff*angular_error


                val_loss += total_loss.item()

                mse_loss, psnr_value, ssim_value, mean_jacobian, std_jacobian, _, nrmse_loss, mssim_value = compute_metrics(target_scan, y_synthetic, synthetic_field, criterion_lddmm, device)
                jacobian_det = jacobian_determinant_torch(synthetic_field)  # shape: (B, H-1, W-1, D-1)
                target_jacobian_det = jacobian_determinant_torch(target_svf)  # shape: (B, H-1, W-1, D-1)

                # Store metrics for aggregation
                ssim_losses.append(ssim_loss.item())
                mean_jacobian_losses.append(mean_jacobian)
                std_jacobian_losses.append(std_jacobian)
                mse_losses.append(mse_loss)
                psnr_losses.append(psnr_value)
                ssim_values.append(ssim_value)
                mssim_values.append(mssim_value)
                avg_jacobian_losses.append(lddmm_loss.item())
                nrmse_losses.append(nrmse_loss)
                disc_adv_losses.append(d_adv.item())
                epe_x_losses.append(epe_x.item())
                epe_y_losses.append(epe_y.item())
                epe_z_losses.append(epe_z.item())
                angular_error_losses.append(angular_error.item())
                
                progress_bar.update(stacked_svf.size(0))
                progress_bar.set_postfix({
                    "Validation SSIM Loss": ssim_loss.item(),
                    "Validation Mse": mse_loss,
                    "Validation PSNR": psnr_value,
                    "Validation SSIM": ssim_value,
                    "Validation MS_SSIM": mssim_value,
                    "Validation Total Loss": total_loss.item(),
                    "Validation Mean Jac": mean_jacobian,
                    "Validation Std Jac": std_jacobian,
                    "Validation LDDMM Loss": lddmm_loss.item(),
                    "Validation Adv Loss": d_adv.item(),
                    "Validation EPE X": epe_x.item(),
                    "Validation EPE Y": epe_y.item(),
                    "Validation EPE Z": epe_z.item(),
                    "Validation Angular": angular_error.item(),
                })

                # Save visualization for a few test samples
                if i %10 == 0:
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
    avg_mssim = np.mean(mssim_values)
    avg_mse = np.mean(mse_losses)
    avg_jacobian = np.mean(avg_jacobian_losses)
    avg_nrmse = np.mean(nrmse_losses)
    avg_disc_adv_loss = np.mean(disc_adv_losses)
    # Calculate average metrics
    avg_epe_x = np.mean(epe_x_losses)
    avg_epe_y = np.mean(epe_y_losses)
    avg_epe_z = np.mean(epe_z_losses)
    avg_angular_error = np.mean(angular_error_losses)
    
    
    avg_metrics = {
        "Validation Loss": avg_val_loss, 
        "Validation SSIM Loss": avg_ssim_loss,
        "Validation Mean Jacobian": avg_mean_jacobian,
        "Validation Std Jacobian": avg_std_jacobian,
        "Validation PSNR": avg_psnr,
        "Validation SSIM": avg_ssim,
        "Validation MS_SSIM": avg_mssim,
        "Validation MSE": avg_mse,
        "Validation LDDMM Loss": avg_jacobian,
        "Validation NRMSE": avg_nrmse,
        "Validation Adv Loss": avg_disc_adv_loss,
        "Validation EPE X": avg_epe_x,
        "Validation EPE Y": avg_epe_y,
        "Validation EPE Z": avg_epe_z,
        "Validation Angular Error": avg_angular_error,
    }
    
    # Convert to DataFrame with just one row
    df = pd.DataFrame([avg_metrics])
    csv_path = f"/scratch/anirudhkaushik/SYNBADD/{save_directory}/{run_name}_validate.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Validation metrics saved to {csv_path}")

    print("\n\nValidation Loss: {:.4f}".format(avg_val_loss), " Validation Align Loss: {:.4f}".format(avg_ssim_loss), " Validation Mean Jacobian: {:.4f}".format(avg_mean_jacobian), " Validation Std Jacobian: {:.4f}".format(avg_std_jacobian), " Validation PSNR: {:.4f}".format(avg_psnr), " Validation SSIM: {:.4f}".format(avg_ssim), " Validation MSE: {:.4f}".format(avg_mse), "Validation LDDMM Loss: {:.4f}".format(avg_jacobian), " Validation NRMSE: {:.4f}".format(avg_nrmse), " Validation Adv Loss: {:.4f}".format(avg_disc_adv_loss), end="\n\n")

    if not opt.debug:
        wandb.log(avg_metrics)
                
    return avg_val_loss

def compute_epe(target_svf, synthetic_field):
    """
    Compute the End-Point Error (EPE) between target SVF and synthetic field.
    EPE is the L2 norm of the difference between predicted and ground truth fields.
    
    Args:
        target_svf: Target stationary velocity field [B, C, H, W, D]
        synthetic_field: Predicted stationary velocity field [B, C, H, W, D]
    
    Returns:
        Mean EPE value across batch and spatial dimensions
    """
    diff = target_svf - synthetic_field
    squared_diff = diff ** 2
    squared_magnitude = squared_diff.sum(dim=1)  # Sum across channels
    epe = torch.sqrt(squared_magnitude)  # L2 norm
    return epe.mean().item()  # Mean across batch and spatial dimensions

def test(generator, vae_model, regression_model, transformer_model, discriminator_model, test_dataloader, criterion_lddmm, device, opt, run_name):
    project_name = opt.project_name
    save_directory = project_name + "/test"
    os.makedirs("/scratch/anirudhkaushik/SYNBADD/" + save_directory, exist_ok=True)
    
    generator.eval()
    vae_model.eval()
    if opt.use_transformer:
        transformer_model.eval()
    else:
        regression_model.eval()
    if opt.use_discriminator:
        discriminator_model.eval()
    
    test_loss = 0
    ssim_values, mean_jacobian_losses, std_jacobian_losses, mse_losses, psnr_losses, ssim_losses, avg_jacobian_losses, nrmse_losses, mssim_values = [], [], [], [], [], [], [], [], []
    disc_adv_losses = []
    epe_x_losses, epe_y_losses, epe_z_losses, angular_error_losses = [], [], [], []
    
    # Define timepoint mapping
    timepoints = ['sc', 'm06', 'm12', 'm18', 'm24', 'm36']
    timepoint_map = {i: tp for i, tp in enumerate(timepoints)}
    
    # Define class mapping
    group_map = {0.0: 'CN', 1.0: 'MCI', 2.0: 'AD'}
    
    # Create dictionaries to store metrics for each timepoint
    timepoint_metrics = {tp: {
        'ssim_losses': [], 'mean_jacobian_losses': [], 'std_jacobian_losses': [],
        'mse_losses': [], 'psnr_losses': [], 'ssim_values': [], 'mssim_values': [],
        'avg_jacobian_losses': [], 'nrmse_losses': [], 'disc_adv_losses': [],
        'epe_x_losses': [], 'epe_y_losses': [], 'epe_z_losses': [], 'angular_error_losses': [],
        'epe_losses': []  # Added EPE for each timepoint
    } for tp in timepoints}
    
    # Create dictionaries to store metrics for each class
    class_metrics = {cls: {
        'ssim_losses': [], 'mean_jacobian_losses': [], 'std_jacobian_losses': [],
        'mse_losses': [], 'psnr_losses': [], 'ssim_values': [], 'mssim_values': [],
        'avg_jacobian_losses': [], 'nrmse_losses': [], 'disc_adv_losses': [],
        'epe_x_losses': [], 'epe_y_losses': [], 'epe_z_losses': [], 'angular_error_losses': []
    } for cls in group_map.values()}

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
                    
                    # with torch.no_grad():
                    #     contextualized_scan, logvar, skips = vae_model(stacked_svf, "encode", None)
                    #     contextualized_scan, logvar, skips = contextualized_scan.detach(), logvar.detach(), [s.detach() for s in skips]

                    # # take mean of skips which is a list
                    # for skip_ind in range(len(skips)):
                    #     s_b, s_c, s_h, s_w, s_d = skips[skip_ind].shape
                    #     skips[skip_ind] = skips[skip_ind].view(batch_size, temporal_dim, s_c, s_h, s_w, s_d)
                    #     # average each across the temporal dimension
                    #     skips[skip_ind] = skips[skip_ind].mean(dim=1)
                    
                    # contextualized_scan = contextualized_scan.view(batch_size, temporal_dim, -1)
                    # if opt.use_transformer:
                    #     contextualized_scan = transformer_model(contextualized_scan, target_timepoint_index)
                    # else:
                    #     contextualized_scan = regression_model(contextualized_scan)
                    # with torch.no_grad():
                    #     contextualized_scan = vae_model(contextualized_scan, "decode", skips)

                    # make stacked_svf normal sized again
                    stacked_svf = stacked_svf.view(batch_size, temporal_dim, c_dim, h, w, d)
                    # [B*T, C, H, W, D] -> [B*T, 512]

                else:
                    contextualized_scan = latest_scan
                # do averaged imputation instead of generation
                contextualized_scan = stacked_svf.mean(dim=1) # take average along temporal dimension
                y_synthetic, synthetic_field = generator(latest_scan, contextualized_scan, target_labels)
                
                # Compute channelwise EPE loss and angular error
                _, epe_x, epe_y, epe_z = channelwise_epe_loss(synthetic_field, target_svf)
                angular_error = angular_error_loss(synthetic_field, target_svf)
                
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
                    total_loss = opt.ssim_coeff*ssim_loss + opt.disc_coeff*d_adv + opt.jacobian_coeff * neg_jacobian_loss + opt.smooth_coeff*smooth_loss + opt.jac_dist_coeff*jac_dist_loss + opt.lddmm_coeff*lddmm_loss + opt.epe_x_coeff*epe_x + opt.epe_y_coeff*epe_y + opt.epe_z_coeff*epe_z + opt.angular_coeff*angular_error
                else:
                    total_loss = opt.ssim_coeff*ssim_loss + opt.jacobian_coeff*neg_jacobian_loss + opt.smooth_coeff*smooth_loss + opt.jac_dist_coeff*jac_dist_loss + opt.lddmm_coeff*lddmm_loss + opt.epe_x_coeff*epe_x + opt.epe_y_coeff*epe_y + opt.epe_z_coeff*epe_z + opt.angular_coeff*angular_error
                
                test_loss += total_loss.item()

                mse_loss, psnr_value, ssim_value, mean_jacobian, std_jacobian, _, nrmse_loss, mssim_value = compute_metrics(target_scan, y_synthetic, synthetic_field, criterion_lddmm, device)
                jacobian_det = jacobian_determinant_torch(synthetic_field)  # shape: (B, H-1, W-1, D-1)
                target_jacobian_det = jacobian_determinant_torch(target_svf)  # shape: (B, H-1, W-1, D-1)

                # Store metrics for overall aggregation
                ssim_losses.append(ssim_loss.item())
                mean_jacobian_losses.append(mean_jacobian)
                std_jacobian_losses.append(std_jacobian)
                mse_losses.append(mse_loss)
                psnr_losses.append(psnr_value)
                ssim_values.append(ssim_value)
                mssim_values.append(mssim_value)
                avg_jacobian_losses.append(lddmm_loss.item())
                nrmse_losses.append(nrmse_loss)
                disc_adv_losses.append(d_adv.item())
                epe_x_losses.append(epe_x.item())
                epe_y_losses.append(epe_y.item())
                epe_z_losses.append(epe_z.item())
                angular_error_losses.append(angular_error.item())

                # Store metrics by timepoint and class
                for b in range(target_timepoint.size(0)):
                    tp_idx = target_timepoint_index[b].item()
                    tp = timepoint_map[tp_idx+1]
                    cls = group_map[class_labels[b].item()]
                    
                    # Get individual sample metrics
                    sample_ssim_loss = ssim_loss.item()
                    sample_mean_jacobian = mean_jacobian
                    sample_std_jacobian = std_jacobian
                    sample_mse_loss = mse_loss
                    sample_psnr_value = psnr_value
                    sample_ssim_value = ssim_value
                    sample_mssim_value = mssim_value
                    sample_jacobian_loss = lddmm_loss.item()
                    sample_nrmse_loss = nrmse_loss
                    sample_d_adv = d_adv.item()
                    sample_epe_x = epe_x.item()
                    sample_epe_y = epe_y.item()
                    sample_epe_z = epe_z.item()
                    sample_angular_error = angular_error.item()
                    
                    # Store in timepoint-specific lists
                    timepoint_metrics[tp]['ssim_losses'].append(sample_ssim_loss)
                    timepoint_metrics[tp]['mean_jacobian_losses'].append(sample_mean_jacobian)
                    timepoint_metrics[tp]['std_jacobian_losses'].append(sample_std_jacobian)
                    timepoint_metrics[tp]['mse_losses'].append(sample_mse_loss)
                    timepoint_metrics[tp]['psnr_losses'].append(sample_psnr_value)
                    timepoint_metrics[tp]['ssim_values'].append(sample_ssim_value)
                    timepoint_metrics[tp]['mssim_values'].append(sample_mssim_value)
                    timepoint_metrics[tp]['avg_jacobian_losses'].append(sample_jacobian_loss)
                    timepoint_metrics[tp]['nrmse_losses'].append(sample_nrmse_loss)
                    timepoint_metrics[tp]['disc_adv_losses'].append(sample_d_adv)
                    timepoint_metrics[tp]['epe_x_losses'].append(sample_epe_x)
                    timepoint_metrics[tp]['epe_y_losses'].append(sample_epe_y)
                    timepoint_metrics[tp]['epe_z_losses'].append(sample_epe_z)
                    timepoint_metrics[tp]['angular_error_losses'].append(sample_angular_error)
                    
                    # Store in class-specific lists
                    class_metrics[cls]['ssim_losses'].append(sample_ssim_loss)
                    class_metrics[cls]['mean_jacobian_losses'].append(sample_mean_jacobian)
                    class_metrics[cls]['std_jacobian_losses'].append(sample_std_jacobian)
                    class_metrics[cls]['mse_losses'].append(sample_mse_loss)
                    class_metrics[cls]['psnr_losses'].append(sample_psnr_value)
                    class_metrics[cls]['ssim_values'].append(sample_ssim_value)
                    class_metrics[cls]['mssim_values'].append(sample_mssim_value)
                    class_metrics[cls]['avg_jacobian_losses'].append(sample_jacobian_loss)
                    class_metrics[cls]['nrmse_losses'].append(sample_nrmse_loss)
                    class_metrics[cls]['disc_adv_losses'].append(sample_d_adv)
                    class_metrics[cls]['epe_x_losses'].append(sample_epe_x)
                    class_metrics[cls]['epe_y_losses'].append(sample_epe_y)
                    class_metrics[cls]['epe_z_losses'].append(sample_epe_z)
                    class_metrics[cls]['angular_error_losses'].append(sample_angular_error)

                progress_bar.update(stacked_svf.size(0))
                progress_bar.set_postfix({
                    "Test SSIM Loss": ssim_loss.item(),
                    "Test LDDMM Loss": lddmm_loss.item(),
                    "Test Mse": mse_loss,
                    "Test PSNR": psnr_value,
                    "Test SSIM": ssim_value,
                    "Test MS_SSIM": mssim_value,
                    "Test Total Loss": total_loss.item(),
                    "Test Mean Jac": mean_jacobian,
                    "Test Std Jac": std_jacobian,
                    "Test NRMSE": nrmse_loss,
                    "Test Adv Loss": d_adv.item(),
                    "Test EPE X": epe_x.item(),
                    "Test EPE Y": epe_y.item(),
                    "Test EPE Z": epe_z.item(),
                    "Test Angular": angular_error.item(),
                })
                
                # Save visualization for a few test samples
                if i%10 == 0:
                    loss_map = {
                        "mse": mse_loss,
                        "psnr": psnr_value,
                        "ssim": ssim_value,
                        "mssim": mssim_value,
                        "mean jacobian": mean_jacobian, 
                        "std jacobian": std_jacobian,
                        "epe_x": epe_x.item(),
                        "epe_y": epe_y.item(),
                        "epe_z": epe_z.item(),
                        "angular": angular_error.item()
                    }

                    fig = plot_slices(latest_scan, target_scan, y_synthetic, jacobian_det, target_jacobian_det, loss_map,
                                     reg_labels[0].item(), group_map[class_labels[0].item()], target_timepoint[0].item(), latest_scan_timepoint[0].item())
                    
                    save_path = f"/scratch/anirudhkaushik/SYNBADD/{save_directory}/test_viz_{run_name}_{i}.png"
                    plt.savefig(save_path)
                    plt.close(fig)
                    
                    if not opt.debug:
                        wandb.log({f"Test Sample": wandb.Image(fig)})
    
    # Calculate average metrics across all samples
    avg_test_loss = test_loss / len(test_dataloader)
    avg_ssim_loss = np.mean(ssim_losses)
    avg_mean_jacobian = np.mean(mean_jacobian_losses)
    avg_std_jacobian = np.mean(std_jacobian_losses)
    avg_psnr = np.mean(psnr_losses)
    avg_ssim = np.mean(ssim_values)
    avg_mssim = np.mean(mssim_values)
    avg_mse = np.mean(mse_losses)
    avg_jacobian = np.mean(avg_jacobian_losses)
    avg_nrmse = np.mean(nrmse_losses)
    avg_disc_adv_loss = np.mean(disc_adv_losses)
    avg_epe_x = np.mean(epe_x_losses)
    avg_epe_y = np.mean(epe_y_losses)
    avg_epe_z = np.mean(epe_z_losses)
    avg_angular_error = np.mean(angular_error_losses)
    
    # Calculate timepoint-specific averages
    timepoint_averages = {}
    for tp in timepoints:
        if len(timepoint_metrics[tp]['ssim_losses']) > 0:  # Only calculate if there are samples for this timepoint
            timepoint_averages[tp] = {
                f"Test SSIM Loss {tp}": np.mean(timepoint_metrics[tp]['ssim_losses']),
                f"Test Mean Jacobian {tp}": np.mean(timepoint_metrics[tp]['mean_jacobian_losses']),
                f"Test Std Jacobian {tp}": np.mean(timepoint_metrics[tp]['std_jacobian_losses']),
                f"Test PSNR {tp}": np.mean(timepoint_metrics[tp]['psnr_losses']),
                f"Test SSIM {tp}": np.mean(timepoint_metrics[tp]['ssim_values']),
                f"Test MS_SSIM {tp}": np.mean(timepoint_metrics[tp]['mssim_values']),
                f"Test MSE {tp}": np.mean(timepoint_metrics[tp]['mse_losses']),
                f"Test LDDMM Loss {tp}": np.mean(timepoint_metrics[tp]['avg_jacobian_losses']),
                f"Test NRMSE {tp}": np.mean(timepoint_metrics[tp]['nrmse_losses']),
                f"Test Adv Loss {tp}": np.mean(timepoint_metrics[tp]['disc_adv_losses']),
                f"Test EPE X {tp}": np.mean(timepoint_metrics[tp]['epe_x_losses']),
                f"Test EPE Y {tp}": np.mean(timepoint_metrics[tp]['epe_y_losses']),
                f"Test EPE Z {tp}": np.mean(timepoint_metrics[tp]['epe_z_losses']),
                f"Test Angular Error {tp}": np.mean(timepoint_metrics[tp]['angular_error_losses']),
            }
    
    # Calculate class-specific averages
    class_averages = {}
    for cls in group_map.values():
        if len(class_metrics[cls]['ssim_losses']) > 0:  # Only calculate if there are samples for this class
            class_averages[cls] = {
                f"Test SSIM Loss {cls}": np.mean(class_metrics[cls]['ssim_losses']),
                f"Test Mean Jacobian {cls}": np.mean(class_metrics[cls]['mean_jacobian_losses']),
                f"Test Std Jacobian {cls}": np.mean(class_metrics[cls]['std_jacobian_losses']),
                f"Test PSNR {cls}": np.mean(class_metrics[cls]['psnr_losses']),
                f"Test SSIM {cls}": np.mean(class_metrics[cls]['ssim_values']),
                f"Test MS_SSIM {cls}": np.mean(class_metrics[cls]['mssim_values']),
                f"Test MSE {cls}": np.mean(class_metrics[cls]['mse_losses']),
                f"Test LDDMM Loss {cls}": np.mean(class_metrics[cls]['avg_jacobian_losses']),
                f"Test NRMSE {cls}": np.mean(class_metrics[cls]['nrmse_losses']),
                f"Test Adv Loss {cls}": np.mean(class_metrics[cls]['disc_adv_losses']),
                f"Test EPE X {cls}": np.mean(class_metrics[cls]['epe_x_losses']),
                f"Test EPE Y {cls}": np.mean(class_metrics[cls]['epe_y_losses']),
                f"Test EPE Z {cls}": np.mean(class_metrics[cls]['epe_z_losses']),
                f"Test Angular Error {cls}": np.mean(class_metrics[cls]['angular_error_losses']),
            }
    
    # Overall average metrics
    avg_metrics = {
        "Test Loss": avg_test_loss, 
        "Test SSIM Loss": avg_ssim_loss,
        "Test Mean Jacobian": avg_mean_jacobian,
        "Test Std Jacobian": avg_std_jacobian,
        "Test PSNR": avg_psnr,
        "Test SSIM": avg_ssim,
        "Test MS_SSIM": avg_mssim,
        "Test MSE": avg_mse,
        "Test LDDMM Loss": avg_jacobian,
        "Test NRMSE": avg_nrmse,
        "Test Adv Loss": avg_disc_adv_loss,
        "Test EPE X": avg_epe_x,
        "Test EPE Y": avg_epe_y,
        "Test EPE Z": avg_epe_z,
        "Test Angular Error": avg_angular_error,
    }
    
    # Combine overall averages with timepoint-specific and class-specific averages for CSV
    all_metrics = avg_metrics.copy()
    for tp, tp_metrics in timepoint_averages.items():
        all_metrics.update(tp_metrics)
    for cls, cls_metrics in class_averages.items():
        all_metrics.update(cls_metrics)
    
    # Convert to DataFrame with just one row
    df = pd.DataFrame([all_metrics])
    csv_path = f"/scratch/anirudhkaushik/SYNBADD/{save_directory}/{run_name}_test.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Test metrics saved to {csv_path}")
    
    # Print overall averages
    print("\n\nTest Loss: {:.4f}".format(avg_test_loss), 
          " Test Align Loss: {:.4f}".format(avg_ssim_loss), 
          " Test Mean Jacobian: {:.4f}".format(avg_mean_jacobian), 
          " Test Std Jacobian: {:.4f}".format(avg_std_jacobian), 
          " Test PSNR: {:.4f}".format(avg_psnr), 
          " Test SSIM: {:.4f}".format(avg_ssim), 
          " Test MS_SSIM: {:.4f}".format(avg_mssim),
          " Test MSE: {:.4f}".format(avg_mse), 
          "Test LDDMM Loss: {:.4f}".format(avg_jacobian), 
          " Test NRMSE: {:.4f}".format(avg_nrmse), 
          " Test Adv Loss: {:.4f}".format(avg_disc_adv_loss),
          " Test EPE X: {:.4f}".format(avg_epe_x),
          " Test EPE Y: {:.4f}".format(avg_epe_y),
          " Test EPE Z: {:.4f}".format(avg_epe_z),
          " Test Angular Error: {:.4f}".format(avg_angular_error),
          end="\n\n")
    
    # Print timepoint-specific metrics
    print("\n=== Timepoint-Specific Metrics ===")
    for tp in timepoints:
        if tp in timepoint_averages:
            print(f"\nTimepoint {tp} Metrics:")
            tp_metrics = timepoint_averages[tp]
            print(f"  Test SSIM Loss: {tp_metrics[f'Test SSIM Loss {tp}']:.4f}", 
                  f" Test Mean Jacobian: {tp_metrics[f'Test Mean Jacobian {tp}']:.4f}", 
                  f" Test Std Jacobian: {tp_metrics[f'Test Std Jacobian {tp}']:.4f}", 
                  f" Test PSNR: {tp_metrics[f'Test PSNR {tp}']:.4f}", 
                  f" Test SSIM: {tp_metrics[f'Test SSIM {tp}']:.4f}", 
                  f" Test MS_SSIM: {tp_metrics[f'Test MS_SSIM {tp}']:.4f}",
                  f" Test MSE: {tp_metrics[f'Test MSE {tp}']:.4f}", 
                  f" Test LDDMM Loss: {tp_metrics[f'Test LDDMM Loss {tp}']:.4f}", 
                  f" Test NRMSE: {tp_metrics[f'Test NRMSE {tp}']:.4f}", 
                  f" Test Adv Loss: {tp_metrics[f'Test Adv Loss {tp}']:.4f}")
            print(f"  Test EPE X: {tp_metrics[f'Test EPE X {tp}']:.4f}",
                  f" Test EPE Y: {tp_metrics[f'Test EPE Y {tp}']:.4f}",
                  f" Test EPE Z: {tp_metrics[f'Test EPE Z {tp}']:.4f}",
                  f" Test Angular Error: {tp_metrics[f'Test Angular Error {tp}']:.4f}")
    
    # Print class-specific metrics
    print("\n=== Class-Specific Metrics ===")
    for cls in group_map.values():
        if cls in class_averages:
            print(f"\nClass {cls} Metrics:")
            cls_metrics = class_averages[cls]
            print(f"  Test SSIM Loss: {cls_metrics[f'Test SSIM Loss {cls}']:.4f}", 
                  f" Test Mean Jacobian: {cls_metrics[f'Test Mean Jacobian {cls}']:.4f}", 
                  f" Test Std Jacobian: {cls_metrics[f'Test Std Jacobian {cls}']:.4f}", 
                  f" Test PSNR: {cls_metrics[f'Test PSNR {cls}']:.4f}", 
                  f" Test SSIM: {cls_metrics[f'Test SSIM {cls}']:.4f}", 
                  f" Test MS_SSIM: {cls_metrics[f'Test MS_SSIM {cls}']:.4f}",
                  f" Test MSE: {cls_metrics[f'Test MSE {cls}']:.4f}", 
                  f" Test LDDMM Loss: {cls_metrics[f'Test LDDMM Loss {cls}']:.4f}", 
                  f" Test NRMSE: {cls_metrics[f'Test NRMSE {cls}']:.4f}", 
                  f" Test Adv Loss: {cls_metrics[f'Test Adv Loss {cls}']:.4f}")
            print(f"  Test EPE X: {cls_metrics[f'Test EPE X {cls}']:.4f}",
                  f" Test EPE Y: {cls_metrics[f'Test EPE Y {cls}']:.4f}",
                  f" Test EPE Z: {cls_metrics[f'Test EPE Z {cls}']:.4f}",
                  f" Test Angular Error: {cls_metrics[f'Test Angular Error {cls}']:.4f}")
    
    # Log overall metrics to wandb
    if not opt.debug:
        wandb_log_data = avg_metrics.copy()
        
        # Add timepoint-specific and class-specific metrics to wandb
        for tp in timepoints:
            if tp in timepoint_averages:
                wandb_log_data.update(timepoint_averages[tp])
        for cls in group_map.values():
            if cls in class_averages:
                wandb_log_data.update(class_averages[cls])
        
        # Log all metrics at once
        wandb.log(wandb_log_data)
        
        # Create and log a summary table with timepoint metrics
        if len(timepoint_averages) > 0:
            # Create data for the table
            table_data = []
            metrics_to_include = ["SSIM", "MS_SSIM", "PSNR", "MSE", "Mean Jacobian", "NRMSE", "EPE X", "EPE Y", "EPE Z", "Angular Error"]
            
            for tp in timepoints:
                if tp in timepoint_averages:
                    row = {"Timepoint": tp}
                    for metric in metrics_to_include:
                        key = f"Test {metric} {tp}"
                        if key in timepoint_averages[tp]:
                            row[metric] = timepoint_averages[tp][key]
                    table_data.append(row)
            
            # Log the table to wandb
            columns = ["Timepoint"] + metrics_to_include
            wandb_table = wandb.Table(columns=columns)
            
            for row in table_data:
                wandb_table.add_data(
                    row["Timepoint"],
                    *[row.get(metric, 0) for metric in metrics_to_include]
                )
            
            wandb.log({"Timepoint Metrics": wandb_table})
        
        # Create and log a summary table with class metrics
        if len(class_averages) > 0:
            # Create data for the table
            class_table_data = []
            metrics_to_include = ["SSIM", "MS_SSIM", "PSNR", "MSE", "Mean Jacobian", "NRMSE", "EPE X", "EPE Y", "EPE Z", "Angular Error"]
            
            for cls in group_map.values():
                if cls in class_averages:
                    row = {"Class": cls}
                    for metric in metrics_to_include:
                        key = f"Test {metric} {cls}"
                        if key in class_averages[cls]:
                            row[metric] = class_averages[cls][key]
                    class_table_data.append(row)
            
            # Log the table to wandb
            columns = ["Class"] + metrics_to_include
            wandb_class_table = wandb.Table(columns=columns)
            
            for row in class_table_data:
                wandb_class_table.add_data(
                    row["Class"],
                    *[row.get(metric, 0) for metric in metrics_to_include]
                )
            
            wandb.log({"Class Metrics": wandb_class_table})
    
    return avg_test_loss