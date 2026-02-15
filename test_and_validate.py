# =============================================================================
# File: test_and_validate.py
# Purpose: Inference/evaluation entrypoint for a specific experiment variant.
# Prerequisites: Trained checkpoints and fold-specific dataset metadata.
# Usually Run Next: Run comparison analysis scripts or aggregate fold metrics.
# Model Components Used: Uses model modules imported in this file; latest subjectwise artifact generation uses test_model_subjectwise.py.
# =============================================================================

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from matplotlib import pyplot as plt

from utils import plot_slices
from losses import compute_metrics, smooth_loss_l2


def validate(generator, vivit_model, val_dataloader, criterion_jacobian, device, opt):
    generator.eval()
    vivit_model.eval()
    
    val_loss = 0
    with torch.no_grad():
        with tqdm(total=len(val_dataloader.dataset), desc="Validation") as progress_bar:
            for i, batch in enumerate(val_dataloader):
                stacked_scan, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint = [batch[k].to(device) for k in ["stacked_scan", "latest_scan", "target_scan", "class_label", "reg_label", "target_timepoint", "latest_timepoint"]]
                x_real, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint = [Variable(v.type(torch.float32)).to(device) for v in [stacked_scan, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint]]
                target_labels = torch.cat([latest_scan_timepoint, class_labels, reg_labels, target_timepoint], dim=1)

                if opt.use_past_data:
                    contextualized_scan = vivit_model(x_real)
                else:
                    contextualized_scan = latest_scan

                y_synthetic, synthetic_field = generator(latest_scan, contextualized_scan, target_labels)

                align_loss = smooth_loss_l2(y_synthetic, target_scan)
                jacobian_loss = criterion_jacobian(target_scan, y_synthetic)

                val_loss = align_loss + opt.jacbian_coeff*jacobian_loss
                val_loss += val_loss.item()

                mse_loss, psnr_value, ssim_value, mean_jacobian, std_jacobian, jacobian_det = compute_metrics(target_scan, y_synthetic, criterion_jacobian, device)

                
                progress_bar.update(x_real.size(0))
                progress_bar.set_postfix({
                    "Smooth l12 Loss": align_loss.item(),
                    "Mse": mse_loss,
                    "PSNR": psnr_value,
                    "SSIM": ssim_value,
                    "Total Loss": val_loss.item(),
                    "Mean Jac": mean_jacobian,
                    "Std Jac": std_jacobian,
                })
                
    return val_loss / len(val_dataloader)

def test(generator, vivit_model, diffeo_reg, test_dataloader, criterion_jacobian, device, opt, run_name):
    generator.eval()
    vivit_model.eval()
    
    test_loss = 0
    align_losses, mean_jacobian_losses, std_jacobian_losses, mse_losses, psnr_losses, ssim_losses = [], [], [], [], [], []

    
    with torch.no_grad():
        with tqdm(total=len(test_dataloader.dataset), desc="Testing") as progress_bar:
             for i, batch in enumerate(test_dataloader):
                stacked_scan, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint = [batch[k].to(device) for k in ["stacked_scan", "latest_scan", "target_scan", "class_label", "reg_label", "target_timepoint", "latest_timepoint"]]
                x_real, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint = [Variable(v.type(torch.float32)).to(device) for v in [stacked_scan, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint]]
                target_labels = torch.cat([latest_scan_timepoint, class_labels, reg_labels, target_timepoint], dim=1)

                if opt.use_past_data:
                    contextualized_scan = vivit_model(x_real)
                else:
                    contextualized_scan = latest_scan

                y_synthetic, synthetic_field = generator(latest_scan, contextualized_scan, target_labels)
                
                align_loss = smooth_loss_l2(y_synthetic, target_scan)
                jacobian_loss = criterion_jacobian(target_scan, y_synthetic)

                test_loss = align_loss + opt.jacbian_coeff*jacobian_loss

                mse_loss, psnr_value, ssim_value, mean_jacobian, std_jacobian, jacobian_det = compute_metrics(target_scan, y_synthetic, criterion_jacobian, device)

                align_losses.append(align_loss.item())
                mean_jacobian_losses.append(mean_jacobian)
                std_jacobian_losses.append(std_jacobian)
                mse_losses.append(mse_loss)
                psnr_losses.append(psnr_value)
                ssim_losses.append(ssim_value)

                progress_bar.update(x_real.size(0))
                progress_bar.set_postfix({
                    "Smooth l12 Loss": align_loss.item(),
                    "Mse": mse_loss,
                    "PSNR": psnr_value,
                    "SSIM": ssim_value,
                    "Total Loss": test_loss.item(),
                    "Mean Jac": mean_jacobian,
                    "Std Jac": std_jacobian,
                })
                
                # Save visualization for a few test samples
                if i % 20 == 0:
                    group_map = {0.0: 'CN', 1.0: 'MCI', 2.0: 'AD'}
                    fig = plot_slices(latest_scan, target_scan, y_synthetic, jacobian_det, align_loss.item(), mean_jacobian,
                                     reg_labels[0].item(), group_map[class_labels[0].item()], target_timepoint[0].item(), latest_scan_timepoint[0].item())
                    
                    save_path = f"/scratch/anirudhkaushik/SYNBADD/test_viz_{run_name}_{i}.png"
                    plt.savefig(save_path)
                    plt.close(fig)
                    
                    if not opt.debug:
                        wandb.log({f"Test Sample {i}": wandb.Image(fig)})
                
    avg_test_loss = test_loss / len(test_dataloader)
    avg_align_loss = np.mean(align_losses)
    avg_mean_jacobian = np.mean(mean_jacobian_losses)
    avg_std_jacobian = np.mean(std_jacobian_losses)
    avg_psnr = np.mean(psnr_losses)
    avg_ssim = np.mean(ssim_losses)
    avg_mse = np.mean(mse_losses)

    
    print("\n\nTest Loss: {:.4f}".format(avg_test_loss), " Test Align Loss: {:.4f}".format(avg_align_loss), " Test Mean Jacobian: {:.4f}".format(avg_mean_jacobian), " Test Std Jacobian: {:.4f}".format(avg_std_jacobian), " Test PSNR: {:.4f}".format(avg_psnr), " Test SSIM: {:.4f}".format(avg_ssim), " Test MSE: {:.4f}".format(avg_mse), end="\n\n")
    
    if not opt.debug:
        wandb.log({
            "Test Loss": test_loss, 
            "Test Smooth L12 Loss": avg_align_loss,
            "Test Mean Jacobian": avg_mean_jacobian,
            "Test Std Jacobian": avg_std_jacobian,
            "Test PSNR": avg_psnr,
            "Test SSIM": avg_ssim,
            "Test MSE": avg_mse
        })
    
    return avg_test_loss