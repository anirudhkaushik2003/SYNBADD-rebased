# =============================================================================
# File: test_model_subjectwise.py
# Purpose: Loads pretrained checkpoints and runs fold-wise test inference to save synthetic scans and synthetic fields per subject in /scratch/anirudhkaushik/SYNBADD_Qualitative_<fold>/<run_name>/.
# Prerequisites: Completed checkpoints from train_VAE_SYNBADD_svf.py in /scratch/anirudhkaushik/SYNBADD/<project_name>/ and ADNI metadata/SVF files in /scratch.
# Usually Run Next: Run adni_ablations_comparision_analysis.py for model-variant comparisons or adni_baseline_comparision_analysis.py / adni_remind_comparison.py for SOTA baseline comparisons.
# Model Components Used: SVFRegressorDiffeoGenerator + VAE encoder/decoder (get_vae_model) + transformer variant selected by --transformer_subtype (default here: GatedCrossAttention) or LatentRegressor fallback, with optional DAFTDiscriminator loading.
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

from datasets import  SVFUltraMaskedSequenceDataset
from generator import SVFRegressorDiffeoGenerator
from losses import compute_metrics, smooth_loss_l2, ssim_only_loss, jacobian_determinant_torch
from utils import get_data_loaders, str2bool, plot_slices, get_run_name, get_optimizers_regressor, get_optimizers_generator, get_optimizers_transformer, get_vae_model
from discriminator import DAFTDiscriminator
from models import LatentRegressor
from transformer_model import TransformerBidirectional, GatedCrossAttentionTransformer, GatedTransformerModel, TransformerModel, CrossAttentionGatedTransformer, CrossAttentionTransformer

random.seed(1337)

import torch.distributed as dist
import torch.multiprocessing as mp

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



def evaluate(opt):
   
    run_name = opt.run_name
    project_name = opt.project_name

    if not opt.debug:

        wandb.init(project=f"{project_name}_Subjectwise", name=run_name)
        wandb.config.update(vars(opt))

    image_size = (opt.img_height, opt.img_height, opt.img_height) 

    save_directory = project_name

    print(f"Image Size: {image_size}, Number of Transformer Layers: {opt.num_layers}, Sequence Length: {opt.seq_len}\n")
    os.makedirs(f"/scratch/anirudhkaushik/SYNBADD/{save_directory}", exist_ok=True)

    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.float32)

    lambda_gp = 1
    accumulation_steps = opt.effective_batch_size // opt.batch_size

    nb_features = [
        [16, 32, 32, 32, 32],
        [32, 32, 32, 32, 32, 32, 16, 16]
    ]

    generator = SVFRegressorDiffeoGenerator(inshape=image_size,
                                latent_input_dim=512,
                                c_dim=opt.c_dim,
                                use_probs=False,
                                int_downsize=1,
                                in_channels=opt.in_channels,
                                int_steps=12,
                                nb_unet_features=nb_features,
                                using_svf=opt.use_deformation_field,
                                using_daft=opt.using_daft).to(device)
    
    # Spatial SVF Encoder
    vae_model = get_vae_model(opt, device=device)

    # Temporal Encoders
    transformer_model, regression_model = None, None
    if opt.use_transformer:
        if opt.transformer_subtype == "GatedCrossAttention":
            print(f"\nUsing GatedCrossAttentionTransformer\n")
            transformer_model = GatedCrossAttentionTransformer(input_dim=512, num_heads=8, num_layers=opt.num_layers, max_seq_len=opt.seq_len, use_positional_embeddings=opt.use_positional_embeddings).to(device)
        elif opt.transformer_subtype == "GatedTransformerModel":
            print(f"\nUsing GatedTransformerModel\n")
            transformer_model = GatedTransformerModel(input_dim=512, num_heads=8, num_layers=opt.num_layers, max_seq_len=opt.seq_len).to(device)
        elif opt.transformer_subtype == "TransformerModel":
            print(f"\nUsing Vanilla TransformerModel\n")
            transformer_model = TransformerModel(input_dim=512, num_heads=8, num_layers=opt.num_layers, max_seq_len=opt.seq_len).to(device)
        elif opt.transformer_subtype == "CrossAttentionTransformer":
            print(f"\nUsing CrossAttentionTransformer\n")
            transformer_model = CrossAttentionTransformer(input_dim=512, num_heads=8, num_layers=opt.num_layers, max_seq_len=opt.seq_len).to(device)
        elif opt.transformer_subtype == "CrossAttentionGatedTransformer":
            print(f"\nUsing CrossAttentionGatedTransformer\n")
            transformer_model = CrossAttentionGatedTransformer(input_dim=512, num_heads=8, num_layers=opt.num_layers, max_seq_len=opt.seq_len).to(device)
        elif opt.transformer_subtype == "BidirectionalTransformer":
            print(f"\nUsing BidirectionalTransformer\n")
            transformer_model = TransformerBidirectional(input_dim=512, num_heads=8, num_layers=opt.num_layers, max_seq_len=opt.seq_len, use_positional_embeddings=opt.use_positional_embeddings, use_bidirectional_embeddings=opt.use_bidirectional_embeddings).to(device)
    else:
        regression_model = LatentRegressor(seq_len=opt.seq_len)
    discriminator_model = None
    if opt.use_discriminator:
        discriminator_model = DAFTDiscriminator(in_channels=1, ndim_non_img=opt.c_dim)

    if opt.n_gpu > 1:
        generator = nn.DataParallel(generator)
        vae_model = nn.DataParallel(vae_model)
        if opt.use_transformer:
            transformer_model = nn.DataParallel(transformer_model)
        else:
            regression_model = nn.DataParallel(regression_model)
        if opt.use_discriminator:
            discriminator_model = nn.DataParallel(discriminator_model)

    generator = generator.to(device)
    vae_model = vae_model.to(device)
    if opt.use_transformer:
        transformer_model = transformer_model.to(device)
    else:
        regression_model = regression_model.to(device)
    if opt.use_discriminator:
        discriminator_model = discriminator_model.to(device)

    optimizer_G, scheduler_G, criterion_lddmm = get_optimizers_generator(opt, generator, device)
    
    # Temporal Encoders
    optimizer_regressor, scheduler_regressor = None, None
    optimizer_transformer, scheduler_transformer = None, None
    if opt.use_transformer:
        optimizer_transformer, scheduler_transformer = get_optimizers_transformer(opt, transformer_model, device=device)
    else:
        optimizer_regressor, scheduler_regressor = get_optimizers_regressor(opt, regression_model)
    
    optimizer_D, scheduler_D = None, None
    if opt.use_discriminator:
        optimizer_D = torch.optim.Adam(discriminator_model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=opt.patience, verbose=True)

    svf_file = "/scratch/anirudhkaushik/svf_metadata.csv"
    if opt.dataset == "ADNI":
        timepoints = ['sc', 'm06', 'm12', 'm18', 'm24', 'm36']
        csv_file = "/scratch/anirudhkaushik/ADNI1_Complete_DataSplits.csv"
    elif opt.dataset == "Toy":
        timepoints = ['sc', 'm06', 'm12', 'm18', 'm24', 'm30', 'm36']
        csv_file = "./metadata.csv"

    # Create train, validation, and test datasets

    # train_dataset = SVFUltraMaskedSequenceDataset(opt, csv_file, svf_file, subset='train', fold=opt.fold, aug_p=opt.aug_p, 
    #                                   timepoints=timepoints, target_timepoint=opt.target_timepoint, use_val_dataset=opt.use_val_dataset)
    # val_dataset = SVFUltraMaskedSequenceDataset(opt, csv_file, svf_file, subset='val', fold=opt.fold, aug_p=0.0, 
    #                                 timepoints=timepoints, target_timepoint=opt.target_timepoint, use_val_dataset=opt.use_val_dataset)
    test_dataset = SVFUltraMaskedSequenceDataset(opt, csv_file, svf_file, subset='test', fold=opt.fold, aug_p=0.0, 
                                     timepoints=timepoints, target_timepoint=opt.target_timepoint, use_val_dataset=opt.use_val_dataset)
    
    # get dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_data_loaders(opt, None, None, test_dataset)

    # Log dataset sizes
    if not opt.debug:
        wandb.log({
            "Train Size": 0,
            "Validation Size": 0,
            "Test Size": len(test_dataset)
        })

    best_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = opt.patience+2

    # Load best model for testing
    print("Loading best model for testing...")
    generator.load_state_dict(torch.load(f"/scratch/anirudhkaushik/SYNBADD/{save_directory}/{run_name}_SYNBADD_SVF.pth"))
    vae_model.load_state_dict(torch.load(f"/scratch/anirudhkaushik/SYNBADD/{save_directory}/{run_name}_vae_SYNBADD_SVF.pth"))
    
    if opt.use_transformer:
        transformer_model.load_state_dict(torch.load(f"/scratch/anirudhkaushik/SYNBADD/{save_directory}/{run_name}_transformer_SYNBADD_SVF.pth"))
    else:
        regression_model.load_state_dict(torch.load(f"/scratch/anirudhkaushik/SYNBADD/{save_directory}/{run_name}_regression_SYNBADD_SVF.pth"))
    if opt.use_discriminator:
        discriminator_model.load_state_dict(torch.load(f"/scratch/anirudhkaushik/SYNBADD/{save_directory}/{run_name}_discriminator_SYNBADD_SVF.pth"))
    
    # Test phase
    test_loss = test(generator, vae_model, regression_model, transformer_model, discriminator_model, test_dataloader, criterion_lddmm, device, opt, run_name)
    print(f"Final test loss: {test_loss:.4f}")
    
    if not opt.debug:
        wandb.log({"Final Test Loss": test_loss})
        wandb.finish()

def test(generator, vae_model, regression_model, transformer_model, discriminator_model, test_dataloader, criterion_lddmm, device, opt, run_name):
    project_name = opt.project_name
    save_directory = project_name + "/test"
    os.makedirs("/scratch/anirudhkaushik/SYNBADD/" + save_directory, exist_ok=True)

    all_scans_save_directory = f"/scratch/anirudhkaushik/SYNBADD_Qualitative_{opt.fold}/{opt.run_name}/"
    os.makedirs(all_scans_save_directory, exist_ok=True)
    
    uid_to_filepath = test_dataloader.dataset.get_file_path_mapping()

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
    epe_losses = []  # Added EPE losses list
    
    # Define timepoint mapping
    timepoints = ['sc', 'm06', 'm12', 'm18', 'm24', 'm36']
    timepoint_map = {i: tp for i, tp in enumerate(timepoints)}
    
    # Create dictionaries to store metrics for each timepoint
    timepoint_metrics = {tp: {
        'ssim_losses': [], 'mean_jacobian_losses': [], 'std_jacobian_losses': [],
        'mse_losses': [], 'psnr_losses': [], 'ssim_values': [], 'mssim_values': [],
        'avg_jacobian_losses': [], 'nrmse_losses': [], 'disc_adv_losses': [],
        'epe_losses': []  # Added EPE for each timepoint
    } for tp in timepoints}

    with torch.no_grad():
        with tqdm(total=len(test_dataloader.dataset), desc="Testing") as progress_bar:
             for i, batch in enumerate(test_dataloader):
                stacked_svf, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint, target_timepoint_index, target_svf, target_uid = [batch[k].to(device) for k in ["stacked_svf", "latest_scan", "target_scan", "class_label", "reg_label", "target_timepoint", "latest_timepoint", "target_timepoint_index", "target_svf", "target_uid"]]
                stacked_svf, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint, target_svf = [Variable(v.type(torch.float32)).to(device) for v in [stacked_svf, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint, target_svf]]
                target_labels = torch.cat([class_labels, reg_labels, target_timepoint], dim=1)

                if opt.use_past_data or opt.use_future_data:
                    # [B, T, C, H, W, D] Collapse batch dimension into time dimension
                    batch_size, temporal_dim, c_dim, h, w, d = stacked_svf.shape
                    stacked_svf = stacked_svf.view(-1, *stacked_svf.shape[2:])
                    
                    with torch.no_grad():
                        contextualized_scan, logvar, skips = vae_model.module.encode(stacked_svf)
                        contextualized_scan, logvar, skips = contextualized_scan.detach(), logvar.detach(), [s.detach() for s in skips]

                    # take mean of skips which is a list
                    for skip_ind in range(len(skips)):
                        s_b, s_c, s_h, s_w, s_d = skips[skip_ind].shape
                        skips[skip_ind] = skips[skip_ind].view(batch_size, temporal_dim, s_c, s_h, s_w, s_d)
                        # average each across the temporal dimension
                        skips[skip_ind] = skips[skip_ind].mean(dim=1)
                    
                    contextualized_scan = contextualized_scan.view(batch_size, temporal_dim, -1)
                    if opt.use_transformer:
                        contextualized_scan = transformer_model(contextualized_scan, target_timepoint_index)
                    else:
                        contextualized_scan = regression_model(contextualized_scan)
                    with torch.no_grad():
                        contextualized_scan = vae_model.module.decode(contextualized_scan, skips)

                    # make stacked_svf normal sized again
                    stacked_svf = stacked_svf.view(batch_size, temporal_dim, c_dim, h, w, d)
                    # [B*T, C, H, W, D] -> [B*T, 512]

                else:
                    contextualized_scan = latest_scan
                
                y_synthetic, synthetic_field = generator(latest_scan, contextualized_scan, target_labels)
                
                # Calculate EPE between target_svf and synthetic_field
                epe_loss = compute_epe(target_svf, synthetic_field)
                
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
                    total_loss = opt.ssim_coeff*ssim_loss + opt.disc_coeff*d_adv + opt.jacobian_coeff * neg_jacobian_loss + opt.smooth_coeff*smooth_loss + opt.jac_dist_coeff*jac_dist_loss + opt.lddmm_coeff*lddmm_loss
                else:
                    total_loss = opt.ssim_coeff*ssim_loss + opt.jacobian_coeff*neg_jacobian_loss + opt.smooth_coeff*smooth_loss + opt.jac_dist_coeff*jac_dist_loss + opt.lddmm_coeff*lddmm_loss
                # total_loss = total_loss / 3
                test_loss += total_loss.item()

                mse_loss, psnr_value, ssim_value, mean_jacobian, std_jacobian, _, nrmse_loss, mssim_value = compute_metrics(target_scan, y_synthetic, synthetic_field, criterion_lddmm, device)
                jacobian_det = jacobian_determinant_torch(synthetic_field)  # shape: (B, H-1, W-1, D-1)
                target_jacobian_det = jacobian_determinant_torch(target_svf)  # shape: (B, H-1, W-1, D-1)

                # for each sample, get target_uid and invert the uid_to_filepath map to get the filepath
                for b in range(target_uid.size(0)):
                    uid = target_uid[b].item()
                    filepath = uid_to_filepath[uid]
                    file_basename = os.path.basename(filepath)
                    # save the synthetic_image and synthetic field
                    syn_image_name = file_basename.replace(".nii.gz", f"_{opt.target_timepoint}_synthetic.nii.gz")
                    syn_image_path = os.path.join(all_scans_save_directory, syn_image_name)

                    syn_field_name = file_basename.replace(".nii.gz", f"_{opt.target_timepoint}_synthetic_field.nii.gz")
                    syn_field_path = os.path.join(all_scans_save_directory, syn_field_name)

                    # Save synthetic image
                    nib.save(nib.Nifti1Image(y_synthetic[b].cpu().numpy(), None), syn_image_path)
                    # Save synthetic field
                    nib.save(nib.Nifti1Image(synthetic_field[b].cpu().numpy(), None), syn_field_path)

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
                epe_losses.append(epe_loss)  # Add EPE to overall metrics

                # Store metrics by timepoint
                for b in range(target_timepoint.size(0)):
                    tp_idx = target_timepoint_index[b].item()
                    tp = timepoint_map[tp_idx+1]
                    
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
                    sample_epe_loss = epe_loss  # Add EPE for each sample
                    
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
                    timepoint_metrics[tp]['epe_losses'].append(sample_epe_loss)  # Add EPE for each timepoint

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
                    "Test EPE": epe_loss,  # Add EPE to progress bar
                })
                
                # Save visualization for a few test samples
                if i%10 == 0:
                    group_map = {0.0: 'CN', 1.0: 'MCI', 2.0: 'AD'}
                    loss_map = {
                        "mse": mse_loss,
                        "psnr": psnr_value,
                        "ssim": ssim_value,
                        "mssim": mssim_value,
                        "mean jacobian": mean_jacobian, 
                        "std jacobian": std_jacobian,
                        "epe": epe_loss  # Add EPE to loss map for visualization
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
    avg_epe = np.mean(epe_losses)  # Calculate average EPE
    
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
                f"Test EPE {tp}": np.mean(timepoint_metrics[tp]['epe_losses']),  # Add EPE for each timepoint
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
        "Test EPE": avg_epe,  # Add EPE to overall metrics
    }
    
    # Combine overall averages with timepoint-specific averages for CSV
    all_metrics = avg_metrics.copy()
    for tp, tp_metrics in timepoint_averages.items():
        all_metrics.update(tp_metrics)
    
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
          " Test EPE: {:.4f}".format(avg_epe),  # Add EPE to print statement
          end="\n\n")
    
    # Print timepoint-specific metrics
    for tp in timepoints:
        if tp in timepoint_averages:
            print(f"Timepoint {tp} Metrics:")
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
                  f" Test Adv Loss: {tp_metrics[f'Test Adv Loss {tp}']:.4f}",
                  f" Test EPE: {tp_metrics[f'Test EPE {tp}']:.4f}")  # Add EPE to timepoint print
    
    # Log overall metrics to wandb
    if not opt.debug:
        wandb_log_data = avg_metrics.copy()
        
        # Add timepoint-specific metrics to wandb
        for tp in timepoints:
            if tp in timepoint_averages:
                wandb_log_data.update(timepoint_averages[tp])
        
        # Log all metrics at once
        wandb.log(wandb_log_data)
        
        # Create and log a summary table with timepoint metrics
        if len(timepoint_averages) > 0:
            # Create data for the table
            table_data = []
            metrics_to_include = ["SSIM", "MS_SSIM", "PSNR", "MSE", "Mean Jacobian", "NRMSE", "EPE"]  # Added EPE to table
            
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
    
    return avg_test_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--fold", type=int, default=1, help="fold number")
    parser.add_argument("--effective_batch_size", type=int, default=4, help="effective batch size after gradient accumulation")
    parser.add_argument("--lr", type=float, default=5e-5, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--aug_p", type=float, default=0.8, help="amount of augmentation to use")
    parser.add_argument("--n_cpu", type=int, default=10, help="number of cpu threads to use during batch generation")
    parser.add_argument("--n_gpu", type=int, default=4, help="number of gpus on machine to use")
    parser.add_argument("--img_height", type=int, default=128, help="size of image height")
    parser.add_argument("--c_dim", type=int, default=3, help="number of dimensions for condition (target_labels)")
    parser.add_argument("--jacobian_coeff", type=float, default=0.1, help="weight for neg jacobian loss")
    parser.add_argument("--jac_dist_coeff", type=float, default=0.05, help="weight for jacobian distance loss")
    parser.add_argument("--disc_coeff", type=float, default=0.05, help="weight for adv loss")
    parser.add_argument("--ssim_coeff", type=float, default=0.2, help="weight for ssim loss")
    parser.add_argument("--smooth_coeff", type=float, default=0.05, help="weight for smooth loss")
    parser.add_argument("--lddmm_coeff", type=float, default=0.5, help="weight for lddmm loss")
    parser.add_argument("--use_past_data", type=str2bool, nargs='?', const=True, default=True, help="use past data for ViVit")
    parser.add_argument("--use_future_data", type=str2bool, nargs='?', const=True, default=False, help="use future data for ViVit")
    parser.add_argument("--dataset", type=str, default="ADNI", help="which dataset to use for training", choices=["ADNI", "Toy"])
    parser.add_argument("--vae_model_path", type=str, default="/scratch/anirudhkaushik/SYNBADD/VAE_10_Aug_per_sample/best_model.pth", help="model path to pre-trained vae")
    parser.add_argument("--target_timepoint", type=str, default="m18", help="timepoint to predict")
    parser.add_argument("--use_positional_embeddings", type=str2bool, nargs='?', const=True, default=True, help="use positional embeddings for transformer")
    parser.add_argument("--use_bidirectional_embeddings", type=str2bool, nargs='?', const=True, default=True, help="use bidirectional embeddings for transformer")
    parser.add_argument("--using_daft", type=str2bool, nargs='?', const=True, default=False, help="use daft for training")
    parser.add_argument("--use_deformation_field", type=str2bool, nargs='?', const=True, default=True, help="use deformation field for alignment loss")
    parser.add_argument("--patience", type=int, default=5, help="scheduler patience")
    parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=False, help="run code in debug mode")
    parser.add_argument("--specials", type=str, default="DAFT_Resnet_SVF_1LayerTransformer_OnlyPast", help="special run parameters")
    parser.add_argument("--use_val_dataset", type=str2bool, nargs='?', const=True, default=False, help="use validation dataset for training")
    parser.add_argument("--in_channels", type=int, default=3, help="number of input channels to the generator")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers in the transformer model")
    parser.add_argument("--seq_len", type=int, default=5, help="number of frames in the transformer model")
    parser.add_argument("--use_discriminator", type=str2bool, nargs='?', const=True, default=True, help="use discriminator for training")
    parser.add_argument("--use_transformer", type=str2bool, nargs='?', const=True, default=True, help="use bidirectional transformer for training")
    parser.add_argument("--transformer_subtype", type=str, default="GatedCrossAttention", help="subtype of transformer to use")
    parser.add_argument("--svf_type", type=str, default="piecewise", help="type of svfs to use")
    parser.add_argument("--warp_baseline", type=str2bool, nargs='?', const=True, default=True, help="warp baseline svf in piecewise mode")
    parser.add_argument("--epe_x_coeff", type=float, default=0, help="weight for x-channel EPE loss")
    parser.add_argument("--epe_y_coeff", type=float, default=0, help="weight for y-channel EPE loss")
    parser.add_argument("--epe_z_coeff", type=float, default=0, help="weight for z-channel EPE loss")
    parser.add_argument("--angular_coeff", type=float, default=0, help="weight for angular error loss")
    parser.add_argument("--project_name", type=str, default="SYNBADD_ADNI_5FoldCrossVal_BidirectionalTransformer+Disc_Exp", help="wandb project name")
    parser.add_argument("--run_name", type=str, default="SYNBADD_ADNI_5FoldCrossVal_BidirectionalTransformer+Disc_Exp", help="wandb project name")

    opt = parser.parse_args()
    evaluate(opt)