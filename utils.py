# =============================================================================
# File: utils.py
# Purpose: Utility layer for run naming, optimizer construction, VAE loading, plotting, and dataloader creation.
# Prerequisites: Imported by most train/test entrypoints.
# Usually Run Next: Supports reproducible configuration and shared setup across experiments.
# Model Components Used: Centralizes get_vae_model used by latest subjectwise train/test pipeline.
# =============================================================================

import torch
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from datasets import  MultiEpochsDataLoader
from losses import VxmLDDMMLoss
from colorama import Fore, Style, init
import voxelmorph.torch as vxm
from voxelmorph.torch.networks import VxmDense
from models import VAE3D_UNetSkip
import numpy as np

init(autoreset=True)  # Automatically reset styles after each print

def get_vae_model(opt, device):
    vae_model = VAE3D_UNetSkip()
    # load vae model
    VAE_DIR_NAME = "SVF Encoder VAE Training"
    # vae_model_path = f"/scratch/anirudhkaushik/SYNBADD/{VAE_DIR_NAME}/best_model.pth"
    vae_model_path = opt.vae_model_path
    print(f"Loading VAE model from {vae_model_path}")
    # it was stored as nn.DataParallel
    state_dict = torch.load(vae_model_path)
    # Remove 'module.' prefix if it was saved with DataParallel
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')
        new_state_dict[new_key] = v
    # Load the state dict into the model
    vae_model.load_state_dict(new_state_dict)
    # freeze vae model
    for param in vae_model.parameters():
        param.requires_grad = False

    vae_model.eval()  # Important: evaluation mode

    return vae_model


def get_optimizers(opt, generator, temporal_encoder, device):
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_temporal = torch.optim.Adam(temporal_encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=opt.patience, verbose=True, min_lr=1e-6)
    scheduler_temporal = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_temporal, mode='min', factor=0.5, patience=opt.patience, verbose=True, min_lr=1e-6)
    
    vxm_path = "/scratch/anirudhkaushik/vxm_checkpoints/best_vxm_model_64.pth"
    vxm_model = VxmDense(inshape=(opt.img_height, opt.img_height, opt.img_height), nb_unet_features=[[16, 32, 32, 32], [32, 32, 32, 32]])
    vxm_model.to(device)
    # Step 2: Load state dict correctly
    state_dict = torch.load(vxm_path)
    # Remove 'module.' prefix if it was saved with DataParallel
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')  # strip the "module." prefix
        new_state_dict[new_key] = v
    vxm_model.load_state_dict(new_state_dict)
    vxm_model.eval()  # Important: evaluation mode

    for param in vxm_model.parameters():
        param.requires_grad = False
    if opt.n_gpu > 1:
        vxm_model = torch.nn.DataParallel(vxm_model, list(range(opt.n_gpu)))

    jacobian_loss = VxmLDDMMLoss(vxm_model, weight_jacobian=0.01, weight_intensity=1.0)

    return optimizer_G, optimizer_temporal, scheduler, scheduler_temporal, jacobian_loss

def get_optimizers_generator(opt, generator, device):
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=opt.patience, verbose=True, min_lr=1e-6)
    

    vxm_path = "/scratch/anirudhkaushik/vxm_checkpoints/best_vxm_model_64.pth"
    vxm_model = VxmDense(inshape=(opt.img_height, opt.img_height, opt.img_height), nb_unet_features=[[16, 32, 32, 32], [32, 32, 32, 32]])
    vxm_model.to(device)
    # Step 2: Load state dict correctly
    state_dict = torch.load(vxm_path)
    # Remove 'module.' prefix if it was saved with DataParallel
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')  # strip the "module." prefix
        new_state_dict[new_key] = v
    vxm_model.load_state_dict(new_state_dict)
    vxm_model.eval()  # Important: evaluation mode

    for param in vxm_model.parameters():
        param.requires_grad = False
    if opt.n_gpu > 1:
        vxm_model = torch.nn.DataParallel(vxm_model, list(range(opt.n_gpu)))

    jacobian_loss = VxmLDDMMLoss(vxm_model, weight_jacobian=0.01, weight_intensity=1.0)

    return optimizer_G, scheduler_G, jacobian_loss

def get_optimizers_regressor(opt, regressor):
    optimizer_regressor = torch.optim.Adam(regressor.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    scheduler_regressor = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_regressor, mode='min', factor=0.5, patience=opt.patience, verbose=True, min_lr=1e-6)

    return optimizer_regressor, scheduler_regressor

def get_optimizers_transformer(opt, transformer, device):
    optimizer_trans = torch.optim.Adam(transformer.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    scheduler_trans = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_trans, mode='min', factor=0.5, patience=opt.patience, verbose=True, min_lr=1e-6)

    return optimizer_trans, scheduler_trans
    

def get_data_loaders(opt, train_dataset, val_dataset, test_dataset):
    # Create dataloaders
    train_dataloader, val_dataloader, test_dataloader = None, None, None
    if train_dataset is not None:
        train_dataloader = MultiEpochsDataLoader(train_dataset,
                                            batch_size=opt.batch_size,
                                            drop_last=False,
                                            shuffle=True,
                                            num_workers=opt.n_cpu,
                                            pin_memory=True,
                                            prefetch_factor=1)
    
    if val_dataset is not None:
        val_dataloader = MultiEpochsDataLoader(val_dataset,
                                            batch_size=max(opt.batch_size //2, 1),
                                            drop_last=False,
                                            shuffle=False,
                                            num_workers=opt.n_cpu,
                                            pin_memory=True,
                                            prefetch_factor=1)
    
    if test_dataset is not None:
        test_dataloader = MultiEpochsDataLoader(test_dataset,
                                            batch_size=max(opt.batch_size //2, 1),
                                            drop_last=False,
                                            shuffle=False,
                                            num_workers=opt.n_cpu,
                                            pin_memory=True,
                                            prefetch_factor=1)
    
    return train_dataloader, val_dataloader, test_dataloader

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def plot_slices(latest_scan, target_scan, warped_scan, jacobian_scan, target_jacobian_scan, loss_map, age, group_label, target_timepoint, latest_scan_timepoint):
    titles = ["Original", "Target", "Warped", "Jacobian", "Target Jaobian"]
    fig, axs = plt.subplots(1, len(titles), figsize=(15, 10))
    timepoint_idx_label_map = {0: 'sc', 1: 'm06', 2: 'm12', 3: 'm18', 4: 'm24', 5: 'm36'}

    latest_scan_timepoint = timepoint_idx_label_map[latest_scan_timepoint]
    target_timepoint = timepoint_idx_label_map[target_timepoint]

    diff_scan = target_scan - warped_scan
    change_scan = warped_scan - latest_scan
    target_change_scan = target_scan - latest_scan

    # jacobian_scan has shape [B, H-1, W-1, D-1]
    jacobian_scan = jacobian_scan[0].detach().cpu().numpy() # [0] to get the first image in the batch
    target_jacobian_scan = target_jacobian_scan[0].detach().cpu().numpy() # [0] to get the first image in the batch
    scans = [latest_scan, target_scan, warped_scan, jacobian_scan, target_jacobian_scan]

    for i, scan in enumerate(scans):
        if i == 3 or i == 4:
                middle_slice = scan[ :, :, scan.shape[-1] // 2]
                axs[i].imshow(middle_slice, cmap='jet')
        else:
            middle_slice = scan[0, 0,  :, :, scan.shape[-1] // 2].cpu().detach().numpy()
            if i > 4:
                axs[i].imshow(middle_slice, cmap='jet')
            else:
                axs[i].imshow(middle_slice, cmap='gray')
        axs[i].set_title(titles[i])
        axs[i].axis('off')

    # add losses to figsuptitle 
    formatted_loss_map = {}
    for key, value in loss_map.items():
        if isinstance(value, torch.Tensor):
            formatted_loss_map[key] = value.item()  # Convert tensor to Python scalar
        else:
            formatted_loss_map[key] = value
            
    # Now use the formatted values for string formatting
    loss_str = ", ".join([f"{key}: {value:.4f}" for key, value in formatted_loss_map.items()])
    fig.suptitle(f"Age: {age}, Group: {group_label}, Latest Timepoint: {latest_scan_timepoint}, Target Timepoint: {target_timepoint}\n{loss_str}")

    return fig

def get_run_name(opt):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run_name_parts = [
        (Fore.YELLOW, "SYNBADD"),
        (Fore.CYAN, str(opt.img_height)),
        (Fore.MAGENTA, str(opt.effective_batch_size)),
        (Fore.GREEN, str(opt.lr)),
        (Fore.BLUE, str(opt.aug_p)),
        (Fore.WHITE, timestamp),
        (Fore.RED, opt.dataset),
        (Fore.LIGHTBLUE_EX, opt.target_timepoint)
    ]

    if opt.use_past_data and opt.use_future_data:
        time_flag = (Fore.LIGHTMAGENTA_EX, "longitudinal")
    elif opt.use_past_data:
        time_flag = (Fore.LIGHTRED_EX, "prev_only")
    elif opt.use_future_data:
        time_flag = (Fore.LIGHTCYAN_EX, "future_only")
    else:
        time_flag = (Fore.LIGHTGREEN_EX, "cross_sectional")

    run_name_parts.append(time_flag)
    run_name_parts.append((Fore.LIGHTYELLOW_EX, opt.specials))

    # Construct the colored run name
    colored_run_name = ""
    plain_run_name = ""
    for i, (color, val) in enumerate(run_name_parts):
        if i == 0:
            colored_run_name += color + val
            plain_run_name += val
        else:
            colored_run_name += color + "_" + val
            plain_run_name += "_" + val

    colored_run_name = colored_run_name.lstrip("_")
    plain_run_name = plain_run_name.lstrip("_")

    print(f"\nRun Name: {colored_run_name}{Style.RESET_ALL}")

    return plain_run_name  # You can return the plain name for logging or saving purposes