# =============================================================================
# File: train_resnet_SYNBADD_svf.py
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
from datasets import  SVFUltraMaskedSequenceDataset
from generator import SVFTransformerDiffeoGenerator
from losses import compute_metrics, smooth_loss_l2, ssim_only_loss, jacobian_determinant_torch
from utils import get_optimizers, get_data_loaders, str2bool, plot_slices, get_run_name, get_optimizers_transformer
from test_resnet_SYNBADD_svf import test, validate
from monai.inferers import sliding_window_inference
#import 3d resnet
from torchvision.models.video import r3d_18, mc3_18, r2plus1d_18
# import weights=R3D_18_Weights.KINETICS400_V1
from torchvision.models.video import R3D_18_Weights
from transformer_model import TransformerModel, TransformerBidirectional
from discriminator import DAFTDiscriminator

random.seed(1337)

def train(opt):
   
    run_name = get_run_name(opt)
    project_name = opt.project_name

    if not opt.debug:

        wandb.init(project=project_name, name=run_name)
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

    generator = SVFTransformerDiffeoGenerator(inshape=image_size,
                                latent_input_dim=512,
                                c_dim=opt.c_dim,
                                use_probs=False,
                                int_downsize=1,
                                in_channels=opt.in_channels,
                                int_steps=12,
                                nb_unet_features=nb_features,
                                using_svf=opt.use_deformation_field).to(device)
    
    resnet_model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    # make the last layer identity so we can use it as a feature extractor
    resnet_model.fc = nn.Identity()
    resnet_model = resnet_model.to(device)
    if opt.use_bidir_transformer:
        transformer_model = TransformerBidirectional(input_dim=512, num_heads=8, num_layers=opt.num_layers, max_seq_len=opt.seq_len).to(device)
    else:
        transformer_model = TransformerModel(input_dim=512, num_heads=8, num_layers=opt.num_layers, max_seq_len=opt.seq_len).to(device)
    discriminator_model = None
    if opt.use_discriminator:
        discriminator_model = DAFTDiscriminator(in_channels=1, ndim_non_img=opt.c_dim).to(device)

    if opt.n_gpu > 1:
        generator = nn.DataParallel(generator, list(range(opt.n_gpu)))
        resnet_model = nn.DataParallel(resnet_model, list(range(opt.n_gpu)))
        transformer_model = nn.DataParallel(transformer_model, list(range(opt.n_gpu)))
        if opt.use_discriminator:
            discriminator_model = nn.DataParallel(discriminator_model, list(range(opt.n_gpu)))

    optimizer_G, optimizer_resnet, scheduler, scheduler_resnet, criterion_lddmm = get_optimizers(opt, generator, resnet_model, device=device)
    optimizer_transformer, scheduler_transformer = get_optimizers_transformer(opt, transformer_model, device=device)
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
    train_dataset = SVFUltraMaskedSequenceDataset(opt, csv_file, svf_file, subset='train', fold=opt.fold, aug_p=opt.aug_p, 
                                      timepoints=timepoints, target_timepoint=opt.target_timepoint, use_val_dataset=opt.use_val_dataset)
    val_dataset = SVFUltraMaskedSequenceDataset(opt, csv_file, svf_file, subset='val', fold=opt.fold, aug_p=0.0, 
                                    timepoints=timepoints, target_timepoint=opt.target_timepoint, use_val_dataset=opt.use_val_dataset)
    test_dataset = SVFUltraMaskedSequenceDataset(opt, csv_file, svf_file, subset='test', fold=opt.fold, aug_p=0.0, 
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
        resnet_model.train()
        transformer_model.train()
        optimizer_G.zero_grad()
        optimizer_resnet.zero_grad()
        optimizer_transformer.zero_grad()

        if opt.use_discriminator:
            optimizer_D.zero_grad()
            discriminator_model.train()

        epoch_loss = 0

        ssim_values, mean_jacobian_losses, std_jacobian_losses, mse_losses, psnr_losses, ssim_losses, avg_jacobian_losses, avg_nrmse_losses = [], [], [], [], [], [], [], []
        disc_adv_losses = []
        
        with tqdm(total=len(train_dataloader.dataset), desc=f"Epoch {epoch+1}/{opt.n_epochs}") as progress_bar:
            for i, batch in enumerate(train_dataloader):
                stacked_svf, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint, target_timepoint_index, target_svf = [batch[k].to(device) for k in ["stacked_svf", "latest_scan", "target_scan", "class_label", "reg_label", "target_timepoint", "latest_timepoint", "target_timepoint_index", "target_svf"]]
                stacked_svf, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint, target_svf = [Variable(v.type(torch.float32)).to(device) for v in [stacked_svf, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint, target_svf]]
                target_labels = torch.cat([class_labels, reg_labels, target_timepoint], dim=1)
                if opt.use_past_data or opt.use_future_data:
                    # [B, T, C, H, W, D] Collapse batch dimension into time dimension
                    batch_size, temporal_dim, c_dim, h, w, d = stacked_svf.shape
                    stacked_svf = stacked_svf.view(-1, *stacked_svf.shape[2:])
                    # iterate and pass it through 3d resnet
                    contextualized_scan = resnet_model(stacked_svf)
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
                total_loss.backward()

                mse_loss, psnr_value, ssim_value, mean_jacobian, std_jacobian, _, nrmse_loss = compute_metrics(target_scan, y_synthetic, synthetic_field, criterion_lddmm, device)
                jacobian_det = jacobian_determinant_torch(synthetic_field) # shape: (B, H-1, W-1, D-1)

                ssim_losses.append(ssim_loss.item())
                mean_jacobian_losses.append(mean_jacobian)
                std_jacobian_losses.append(std_jacobian)
                mse_losses.append(mse_loss)
                psnr_losses.append(psnr_value)
                ssim_values.append(ssim_value)
                avg_jacobian_losses.append(lddmm_loss.item())
                avg_nrmse_losses.append(nrmse_loss)
                disc_adv_losses.append(d_adv.item())

                epoch_loss += total_loss.item()

                if (i + 1) % accumulation_steps == 0:
                    optimizer_G.step()
                    optimizer_resnet.step()
                    optimizer_transformer.step()
                    optimizer_G.zero_grad()
                    optimizer_resnet.zero_grad()
                    optimizer_transformer.zero_grad()

                    if opt.use_discriminator:
                        optimizer_D.step()
                        optimizer_D.zero_grad()

                progress_bar.update(stacked_svf.size(0))
                progress_bar.set_postfix({
                    "SSIM Loss": ssim_loss.item(),
                    "LDDMM Loss": lddmm_loss.item(),
                    "Mse": mse_loss,
                    "PSNR": psnr_value,
                    "SSIM": ssim_value,
                    "Total Loss": total_loss.item(),
                    "Mean Jac": mean_jacobian,
                    "Std Jac": std_jacobian,
                    "Nrmse": nrmse_loss,
                    "ADV Loss": d_adv.item(),
                })

        epoch_loss /= len(train_dataloader)
        avg_ssim_loss = np.mean(ssim_losses)
        avg_mean_jacobian = np.mean(mean_jacobian_losses)
        avg_std_jacobian = np.mean(std_jacobian_losses)
        avg_psnr = np.mean(psnr_losses)
        avg_ssim = np.mean(ssim_values)
        avg_mse = np.mean(mse_losses)
        avg_jacobian_loss = np.mean(avg_jacobian_losses)
        avg_nrmse_loss = np.mean(avg_nrmse_losses)
        avg_disc_adv_loss = np.mean(disc_adv_losses)
        
        # Validation phase
        if opt.use_val_dataset:
            val_loss = validate(generator, resnet_model, transformer_model, discriminator_model, val_dataloader, criterion_lddmm, device, opt, run_name)
        else:
            val_loss = epoch_loss
        
        scheduler.step(val_loss)  # Using validation loss for scheduler
        scheduler_resnet.step(val_loss)
        scheduler_transformer.step(val_loss)

        if opt.use_discriminator:
            scheduler_D.step(val_loss)
        
        if not opt.debug:
            wandb.log({
                "Epoch": epoch,
                "Train Loss": epoch_loss, 
                "Train SSIM Loss": avg_ssim_loss,
                "Train Validation Loss": val_loss,
                "Train Learning Rate": optimizer_G.param_groups[0]['lr'],
                "Train Mean Jac": avg_mean_jacobian,
                "Train Std Jac": avg_std_jacobian,
                "Train PSNR": avg_psnr,
                "Train SSIM": avg_ssim,
                "Train MSE": avg_mse,
                "Train NRMSE": avg_nrmse_loss,
                "Train LDDMM Loss": avg_jacobian_loss,
                "Train ADV Loss": avg_disc_adv_loss,
            })

        print(f"Epoch {epoch+1}/{opt.n_epochs}: Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            group_map = {0.0: 'CN', 1.0: 'MCI', 2.0: 'AD'}
            best_loss = val_loss
            patience_counter = 0
            
            # Save best model
            model_save_path = f"/scratch/anirudhkaushik/SYNBADD/{save_directory}/{run_name}_SYNBADD_SVF.pth"
            torch.save(generator.state_dict(), model_save_path)
            torch.save(resnet_model.state_dict(), f"/scratch/anirudhkaushik/SYNBADD/{save_directory}/{run_name}_resnet_SYNBADD_SVF.pth")
            torch.save(transformer_model.state_dict(), f"/scratch/anirudhkaushik/SYNBADD/{save_directory}/{run_name}_transformer_SYNBADD_SVF.pth")
            if opt.use_discriminator:
                torch.save(discriminator_model.state_dict(), f"/scratch/anirudhkaushik/SYNBADD/{save_directory}/{run_name}_discriminator_SYNBADD_SVF.pth")
            
            print(f"Saved best model at epoch {epoch+1} with validation loss: {val_loss:.4f}")

            if not opt.debug:
                # Visualize best validation result
                with torch.no_grad():
                    val_batch = next(iter(val_dataloader))
                    stacked_svf, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint, target_timepoint_index, target_svf = [val_batch[k].to(device) for k in ["stacked_svf", "latest_scan", "target_scan", "class_label", "reg_label", "target_timepoint", "latest_timepoint", "target_timepoint_index", "target_svf"]]
                    stacked_svf, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint, target_svf = [Variable(v.type(torch.float32)).to(device) for v in [stacked_svf, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint, target_svf]]
                    target_labels = torch.cat([class_labels, reg_labels, target_timepoint], dim=1)
                    if opt.use_past_data or opt.use_future_data:
                        # [B, T, C, H, W, D] Collapse batch dimension into time dimension
                        batch_size, temporal_dim, c_dim, h, w, d = stacked_svf.shape
                        stacked_svf = stacked_svf.view(-1, *stacked_svf.shape[2:])
                        # iterate and pass it through 3d resnet
                        contextualized_scan = resnet_model(stacked_svf)
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

                    mse_loss, psnr_value, ssim_value, mean_jacobian, std_jacobian, _, nrmse_loss = compute_metrics(target_scan, y_synthetic, synthetic_field, criterion_lddmm, device)
                    jacobian_det = jacobian_determinant_torch(synthetic_field) # shape: (B, H-1, W-1, D-1)
                    target_jacobian_det = jacobian_determinant_torch(target_svf) # shape: (B, H-1, W-1, D-1)

                    loss_map = {"mse": mse_loss,"psnr": psnr_value,"ssim": ssim_value,"mean jacobian": mean_jacobian, "std jacobian": std_jacobian, "nrmse": nrmse_loss}
                    
                    fig = plot_slices(latest_scan, target_scan, y_synthetic, jacobian_det, target_jacobian_det, loss_map,
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
    generator.load_state_dict(torch.load(f"/scratch/anirudhkaushik/SYNBADD/{save_directory}/{run_name}_SYNBADD_SVF.pth"))
    resnet_model.load_state_dict(torch.load(f"/scratch/anirudhkaushik/SYNBADD/{save_directory}/{run_name}_resnet_SYNBADD_SVF.pth"))
    transformer_model.load_state_dict(torch.load(f"/scratch/anirudhkaushik/SYNBADD/{save_directory}/{run_name}_transformer_SYNBADD_SVF.pth"))
    if opt.use_discriminator:
        discriminator_model.load_state_dict(torch.load(f"/scratch/anirudhkaushik/SYNBADD/{save_directory}/{run_name}_discriminator_SYNBADD_SVF.pth"))
    
    # Test phase
    test_loss = test(generator, resnet_model, transformer_model, discriminator_model, test_dataloader, criterion_lddmm, device, opt, run_name)
    print(f"Final test loss: {test_loss:.4f}")
    
    if not opt.debug:
        wandb.log({"Final Test Loss": test_loss})
        wandb.finish()

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
    parser.add_argument("--jac_dist_coeff", type=float, default=0.2, help="weight for jacobian distance loss")
    parser.add_argument("--disc_coeff", type=float, default=0.5, help="weight for adv loss")
    parser.add_argument("--smooth_coeff", type=float, default=0.1, help="weight for smooth loss")
    parser.add_argument("--lddmm_coeff", type=float, default=0.1, help="weight for lddmm loss")
    parser.add_argument("--use_past_data", type=str2bool, nargs='?', const=True, default=True, help="use past data for ViVit")
    parser.add_argument("--use_future_data", type=str2bool, nargs='?', const=True, default=False, help="use future data for ViVit")
    parser.add_argument("--dataset", type=str, default="ADNI", help="which dataset to use for training", choices=["ADNI", "Toy"])
    parser.add_argument("--target_timepoint", type=str, default="m18", help="timepoint to predict")
    parser.add_argument("--use_deformation_field", type=str2bool, nargs='?', const=True, default=True, help="use deformation field for alignment loss")
    parser.add_argument("--patience", type=int, default=5, help="early stopping patience")
    parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=False, help="run code in debug mode")
    parser.add_argument("--specials", type=str, default="DAFT_Resnet_SVF_1LayerTransformer_OnlyPast", help="special run parameters")
    parser.add_argument("--use_val_dataset", type=str2bool, nargs='?', const=True, default=False, help="use validation dataset for training")
    parser.add_argument("--in_channels", type=int, default=3, help="number of input channels to the generator")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers in the transformer model")
    parser.add_argument("--seq_len", type=int, default=5, help="number of frames in the transformer model")
    parser.add_argument("--use_discriminator", type=str2bool, nargs='?', const=True, default=True, help="use discriminator for training")
    parser.add_argument("--use_bidir_transformer", type=str2bool, nargs='?', const=True, default=True, help="use bidirectional transformer for training")
    parser.add_argument("--project_name", type=str, default="SYNBADD_ADNI_5FoldCrossVal_BidirectionalTransformer+Disc_Exp", help="wandb project name")

    opt = parser.parse_args()
    train(opt)