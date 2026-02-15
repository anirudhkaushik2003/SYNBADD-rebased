# =============================================================================
# File: evaluate_model.py
# Purpose: Support or experiment file within SYNBADD repository.
# Prerequisites: Depends on imports and paths defined in this file.
# Usually Run Next: Typically called by a train/test/eval script in this repo.
# Model Components Used: See imports in this file for exact model/loss dependencies.
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

from datasets import  SVFUltraMaskedSequenceDataset
from generator import SVFRegressorDiffeoGenerator
from losses import compute_metrics, smooth_loss_l2, ssim_only_loss, jacobian_determinant_torch
from utils import get_data_loaders, str2bool, plot_slices, get_run_name, get_optimizers_regressor, get_optimizers_generator, get_optimizers_transformer, get_vae_model
from test_VAE_SYNBADD_svf import test, validate
from monai.inferers import sliding_window_inference
#import 3d resnet
# import weights=R3D_18_Weights.KINETICS400_V1
from torchvision.models.video import R3D_18_Weights
from discriminator import DAFTDiscriminator
from models import LatentRegressor
from transformer_model import TransformerBidirectional

random.seed(1337)

import torch.distributed as dist
import torch.multiprocessing as mp

def setup_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()


def evaluate(opt):
   
    run_name = opt.run_name
    project_name = opt.project_name

    if not opt.debug:

        wandb.init(project=f"{project_name}_EVAL", name=run_name)
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
    if opt.use_bidir_transformer:
        transformer_model = TransformerBidirectional(input_dim=512, num_heads=8, num_layers=opt.num_layers, max_seq_len=opt.seq_len, use_positional_embeddings=opt.use_positional_embeddings, use_bidirectional_embeddings=opt.use_bidirectional_embeddings).to(device)
    else:
        regression_model = LatentRegressor(seq_len=opt.seq_len).to(device)

    discriminator_model = None
    if opt.use_discriminator:
        discriminator_model = DAFTDiscriminator(in_channels=1, ndim_non_img=opt.c_dim).to(device)

    if opt.n_gpu > 1:
        generator = nn.DataParallel(generator, list(range(opt.n_gpu)))
        vae_model = nn.DataParallel(vae_model, list(range(opt.n_gpu)))
        regression_model = nn.DataParallel(regression_model, list(range(opt.n_gpu)))
        if opt.use_discriminator:
            discriminator_model = nn.DataParallel(discriminator_model, list(range(opt.n_gpu)))

    optimizer_G, scheduler_G, criterion_lddmm = get_optimizers_generator(opt, generator, device)
    
    # Temporal Encoders
    optimizer_regressor, scheduler_regressor = None, None
    optimizer_transformer, scheduler_transformer = None, None
    if opt.use_bidir_transformer:
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
    early_stopping_patience = opt.patience+2

    # Load best model for testing
    print("Loading best model for testing...")
    generator.load_state_dict(torch.load(f"/scratch/anirudhkaushik/SYNBADD/{save_directory}/{run_name}_SYNBADD_SVF.pth"))
    vae_model.load_state_dict(torch.load(f"/scratch/anirudhkaushik/SYNBADD/{save_directory}/{run_name}_vae_SYNBADD_SVF.pth"))
    
    if opt.use_bidir_transformer:
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
    parser.add_argument("--use_bidir_transformer", type=str2bool, nargs='?', const=True, default=True, help="use bidirectional transformer for training")
    parser.add_argument("--project_name", type=str, default="SYNBADD_ADNI_5FoldCrossVal_BidirectionalTransformer+Disc_Exp", help="wandb project name")
    parser.add_argument("--run_name", type=str, default="SYNBADD_ADNI_5FoldCrossVal_BidirectionalTransformer+Disc_Exp", help="wandb project name")

    opt = parser.parse_args()
    evaluate(opt)
