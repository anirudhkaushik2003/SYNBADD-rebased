# =============================================================================
# File: train_model.py
# Purpose: Training entrypoint for a specific experiment variant or legacy branch.
# Prerequisites: Relevant dataset metadata, supporting modules (datasets/generator/models/losses/utils), and configured /scratch output paths.
# Usually Run Next: Run corresponding test/evaluation scripts to generate outputs and comparison metrics.
# Model Components Used: See imports/CLI in this file for exact model stack; active latest flagship training is train_VAE_SYNBADD_svf.py.
# =============================================================================

import argparse
import os
import numpy as np
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from datetime import datetime

from datasets import ExampleDataset, MultiEpochsDataLoader, ADNIDataset
from discriminator import weights_init_normal, SFCNDiscriminator
from generator import DiffeoGenerator
from losses import r1_reg, smooth_loss_l2
from matplotlib import pyplot as plt

random.seed(1337)

def train(opt):
    os.makedirs("saved_models", exist_ok=True)
    
    # Initialize Weights & Biases (wandb)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"CounterSynth_{opt.img_height}_{opt.effective_batch_size}_{opt.lr}_{opt.aug_p}_{timestamp}"
    wandb.init(entity="anirudhkaushik2003", project="CounterSynth", name=run_name)
    wandb.config.update(vars(opt))  # Log hyperparameters
    
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.float32)

    # Loss functions
    criterion_cls = F.binary_cross_entropy_with_logits
    criterion_regression = F.mse_loss
    
    lambda_gp = 1
    accumulation_steps = opt.effective_batch_size // opt.batch_size

    nb_features = [
        [16, 32, 32, 32, 32],  # encoder
        [32, 32, 32, 32, 32, 32, 16, 16]  # decoder
    ]

    generator = DiffeoGenerator(inshape=(opt.img_height, opt.img_height, opt.img_height),
                                c_dim=2,
                                use_probs=False,
                                int_downsize=2,
                                in_channels=1,
                                int_steps=7,
                                nb_unet_features=nb_features).to(device)

    if opt.n_gpu > 1:
        generator = nn.DataParallel(generator, list(range(opt.n_gpu)))

    discriminator = SFCNDiscriminator(in_channels=1,
                                      channel_number=[16, 32, 64, 64, 64, 128, 64]).to(device)
    if opt.n_gpu > 1:
        discriminator = nn.DataParallel(discriminator, list(range(opt.n_gpu)))
    discriminator.apply(weights_init_normal)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.2 * opt.lr, betas=(opt.b1, opt.b2))

    dataset = ADNIDataset("/scratch/anirudhkaushik/ADNI1_Complete_Updated.csv", 'train', aug_p=opt.aug_p)
    dataloader = MultiEpochsDataLoader(dataset,
                                       batch_size=opt.batch_size,
                                       drop_last=False,
                                       shuffle=True,
                                       num_workers=opt.n_cpu,
                                       pin_memory=True,
                                       prefetch_factor=1)

    for epoch in range(opt.n_epochs):
        discriminator_adv_loss, discriminator_cls_loss, discriminator_reg_loss = [], [], []
        generator_adv_loss, generator_cls_loss, generator_reg_loss = [], [], []

        optimizer_D.zero_grad()
        optimizer_G.zero_grad()

        with tqdm(total=len(dataloader.dataset)) as progress_bar:
            for i, (imgs, class_labels, reg_labels) in enumerate(dataloader):
                x_real = Variable(imgs.type(torch.float32)).to(device)
                class_labels = Variable(class_labels.type(torch.float32)).to(device)
                reg_labels = Variable(reg_labels.type(torch.float32)).to(device)
                
                rand_idx = torch.randperm(class_labels.size(0))
                sampled_class_labels = class_labels[rand_idx]
                sampled_reg_labels = reg_labels[rand_idx]

                diff_labels = (sampled_reg_labels - reg_labels)
                target_labels = torch.cat([sampled_class_labels, sampled_reg_labels], dim=1)
                
                x_real.requires_grad_()
                out_real, out_cls, out_reg = discriminator(x_real)
                d_loss_real = torch.mean(F.softplus(-out_real))
                d_loss_cls = criterion_cls(out_cls, class_labels) + criterion_regression(out_reg, reg_labels)
                d_loss_cls = torch.mean(d_loss_cls)
                
                x_fake, preint_flow = generator(x_real, target_labels)
                out_fake, out_fake_cls, out_fake_reg = discriminator(x_fake.detach())
                d_loss_fake = torch.mean(F.softplus(out_fake))
                d_loss_gp = r1_reg(out_real, x_real)
                d_adv = d_loss_real + d_loss_fake
                
                discriminator_adv_loss.append(d_adv.item())
                discriminator_cls_loss.append(d_loss_cls.item())
                discriminator_reg_loss.append(d_loss_gp.item())
                
                d_loss = (d_adv + d_loss_cls + lambda_gp * d_loss_gp) / accumulation_steps
                d_loss.backward()
                
                if (i + 1) % accumulation_steps == 0:
                    optimizer_D.step()
                    optimizer_D.zero_grad()
                
                if i % opt.n_critic == 0:
                    out_src, out_cls, out_reg = discriminator(x_fake)
                    g_loss_fake = torch.mean(F.softplus(-out_src))
                    g_loss_cls = criterion_cls(out_cls, sampled_class_labels) + criterion_regression(out_reg, sampled_reg_labels)
                    
                    generator_adv_loss.append(g_loss_fake.item())
                    generator_cls_loss.append(g_loss_cls.item())
                    
                    int_loss_l2 = smooth_loss_l2(preint_flow)
                    generator_reg_loss.append(int_loss_l2.item())
                    
                    g_loss = (g_loss_fake + g_loss_cls + 10 * int_loss_l2) / accumulation_steps
                    g_loss.backward()
                    
                    if (i + 1) % accumulation_steps == 0:
                        optimizer_G.step()
                        optimizer_G.zero_grad()
                
                wandb.log({
                    "discriminator_adv_loss": np.mean(discriminator_adv_loss),
                    "discriminator_cls_loss": np.mean(discriminator_cls_loss),
                    "generator_adv_loss": np.mean(generator_adv_loss),
                    "generator_cls_loss": np.mean(generator_cls_loss),
                    "generator_int_loss": np.mean(generator_reg_loss)
                })
                
                progress_bar.update(x_real.size(0))
        
        if epoch % 20 == 0:
            model_path = f"/scratch/anirudhkaushik/saved_models/generator_{epoch}.pth"
            torch.save(generator.state_dict(), model_path)

            sample_idx = random.randint(0, x_fake.shape[0] - 1)
            img_slice = x_fake[sample_idx, 0, :, :, opt.img_height//2].detach().cpu().numpy()
            reg_label = reg_labels[sample_idx].cpu().numpy()
            sampled_reg_label = sampled_reg_labels[sample_idx].cpu().numpy()
            
            plt.imshow(img_slice, cmap='gray')
            plt.title(f"Reg Label: {reg_label}, Sampled Reg Label: {sampled_reg_label}")
            plt.axis('off')
            
            img_path = f"saved_models/sample_image_epoch_{epoch}.png"
            plt.savefig(img_path)
            wandb.log({"Sample Image": wandb.Image(img_path)})
    
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--effective_batch_size", type=int, default=128, help="effective batch size after gradient accumulation")
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--aug_p", type=float, default=0.8, help="amount of augmentation to use")
    parser.add_argument("--n_cpu", type=int, default=10, help="number of cpu threads to use during batch generation")
    parser.add_argument("--n_gpu", type=int, default=4, help="number of gpus on machine to use")
    parser.add_argument("--img_height", type=int, default=128, help="size of image height") 
    parser.add_argument("--n_critic", type=int, default=1, help="number of training iterations for discriminator")

    opt = parser.parse_args()
    train(opt)
