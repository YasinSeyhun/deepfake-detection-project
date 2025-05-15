import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
import argparse
from pathlib import Path

from src.models.stylegan.model import StyleGANGenerator
from src.data.dataset import FFHQDataset
from src.training.utils import (
    compute_gradient_penalty,
    compute_path_length_penalty,
    compute_r1_penalty,
)

def train(args):
    # Initialize wandb
    wandb.init(
        project="deepfake-detection",
        config=vars(args),
        name=f"stylegan_ffhq_{args.resolution}"
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize models
    generator = StyleGANGenerator(
        latent_dim=args.latent_dim,
        n_mlp=args.n_mlp,
        channel_multiplier=args.channel_multiplier,
    ).to(device)
    
    discriminator = Discriminator(
        size=args.resolution,
        channel_multiplier=args.channel_multiplier,
    ).to(device)
    
    # Initialize optimizers
    g_optimizer = optim.Adam(
        generator.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )
    
    d_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )
    
    # Load dataset
    dataset = FFHQDataset(
        root=args.data_dir,
        resolution=args.resolution,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Training loop
    step = 0
    for epoch in range(args.epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for real_images in progress_bar:
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Train discriminator
            d_optimizer.zero_grad()
            
            # Generate fake images
            z = torch.randn(batch_size, args.latent_dim, device=device)
            fake_images = generator(z)
            
            # Compute discriminator loss
            real_pred = discriminator(real_images)
            fake_pred = discriminator(fake_images.detach())
            
            d_loss = (
                F.softplus(-real_pred).mean() +
                F.softplus(fake_pred).mean()
            )
            
            # Add gradient penalty
            if args.use_gp:
                gp = compute_gradient_penalty(
                    discriminator, real_images, fake_images.detach()
                )
                d_loss = d_loss + args.gp_lambda * gp
            
            # Add R1 penalty
            if args.use_r1:
                r1_penalty = compute_r1_penalty(discriminator, real_images)
                d_loss = d_loss + args.r1_lambda * r1_penalty
            
            d_loss.backward()
            d_optimizer.step()
            
            # Train generator
            g_optimizer.zero_grad()
            
            # Generate fake images
            z = torch.randn(batch_size, args.latent_dim, device=device)
            fake_images = generator(z)
            fake_pred = discriminator(fake_images)
            
            # Compute generator loss
            g_loss = F.softplus(-fake_pred).mean()
            
            # Add path length penalty
            if args.use_pl:
                pl_penalty = compute_path_length_penalty(generator, z)
                g_loss = g_loss + args.pl_lambda * pl_penalty
            
            g_loss.backward()
            g_optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix({
                "D_loss": d_loss.item(),
                "G_loss": g_loss.item(),
            })
            
            # Log metrics
            if step % args.log_interval == 0:
                wandb.log({
                    "D_loss": d_loss.item(),
                    "G_loss": g_loss.item(),
                    "epoch": epoch,
                    "step": step,
                })
                
                writer.add_scalar("Loss/Discriminator", d_loss.item(), step)
                writer.add_scalar("Loss/Generator", g_loss.item(), step)
            
            # Save checkpoint
            if step % args.save_interval == 0:
                torch.save({
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "g_optimizer": g_optimizer.state_dict(),
                    "d_optimizer": d_optimizer.state_dict(),
                    "epoch": epoch,
                    "step": step,
                }, checkpoint_dir / f"checkpoint_{step}.pt")
            
            step += 1
    
    # Save final model
    torch.save({
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "g_optimizer": g_optimizer.state_dict(),
        "d_optimizer": d_optimizer.state_dict(),
        "epoch": args.epochs,
        "step": step,
    }, checkpoint_dir / "final_model.pt")
    
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Model parameters
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--use_gp", action="store_true")
    parser.add_argument("--gp_lambda", type=float, default=10.0)
    parser.add_argument("--use_r1", action="store_true")
    parser.add_argument("--r1_lambda", type=float, default=10.0)
    parser.add_argument("--use_pl", action="store_true")
    parser.add_argument("--pl_lambda", type=float, default=2.0)
    
    # Logging parameters
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    
    args = parser.parse_args()
    train(args) 