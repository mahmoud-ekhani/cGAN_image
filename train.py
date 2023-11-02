import argparse
import os
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from models.generator import Generator
from models.discriminator import Discriminator
from data.loader import get_dataloader

# Training the models
def train(generator, discriminator, criterion, optimizer_G, scheduler_G, optimizer_D, scheduler_D, 
          dataloader, val_dataloader, device, epochs, latent_dim, start_epoch, patience):
    
    min_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(start_epoch, start_epoch+epochs):
        # Train loop
        for i, (imgs, labels) in enumerate(dataloader):
            real = torch.ones(imgs.size(0), 1).to(device)
            fake = torch.zeros(imgs.size(0), 1).to(device)

            # Train the generator
            optimizer_G.zero_grad()
            
            noise = torch.randn(imgs.size(0), latent_dim).to(device)
            gen_labels = torch.randint(0, 10, (imgs.size(0),)).to(device)
            gen_imgs = generator(noise, gen_labels)

            validity = discriminator(gen_imgs, gen_labels)
            g_loss = criterion(validity, real)
            g_loss.backward()
            optimizer_G.step()

            # Train the discriminator
            optimizer_D.zero_grad()

            real_validity = discriminator(imgs, labels)
            d_real_loss = criterion(real_validity, real)

            fake_validity = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = criterion(fake_validity, fake)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

        # Validation loop
        with torch.no_grad():
            val_g_losses = []
            val_d_losses = []
            for i, (imgs, labels) in enumerate(val_dataloader):
                real = torch.ones(imgs.size(0), 1).to(device)
                fake = torch.zeros(imgs.size(0), 1).to(device)
                
                noise = torch.randn(imgs.size(0), latent_dim).to(device)
                gen_imgs = generator(noise, labels)  # Use actual labels

                validity = discriminator(gen_imgs, labels)
                g_loss = criterion(validity, real)
                val_g_losses.append(g_loss.item())

                real_validity = discriminator(imgs, labels)
                d_real_loss = criterion(real_validity, real)

                fake_validity = discriminator(gen_imgs.detach(), labels)
                d_fake_loss = criterion(fake_validity, fake)

                d_loss = (d_real_loss + d_fake_loss) / 2
                val_d_losses.append(d_loss.item())
            
            avg_val_g_loss = sum(val_g_losses) / len(val_g_losses)
            avg_val_d_loss = sum(val_d_losses) / len(val_d_losses)
            print(f"Epoch {epoch}/{epochs} - Generator Val Loss: {avg_val_g_loss:.4f}, Discriminator Val Loss: {avg_val_d_loss:.4f}")

        # Early stopping
        combined_val_loss = (avg_val_g_loss + avg_val_d_loss) / 2
        if combined_val_loss < min_val_loss:
            min_val_loss = combined_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

        # Step the schedulers to adjust the learning rates
        scheduler_G.step()
        scheduler_D.step()

        # Save generated samples for visualization
        save_image(gen_imgs.data, f"outputs/images/{epoch}.png", nrow=10, normalize=True)
        
        # Save model checkpoints
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'scheduler_G_state_dict': scheduler_G.state_dict(),
            'scheduler_D_state_dict': scheduler_D.state_dict(),
            }
        torch.save(checkpoint, f"checkpoint/cgan_checkpoint_{epoch}.pth")

def main(args):
    # Setup device, models, optimizers, and loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, val_dataloader = get_dataloader(batch_size=args.batch_size, train_size=args.train_size, seed=1910)

    generator = Generator(img_shape=args.img_shape[-1], latent_dim=args.latent_dim, n_classes=args.n_classes).to(device)
    discriminator = Discriminator(img_shape=args.img_shape[-1], n_classes=args.n_classes).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr)

    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.9)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=10, gamma=0.9)

    criterion = nn.BCELoss()

    # Find the latest checkpoint in the 'checkpoints' folder
    checkpoint_files = sorted(glob.glob("checkpoint/cgan_checkpoint_*.path"), key=os.path.getmtime)
    start_epoch = 0

    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        print(f"Loading checkpoint from {latest_checkpoint}...")
        checkpoint = torch.load(latest_checkpoint)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")

    # Train the models
    train(generator, discriminator, criterion, optimizer_G, scheduler_G, optimizer_D, scheduler_D, 
          train_dataloader, val_dataloader, device, args.epochs, args.latent_dim, start_epoch=start_epoch, patience=args.patience)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=64, help="size of each image batch")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--img_shape", type=tuple, default=(1, 64, 64), help="size of image dimensions")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--train_size", type=float, default=0.8, help="percentage of training data, the rest will be used for evaluation")
    parser.add_argument("--patience", type=int, default=20, help="early stopping patience")
    args = parser.parse_args()

    main(args)
