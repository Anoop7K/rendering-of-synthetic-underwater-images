import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn

from data.dataset import UnderwaterDataset
from models.generator import UNetGenerator
from models.discriminator import PatchGANDiscriminator
from utils.visualize import save_sample

# Load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset & Dataloader
dataset = UnderwaterDataset(
    csv_file=config["csv_file"],
    groundtruth_dir=config["groundtruth_dir"],
    jerlov_dir=config["jerlov_dir"],
    input_dir=config["input_dir"]
)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# Models
generator = UNetGenerator(in_channels=6, out_channels=3).to(device)
discriminator = PatchGANDiscriminator(in_channels=9).to(device)  # GT+Jerlov+Output

# Losses & Optimizers
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()

optimizer_G = optim.Adam(generator.parameters(), lr=config["lr"], betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=config["lr"], betas=(0.5, 0.999))

os.makedirs(config["results_dir"], exist_ok=True)
os.makedirs(os.path.join(config["results_dir"], "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(config["results_dir"], "samples"), exist_ok=True)

# Training
for epoch in range(config["epochs"]):
    loop = tqdm(dataloader, leave=False)
    for i, (gt, jerlov, real_under) in enumerate(loop):
        gt, jerlov, real_under = gt.to(device), jerlov.to(device), real_under.to(device)

        # --------------------
        # Train Generator
        # --------------------
        optimizer_G.zero_grad()
        input_cond = torch.cat([gt, jerlov], dim=1)
        fake_under = generator(input_cond)

        pred_fake = discriminator(torch.cat([gt, jerlov, fake_under], dim=1))
        valid = torch.ones_like(pred_fake, device=device)

        loss_GAN = criterion_GAN(pred_fake, valid)
        loss_L1 = criterion_L1(fake_under, real_under) * config["lambda_L1"]
        loss_G = loss_GAN + loss_L1
        loss_G.backward()
        optimizer_G.step()

        # --------------------
        # Train Discriminator
        # --------------------
        optimizer_D.zero_grad()
        pred_real = discriminator(torch.cat([gt, jerlov, real_under], dim=1))
        loss_real = criterion_GAN(pred_real, valid)

        pred_fake = discriminator(torch.cat([gt, jerlov, fake_under.detach()], dim=1))
        fake = torch.zeros_like(pred_fake, device=device)
        loss_fake = criterion_GAN(pred_fake, fake)

        loss_D = (loss_real + loss_fake) * 0.5
        loss_D.backward()
        optimizer_D.step()

        loop.set_description(f"Epoch[{epoch+1}/{config['epochs']}]")
        loop.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())

    # Save checkpoint
    if (epoch + 1) % config["save_freq"] == 0:
        torch.save(generator.state_dict(),
                   f"{config['results_dir']}/checkpoints/G_epoch{epoch+1}.pth")
        torch.save(discriminator.state_dict(),
                   f"{config['results_dir']}/checkpoints/D_epoch{epoch+1}.pth")

        save_sample(gt, jerlov, real_under, fake_under, epoch, config["results_dir"])
