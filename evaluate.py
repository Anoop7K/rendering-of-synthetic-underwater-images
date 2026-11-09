"""Evaluate a trained generator (and, optionally, discriminator) on the dataset.

Reports the metrics a conditional-GAN image synthesis model is normally judged on:
PSNR, SSIM, L1, VGG perceptual loss, adversarial loss, and discriminator accuracy.

Usage:
    python evaluate.py --generator results/checkpoints/G_epoch200.pth
    python evaluate.py --generator results/checkpoints/G_epoch200.pth \
                        --discriminator results/checkpoints/D_epoch200.pth \
                        --output results/metrics.json
"""

import os
import json
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import UnderwaterDataset
from models.generator import UNetGenerator
from models.discriminator import PatchGANDiscriminator
from utils.metrics import psnr, ssim, l1, PerceptualLoss, adversarial_loss_g, discriminator_metrics


def evaluate(config, generator_path, discriminator_path=None, batch_size=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = UnderwaterDataset(
        csv_file=config["csv_file"],
        groundtruth_dir=config["groundtruth_dir"],
        jerlov_dir=config["jerlov_dir"],
        input_dir=config["input_dir"],
    )
    dataloader = DataLoader(dataset, batch_size=batch_size or config["batch_size"], shuffle=False)

    generator = UNetGenerator(in_channels=6, out_channels=3).to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()

    discriminator = None
    if discriminator_path:
        discriminator = PatchGANDiscriminator(in_channels=9).to(device)
        discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))
        discriminator.eval()

    perceptual_loss_fn = PerceptualLoss(device)

    totals = {"psnr": 0.0, "ssim": 0.0, "l1": 0.0, "perceptual": 0.0}
    if discriminator is not None:
        totals.update({"adversarial_g": 0.0, "d_loss": 0.0, "d_accuracy": 0.0})
    num_batches = 0

    with torch.no_grad():
        for gt, jerlov, real_under in tqdm(dataloader, desc="Evaluating"):
            gt, jerlov, real_under = gt.to(device), jerlov.to(device), real_under.to(device)
            input_cond = torch.cat([gt, jerlov], dim=1)
            fake_under = generator(input_cond)

            totals["psnr"] += psnr(fake_under, real_under)
            totals["ssim"] += ssim(fake_under, real_under)
            totals["l1"] += l1(fake_under, real_under)
            totals["perceptual"] += perceptual_loss_fn(fake_under, real_under)

            if discriminator is not None:
                totals["adversarial_g"] += adversarial_loss_g(discriminator, gt, jerlov, fake_under)
                d_loss, d_acc = discriminator_metrics(discriminator, gt, jerlov, real_under, fake_under)
                totals["d_loss"] += d_loss
                totals["d_accuracy"] += d_acc

            num_batches += 1

    return {k: v / num_batches for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--generator", required=True, help="path to generator weights")
    parser.add_argument("--discriminator", default=None, help="path to discriminator weights (optional)")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output", default=None, help="path to save metrics as JSON")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    metrics = evaluate(config, args.generator, args.discriminator, args.batch_size)

    print("\nEvaluation results:")
    for name, value in metrics.items():
        print(f"  {name:12s}: {value:.4f}")

    output_path = args.output or os.path.join(config["results_dir"], "metrics.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {output_path}")


if __name__ == "__main__":
    main()
