"""Evaluation metrics for the underwater image generator.

Covers the metrics a conditional-GAN image-to-image model is normally
judged on:
    - PSNR            : pixel-level fidelity to the target image
    - SSIM             : structural/perceptual similarity
    - L1 / MAE         : the same pixel loss used during training
    - Perceptual loss  : VGG16 feature distance (style/content fidelity)
    - Adversarial loss : how well the generator fools the discriminator
    - Discriminator accuracy : how well D tells real/fake apart
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as sk_ssim

try:
    from torchvision.models import vgg16, VGG16_Weights
except ImportError:
    from torchvision.models import vgg16
    VGG16_Weights = None

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def psnr(fake, real, max_val=1.0):
    """Peak Signal-to-Noise Ratio, averaged over the batch. Inputs: (B,C,H,W) in [0, max_val]."""
    mse = F.mse_loss(fake, real, reduction="none").mean(dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    return (20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)).mean().item()


def ssim(fake, real):
    """Structural Similarity Index, averaged over the batch. Inputs: (B,C,H,W) in [0, 1]."""
    fake_np = fake.detach().cpu().numpy()
    real_np = real.detach().cpu().numpy()
    scores = []
    for f_img, r_img in zip(fake_np, real_np):
        f_img = np.transpose(f_img, (1, 2, 0))
        r_img = np.transpose(r_img, (1, 2, 0))
        scores.append(sk_ssim(r_img, f_img, channel_axis=2, data_range=1.0))
    return float(np.mean(scores))


def l1(fake, real):
    """Mean absolute error, the same pixel loss used during training."""
    return F.l1_loss(fake, real).item()


class PerceptualLoss(nn.Module):
    """VGG16 feature-space distance between the generated and target image."""

    def __init__(self, device, layers=("3", "8", "15", "22")):
        super().__init__()
        if VGG16_Weights is not None:
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        else:
            vgg = vgg16(pretrained=True).features
        self.vgg = vgg.to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.layers = set(layers)
        self.mean = _IMAGENET_MEAN.to(device)
        self.std = _IMAGENET_STD.to(device)

    def _normalize(self, x):
        return (x - self.mean) / self.std

    def _extract(self, x):
        x = self._normalize(x)
        feats = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                feats.append(x)
        return feats

    @torch.no_grad()
    def __call__(self, fake, real):
        fake_feats = self._extract(fake)
        real_feats = self._extract(real)
        loss = sum(F.l1_loss(f, r) for f, r in zip(fake_feats, real_feats))
        return (loss / len(fake_feats)).item()


def adversarial_loss_g(discriminator, gt, jerlov, fake):
    """BCE loss of the generator against an all-real target (lower = better at fooling D)."""
    criterion = nn.BCEWithLogitsLoss()
    pred_fake = discriminator(torch.cat([gt, jerlov, fake], dim=1))
    valid = torch.ones_like(pred_fake)
    return criterion(pred_fake, valid).item()


def discriminator_metrics(discriminator, gt, jerlov, real, fake):
    """BCE loss and binary accuracy of the discriminator on a real/fake pair."""
    criterion = nn.BCEWithLogitsLoss()

    pred_real = discriminator(torch.cat([gt, jerlov, real], dim=1))
    pred_fake = discriminator(torch.cat([gt, jerlov, fake], dim=1))

    valid = torch.ones_like(pred_real)
    zeros = torch.zeros_like(pred_fake)

    loss_d = 0.5 * (criterion(pred_real, valid) + criterion(pred_fake, zeros))

    real_acc = (torch.sigmoid(pred_real) > 0.5).float().mean()
    fake_acc = (torch.sigmoid(pred_fake) <= 0.5).float().mean()
    accuracy = 0.5 * (real_acc + fake_acc)

    return loss_d.item(), accuracy.item()
