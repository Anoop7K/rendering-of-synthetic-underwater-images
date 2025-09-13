import os
import cv2
import torch

def save_sample(gt, jerlov, real, fake, epoch, results_dir):
    gt = (gt[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    jerlov = (jerlov[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    real = (real[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    fake = (fake[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")

    out_dir = os.path.join(results_dir, "samples")
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(f"{out_dir}/epoch{epoch+1}_gt.png", gt)
    cv2.imwrite(f"{out_dir}/epoch{epoch+1}_jerlov.png", jerlov)
    cv2.imwrite(f"{out_dir}/epoch{epoch+1}_real.png", real)
    cv2.imwrite(f"{out_dir}/epoch{epoch+1}_fake.png", fake)
