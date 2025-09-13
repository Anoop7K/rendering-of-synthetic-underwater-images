import torch
import argparse
import cv2
import numpy as np
from models.generator import UNetGenerator

parser = argparse.ArgumentParser()
parser.add_argument("--weights", required=True, help="path to generator weights")
parser.add_argument("--groundTruth", required=True, help="clean image path")
parser.add_argument("--jerlov", required=True, help="jerlov image path")
parser.add_argument("--output", required=True, help="output path")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load images
gt = cv2.imread(args.groundTruth)
jerlov = cv2.imread(args.jerlov)

gt = torch.from_numpy(gt.transpose(2, 0, 1)).float() / 255.0
jerlov = torch.from_numpy(jerlov.transpose(2, 0, 1)).float() / 255.0

input_cond = torch.cat([gt, jerlov], dim=0).unsqueeze(0).to(device)

# Load generator
generator = UNetGenerator(in_channels=6, out_channels=3).to(device)
generator.load_state_dict(torch.load(args.weights, map_location=device))
generator.eval()

with torch.no_grad():
    fake_under = generator(input_cond).squeeze().cpu().numpy().transpose(1, 2, 0)

cv2.imwrite(args.output, (fake_under * 255).astype(np.uint8))
print(f"Saved result at {args.output}")
