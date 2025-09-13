import torch
from torch.utils.data import Dataset
import cv2
import os
import pandas as pd

class UnderwaterDataset(Dataset):
    def __init__(self, csv_file, groundtruth_dir, jerlov_dir, input_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.gt_dir = groundtruth_dir
        self.jerlov_dir = jerlov_dir
        self.input_dir = input_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        gt_path = os.path.join(self.gt_dir, self.annotations.iloc[idx, 0])
        jerlov_path = os.path.join(self.jerlov_dir, self.annotations.iloc[idx, 1])
        input_path = os.path.join(self.input_dir, self.annotations.iloc[idx, 2])

        gt = cv2.imread(gt_path)
        jerlov = cv2.imread(jerlov_path)
        input_img = cv2.imread(input_path)

        gt = torch.from_numpy(gt.transpose(2, 0, 1)).float() / 255.0
        jerlov = torch.from_numpy(jerlov.transpose(2, 0, 1)).float() / 255.0
        input_img = torch.from_numpy(input_img.transpose(2, 0, 1)).float() / 255.0

        return gt, jerlov, input_img
